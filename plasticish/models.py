"""
Core neural models implementing bio-inspired plasticity mechanisms.

This module contains the fundamental building blocks of the Plastic Brain
architecture, which addresses the stability-plasticity dilemma through
a modular, pluggable multi-layer design with neuromodulation.

Key Components:
- PretrainedEyes: Frozen feature extractor (visual cortex)
- NeuromodulatedBlock: Pluggable plastic layer with local learning rules
- EpisodicMemoryBank: KNN-based hippocampal memory
- PlasticBrain: Full network with pluggable layer support
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional, List, Literal


class PretrainedEyes(nn.Module):
    """
    Visual feature extractor using frozen pretrained ResNet18.
    
    This component serves as the sensory input layer, analogous to the
    visual cortex in biological systems. By freezing weights, we ensure
    stable feature representations that don't drift during continual learning.
    
    Architecture:
    - ResNet18 (ImageNet pretrained)
    - Final FC layer removed
    - 512-dimensional output
    
    Args:
        device: torch device to place the model on
    
    Returns:
        512-dimensional feature vectors
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.out_dim = 512
        
        weights = ResNet18_Weights.IMAGENET1K_V1
        base = resnet18(weights=weights)
        self.model = nn.Sequential(*list(base.children())[:-1]).to(device)
        self.model.eval()
        
        # Freeze all parameters - this is our "sensory apparatus"
        for p in self.model.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        """Extract visual features."""
        with torch.no_grad():
            return self.model(x).squeeze()


class NeuromodulatedBlock(nn.Module):
    """
    A pluggable neuromodulated plastic layer implementing local learning rules.
    
    This layer combines ideas from:
    - Sparse coding (Olshausen & Field, 1996)
    - Hebbian learning with neuromodulation
    - Synaptic consolidation and maturity
    - Neurogenesis (neuron recycling)
    
    Architecture:
        input -> normalize -> encoder -> sparse_topk -> decoder -> gate -> output
                                          (concepts)      (imagination)
    
    Key Features:
    - Local plasticity: dW = Reward * Post * (Pre - W)
    - Maturity-based protection: Frequently used neurons become "mature" and protected
    - Neurogenesis: Dead neurons are recycled to learn new concepts
    - Gating: LayerScale-style output modulation
    - Mode-dependent behavior: inference, plastic, panic
    
    Args:
        in_dim: Input feature dimension
        hidden_dim: Number of hidden neurons (concept space)
        sparsity_k: Number of active neurons (sparse code size)
        lr: Base learning rate for plasticity
        device: torch device
    
    Training Modes:
        - "inference": No learning, just forward pass
        - "plastic": Normal Hebbian plasticity with maturity protection
        - "panic": Aggressive adaptation with boosted signals and forced neurogenesis
    """
    
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 4096,
        sparsity_k: int = 128,
        lr: float = 0.04,
        device: str = 'cpu'
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.k = sparsity_k
        self.lr = lr
        self.device = device
        
        # === SYNAPSES (Encoder/Decoder pair) ===
        # Encoder: Projects input to sparse concept space
        self.encoder = nn.Linear(in_dim, hidden_dim, bias=False)
        # Decoder: Reconstructs input from sparse concepts (associative recall)
        self.decoder = nn.Linear(hidden_dim, in_dim, bias=False)
        
        # Initialize encoder orthogonally for diverse feature detection
        nn.init.orthogonal_(self.encoder.weight)
        # Decoder starts at zero - learns through plasticity
        nn.init.zeros_(self.decoder.weight)
        
        # === GATING (LayerScale-style) ===
        # Modulates contribution of this layer to residual stream
        self.gate = nn.Parameter(torch.ones(in_dim, device=device) * 0.01)
        
        # === METADATA (Not learned by gradient descent) ===
        # Frequency tracking: How often each neuron fires
        self.register_buffer("freq_count", torch.zeros(hidden_dim, device=device))
        # Maturity mask: Protected neurons (1 = mature, 0 = plastic)
        self.register_buffer("maturity_mask", torch.zeros(hidden_dim, device=device))
        
        # Cache for plasticity step
        self.cache = None
        
        self.to(device)
    
    def consolidate(self) -> int:
        """
        Mark frequently active neurons as mature (protected from neurogenesis).
        
        This implements synaptic consolidation - neurons that have proven useful
        become protected from being recycled during neurogenesis.
        
        Returns:
            Number of mature neurons after consolidation
        """
        threshold = self.freq_count.mean()
        new_mature = (self.freq_count > threshold).float()
        self.maturity_mask = torch.max(self.maturity_mask, new_mature)
        self.freq_count.zero_()
        return int(self.maturity_mask.sum().item())
    
    def neurogenesis(self, force: bool = False):
        """
        Recycle dead/weak neurons to learn new concepts.
        
        Dead neurons (low activity) are reinitialized with random weights,
        giving the network capacity to learn new patterns without growing.
        
        Args:
            force: If True, use aggressive threshold (for panic mode)
        """
        threshold = 0.05 if force else 0.01
        activity = self.freq_count / (self.freq_count.max() + 1e-6)
        dead = activity < threshold
        
        # Only recycle non-mature neurons (unless super-storm force)
        targets = dead & (self.maturity_mask == 0)
        indices = targets.nonzero(as_tuple=True)[0]
        
        if len(indices) > 0:
            with torch.no_grad():
                # Reinitialize dead neurons with random directions
                new_w = torch.randn(len(indices), self.in_dim, device=self.device)
                self.encoder.weight.data[indices] = F.normalize(new_w, p=2, dim=1)
                self.decoder.weight.data[:, indices] = 0
                self.freq_count[indices] = 0
    
    def forward(
        self,
        x: torch.Tensor,
        mode: Literal["inference", "plastic", "panic"] = "inference"
    ) -> torch.Tensor:
        """
        Forward pass with mode-dependent behavior.
        
        Architecture: input -> norm -> encode -> sparse -> decode -> gate -> residual
        
        Args:
            x: Input features [batch_size, in_dim]
            mode: Operating mode
                - "inference": No learning, standard forward
                - "plastic": Learning enabled with maturity protection
                - "panic": Boosted signals + aggressive learning
        
        Returns:
            Output features [batch_size, in_dim] (residual added to input)
        """
        # Normalize input for stable learning
        x_norm = F.normalize(x, p=2, dim=1)
        
        # Project to high-dimensional concept space
        hidden = self.encoder(x_norm)
        
        # Signal boosting in panic mode (helps weak signals reach threshold)
        if mode == "panic":
            hidden = hidden * 2.0
        
        # k-Winner-Take-All sparsity
        top_val, top_idx = torch.topk(hidden, self.k, dim=1)
        mask = torch.zeros_like(hidden).scatter_(1, top_idx, 1.0)
        activations = hidden * mask
        
        # Associative recall via decoder
        imagination = self.decoder(activations)
        
        # Gated residual output
        out = x + (self.gate * imagination)
        
        # Cache for plasticity step if in learning mode
        if mode in ["plastic", "panic"]:
            self.cache = {
                "x": x_norm,
                "activations": activations,
                "mask": mask,
                "mode": mode
            }
        
        return out
    
    def plasticity_step(self, reward_signal: torch.Tensor):
        """
        Apply local Hebbian learning rule with neuromodulation.
        
        Update Rule: dW = LR * Reward * (Pre - W) for active neurons
        
        The reward signal modulates learning:
        - Positive reward: Strengthen associations (attract)
        - Negative reward: Weaken associations (repel)
        
        Args:
            reward_signal: Per-sample reward [batch_size] in [-1, 1]
        """
        if self.cache is None:
            return
        
        x_norm = self.cache["x"]
        activations = self.cache["activations"]
        mask = self.cache["mask"]
        mode = self.cache["mode"]
        
        with torch.no_grad():
            # Update frequency counts (exponential moving average)
            batch_act = mask.sum(dim=0)
            self.freq_count = 0.95 * self.freq_count + 0.05 * batch_act
            
            # Modulate input by reward (Positive=Attract, Negative=Repel)
            x_mod = x_norm * reward_signal.view(-1, 1)
            
            # Maturity protection: reduce learning rate for mature neurons
            panic_factor = 0.5 if mode == "panic" else 0.99
            lr_vec = self.lr * (1.0 - panic_factor * self.maturity_mask).view(-1, 1)
            
            # Compute weight update direction
            inp_proj = (activations.t() @ x_mod) / (batch_act.view(-1, 1) + 1e-6)
            active = (batch_act > 0).view(-1, 1)
            diff = inp_proj - self.encoder.weight.data
            
            # Apply update to encoder
            self.encoder.weight.data += lr_vec * diff * active
            
            # Decoder follows encoder (slow tracking)
            target = self.encoder.weight.data.t()
            self.decoder.weight.data = 0.95 * self.decoder.weight.data + 0.05 * target
            
            # Stochastic neurogenesis
            if torch.rand(1).item() < (0.1 if mode == "panic" else 0.02):
                self.neurogenesis(force=(mode == "panic"))
        
        self.cache = None
    
    def get_stats(self) -> dict:
        """Get diagnostic statistics about this layer."""
        return {
            'n_mature': int(self.maturity_mask.sum().item()),
            'n_active': int((self.freq_count > 0.1).sum().item()),
            'gate_mean': self.gate.mean().item(),
            'freq_mean': self.freq_count.mean().item(),
            'freq_max': self.freq_count.max().item()
        }


class EpisodicMemoryBank(nn.Module):
    """
    Hippocampus-inspired episodic memory using k-Nearest Neighbors.
    
    Instead of learning weight matrices for classification, this module
    stores (key, value) pairs and performs soft voting based on similarity.
    This provides:
    - One-shot learning: New memories are instantly retrievable
    - No forgetting: Old memories persist until overwritten
    - Interpretable: Can inspect which memories influenced prediction
    
    Architecture:
        query -> normalize -> similarity -> topk -> soft_vote -> logits
    
    Args:
        input_dim: Feature dimension for keys
        memory_size: Maximum number of stored memories
        k_neighbors: Number of neighbors for voting
        num_classes: Number of output classes
        device: torch device
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        memory_size: int = 60000,
        k_neighbors: int = 50,
        num_classes: int = 10,
        device: str = 'cpu'
    ):
        super().__init__()
        self.input_dim = input_dim
        self.memory_size = memory_size
        self.k = k_neighbors
        self.num_classes = num_classes
        self.device = device
        
        # Memory storage
        self.register_buffer("keys", torch.zeros(memory_size, input_dim, device=device))
        self.register_buffer("values", torch.ones(memory_size, device=device).long() * -1)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long, device=device))
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Query memory and return class logits via soft voting.
        
        Args:
            x: Query features [batch_size, input_dim]
        
        Returns:
            logits: Class scores [batch_size, num_classes]
        """
        x_n = F.normalize(x, p=2, dim=1)
        k_n = F.normalize(self.keys, p=2, dim=1)
        scores = torch.matmul(x_n, k_n.t())
        
        # Ignore empty memory slots
        empty_mask = (self.values == -1)
        scores.masked_fill_(empty_mask.view(1, -1), -1e9)
        
        top_scores, top_indices = torch.topk(scores, self.k, dim=1)
        top_labels = self.values[top_indices]
        
        # Soft voting weighted by similarity
        logits = torch.zeros(x.shape[0], self.num_classes, device=self.device)
        for i in range(self.k):
            lbl = top_labels[:, i]
            score = top_scores[:, i]
            valid = (lbl != -1)
            if valid.sum() > 0:
                safe_lbl = lbl.clone()
                safe_lbl[~valid] = 0
                safe_score = score.clone()
                safe_score[~valid] = 0
                logits.scatter_add_(1, safe_lbl.view(-1, 1), safe_score.view(-1, 1))
        
        return logits
    
    def update(self, x: torch.Tensor, y: torch.Tensor):
        """
        Store new memories (FIFO replacement).
        
        Args:
            x: Features to store [batch_size, input_dim]
            y: Labels [batch_size]
        """
        with torch.no_grad():
            batch_size = x.shape[0]
            ptr = int(self.ptr)
            indices = torch.arange(ptr, ptr + batch_size, device=self.device) % self.memory_size
            self.keys[indices] = F.normalize(x, p=2, dim=1)
            self.values[indices] = y
            self.ptr[0] = (ptr + batch_size) % self.memory_size
    
    def clear(self):
        """Clear all memories."""
        self.keys.zero_()
        self.values.fill_(-1)
        self.ptr.zero_()
    
    def get_fill_ratio(self) -> float:
        """Get the fraction of memory slots filled."""
        return (self.values != -1).float().mean().item()


class PlasticBrain(nn.Module):
    """
    Full plastic brain with pluggable neuromodulated layers.
    
    This is the main model class that combines:
    - PretrainedEyes: Frozen visual feature extractor
    - NeuromodulatedBlock(s): Pluggable plastic layers
    - EpisodicMemoryBank: KNN-based output memory
    
    The architecture supports:
    - Dynamic layer addition/removal
    - Multiple training modes (inference, plastic, panic)
    - Self-supervised reward-based learning
    - Consolidation and neurogenesis
    
    Example:
        >>> brain = PlasticBrain(num_layers=4, device='cuda')
        >>> brain.add_layer()  # Add a 5th layer
        >>> brain.remove_layer(2)  # Remove layer at index 2
        >>> brain.train_step(images, labels, mode='plastic', memorize=True)
    
    Args:
        in_dim: Input feature dimension (from eyes)
        hidden_dim: Hidden neurons per layer
        sparsity_k: Active neurons per layer
        num_layers: Initial number of plastic layers
        memory_size: Episodic memory capacity
        k_neighbors: KNN voting neighbors
        num_classes: Number of output classes
        lr_brain: Learning rate for plastic layers
        device: torch device
    """
    
    def __init__(
        self,
        in_dim: int = 512,
        hidden_dim: int = 4096,
        sparsity_k: int = 128,
        num_layers: int = 4,
        memory_size: int = 60000,
        k_neighbors: int = 50,
        num_classes: int = 10,
        lr_brain: float = 0.04,
        device: str = 'cpu'
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.sparsity_k = sparsity_k
        self.num_classes = num_classes
        self.lr_brain = lr_brain
        self.device = device
        
        # === EYES: Pretrained feature extractor ===
        self.eye = PretrainedEyes(device=device)
        
        # === BRAIN: Pluggable plastic layers ===
        self.brain = nn.ModuleList([
            NeuromodulatedBlock(
                in_dim=in_dim,
                hidden_dim=hidden_dim,
                sparsity_k=sparsity_k,
                lr=lr_brain,
                device=device
            )
            for _ in range(num_layers)
        ])
        
        # === MEMORY: Episodic memory bank ===
        self.memory = EpisodicMemoryBank(
            input_dim=in_dim,
            memory_size=memory_size,
            k_neighbors=k_neighbors,
            num_classes=num_classes,
            device=device
        )
        
        # Batch normalization before memory lookup
        self.bn = nn.BatchNorm1d(in_dim).to(device)
        
    @property
    def num_layers(self) -> int:
        """Current number of plastic layers."""
        return len(self.brain)
    
    # === LAYER MANAGEMENT (Pluggable Architecture) ===
    
    def add_layer(
        self,
        index: Optional[int] = None,
        hidden_dim: Optional[int] = None,
        sparsity_k: Optional[int] = None,
        lr: Optional[float] = None
    ) -> int:
        """
        Add a new plastic layer.
        
        Args:
            index: Position to insert (None = append at end)
            hidden_dim: Hidden size (None = use default)
            sparsity_k: Sparsity (None = use default)
            lr: Learning rate (None = use default)
        
        Returns:
            Index of the new layer
        """
        new_layer = NeuromodulatedBlock(
            in_dim=self.in_dim,
            hidden_dim=hidden_dim or self.hidden_dim,
            sparsity_k=sparsity_k or self.sparsity_k,
            lr=lr or self.lr_brain,
            device=self.device
        )
        
        if index is None:
            self.brain.append(new_layer)
            return len(self.brain) - 1
        else:
            self.brain.insert(index, new_layer)
            return index
    
    def remove_layer(self, index: int) -> NeuromodulatedBlock:
        """
        Remove a plastic layer.
        
        Args:
            index: Layer index to remove
        
        Returns:
            The removed layer (for potential re-use)
        """
        if len(self.brain) <= 1:
            raise ValueError("Cannot remove the last layer")
        return self.brain.pop(index)
    
    def replace_layer(
        self,
        index: int,
        new_layer: Optional[NeuromodulatedBlock] = None
    ) -> NeuromodulatedBlock:
        """
        Replace a layer with a new one.
        
        Args:
            index: Layer index to replace
            new_layer: Replacement layer (None = create fresh default)
        
        Returns:
            The old layer that was replaced
        """
        old_layer = self.brain[index]
        
        if new_layer is None:
            new_layer = NeuromodulatedBlock(
                in_dim=self.in_dim,
                hidden_dim=self.hidden_dim,
                sparsity_k=self.sparsity_k,
                lr=self.lr_brain,
                device=self.device
            )
        
        self.brain[index] = new_layer
        return old_layer
    
    def freeze_layer(self, index: int):
        """Freeze a layer (disable plasticity)."""
        layer = self.brain[index]
        layer.maturity_mask.fill_(1.0)
    
    def unfreeze_layer(self, index: int):
        """Unfreeze a layer (enable plasticity)."""
        layer = self.brain[index]
        layer.maturity_mask.zero_()
    
    # === FORWARD PASS ===
    
    def forward(
        self,
        x: torch.Tensor,
        mode: Literal["inference", "plastic", "panic"] = "inference"
    ) -> torch.Tensor:
        """
        Forward pass through eyes and brain layers.
        
        Args:
            x: Input images [batch_size, C, H, W]
            mode: Operating mode for plastic layers
        
        Returns:
            Processed features [batch_size, in_dim]
        """
        # Extract visual features (frozen)
        with torch.no_grad():
            feat = self.eye(x)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
        
        # Process through plastic layers
        thought = feat
        for layer in self.brain:
            thought = layer(thought, mode=mode)
        
        # Normalize for memory lookup
        thought = self.bn(thought)
        return thought
    
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get class predictions for input images.
        
        Args:
            x: Input images [batch_size, C, H, W]
        
        Returns:
            Predicted class indices [batch_size]
        """
        thought = self.forward(x, mode="inference")
        logits = self.memory(thought)
        return logits.argmax(dim=1)
    
    # === TRAINING ===
    
    def train_step(
        self,
        x: torch.Tensor,
        y: torch.Tensor,
        mode: Literal["plastic", "panic"] = "plastic",
        memorize: bool = False
    ) -> float:
        """
        Perform one training step with self-supervised reward.
        
        The reward signal is computed from memory predictions:
        - Correct prediction: reward = +1 (reinforce)
        - Wrong prediction: reward = -0.2 (weak punishment)
        - Fresh memory (< 100 samples): reward = +1 (encourage all)
        - Panic mode: reward = +1 always (adapt to everything)
        
        Args:
            x: Input images [batch_size, C, H, W]
            y: Labels [batch_size]
            mode: Training mode ("plastic" or "panic")
            memorize: Whether to store in episodic memory
        
        Returns:
            Batch accuracy
        """
        thought = self.forward(x, mode=mode)
        
        # Self-supervised reward from memory
        logits = self.memory(thought)
        preds = logits.argmax(dim=1)
        
        # Compute reward signal
        if self.memory.ptr < 100:
            # Fresh memory: encourage all learning
            reward = torch.ones_like(preds).float()
        else:
            is_correct = (preds == y).float()
            reward = torch.where(is_correct == 1.0, 1.0, torch.tensor(-0.2, device=self.device))
        
        # Panic mode: just adapt, assume everything is rewarding
        if mode == "panic":
            reward = torch.ones_like(reward)
        
        # Apply plasticity to all layers
        for layer in self.brain:
            layer.plasticity_step(reward)
        
        # Optionally store in memory
        if memorize:
            self.memory.update(thought.detach(), y)
        
        # Return accuracy
        acc = (preds == y).float().mean().item()
        return acc
    
    def consolidate(self) -> dict:
        """
        Consolidate all layers (mark active neurons as mature).
        
        Returns:
            Dictionary with mature neuron counts per layer
        """
        stats = {}
        for i, layer in enumerate(self.brain):
            n_mature = layer.consolidate()
            stats[f'layer_{i}'] = n_mature
        return stats
    
    # === EVALUATION ===
    
    def evaluate(self, loader: torch.utils.data.DataLoader) -> float:
        """
        Evaluate accuracy on a data loader.
        
        Args:
            loader: DataLoader to evaluate on
        
        Returns:
            Accuracy as percentage (0-100)
        """
        self.eval()
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in loader:
                x, y = x.to(self.device), y.to(self.device)
                preds = self.predict(x)
                correct += preds.eq(y).sum().item()
                total += y.size(0)
        
        return 100.0 * correct / total if total > 0 else 0.0
    
    def get_stats(self) -> dict:
        """Get comprehensive statistics about the brain state."""
        stats = {
            'num_layers': self.num_layers,
            'memory_fill': self.memory.get_fill_ratio(),
            'layers': {}
        }
        
        for i, layer in enumerate(self.brain):
            stats['layers'][f'layer_{i}'] = layer.get_stats()
        
        return stats


# === LEGACY SUPPORT ===
# Keep the old TriarchicBrain for backwards compatibility

class TriarchicBrain(nn.Module):
    """
    Legacy: Three-layer adaptive neural architecture for continual learning.
    
    NOTE: This class is kept for backwards compatibility. For new projects,
    use PlasticBrain which offers pluggable multi-layer support.
    
    This architecture solves the stability-plasticity dilemma through three
    interacting components:
    
    1. **BASE (Structural Plasticity/Neurogenesis)**:
       - Sparse high-dimensional projection (512 → 8192)
       - Dynamic neuron generation when novel inputs detected
       - k-Winner-Take-All sparsity for orthogonal representations
    
    2. **CORE (Synaptic Consolidation)**:
       - Slow-learning weight matrix with importance-based protection
       - Protected weights resist overwriting during new task learning
    
    3. **OVERLAY (Fast Context Adaptation)**:
       - Fast-learning, fast-decaying weight matrix
       - Handles temporary distributional shifts without permanent changes
    """
    
    def __init__(self, in_size=512, hidden_size=8192, out_size=10, sparsity_k=32, device='cpu'):
        super().__init__()
        
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.sparsity_k = sparsity_k
        self.device = device
        
        # === LAYER 1: BASE (Structural Plasticity) ===
        self.projection = torch.randn(hidden_size, in_size, device=device) * 0.01
        self.projection = F.normalize(self.projection, p=2, dim=1)
        self.neuron_utility = torch.zeros(hidden_size, device=device)
        
        # === LAYER 2: CORE (Long-Term Memory with Consolidation) ===
        self.w_core = torch.zeros(out_size, hidden_size, device=device)
        self.importance = torch.zeros(out_size, hidden_size, device=device)
        
        # === LAYER 3: OVERLAY (Short-Term Context) ===
        self.w_context = torch.zeros(out_size, hidden_size, device=device)
        
        # Hyperparameters
        self.neurogenesis_threshold = 0.85
        self.utility_decay = 0.995
        self.context_lr = 0.5
        self.context_decay = 0.8
        self.core_lr = 0.05
        self.consolidation_rate = 2.0
    
    def get_sparse_h(self, x):
        """Compute sparse hidden representation with dynamic neurogenesis."""
        hidden = torch.mm(x, self.projection.t())
        
        max_val, _ = torch.max(hidden, dim=1)
        poorly_represented = (max_val < self.neurogenesis_threshold)
        
        if poorly_represented.any():
            noisy_util = self.neuron_utility + torch.rand_like(self.neuron_utility) * 0.01
            n_needed = poorly_represented.sum().item()
            _, victim_indices = torch.topk(noisy_util, k=n_needed, largest=False)
            
            self.projection[victim_indices] = x[poorly_represented].detach()
            self.projection = F.normalize(self.projection, p=2, dim=1)
            
            self.neuron_utility[victim_indices] = 1.0
            self.w_core[:, victim_indices] = 0.0
            self.w_context[:, victim_indices] = 0.0
            self.importance[:, victim_indices] = 0.0
            
            hidden = torch.mm(x, self.projection.t())
        
        vals, indices = torch.topk(hidden, k=self.sparsity_k, dim=1)
        threshold = vals[:, -1].unsqueeze(1)
        sparse_h = F.relu(hidden - threshold)
        
        self.neuron_utility *= self.utility_decay
        fired_mask = (sparse_h > 0).float().sum(dim=0)
        self.neuron_utility += (fired_mask > 0).float()
        
        return F.normalize(sparse_h, p=2, dim=1)
    
    def forward(self, x):
        """Forward pass combining Core and Context signals."""
        sparse_h = self.get_sparse_h(x)
        core_signal = torch.mm(sparse_h, self.w_core.t())
        context_signal = torch.mm(sparse_h, self.w_context.t())
        return core_signal + context_signal, sparse_h
    
    def update_batch(self, x, targets):
        """Perform one learning step with triarchic update rules."""
        sparse_h = self.get_sparse_h(x)
        
        core_logits = torch.mm(sparse_h, self.w_core.t())
        context_logits = torch.mm(sparse_h, self.w_context.t())
        total_logits = core_logits + context_logits
        probs = torch.softmax(total_logits, dim=1)
        
        target_oh = F.one_hot(targets, num_classes=self.out_size).float()
        error = target_oh - probs
        delta = torch.mm(error.t(), sparse_h) / x.shape[0]
        
        self.w_context += self.context_lr * delta
        self.w_context *= self.context_decay
        
        brake = torch.clamp(self.importance * 2.0, 0.0, 1.0)
        effective_delta = delta * (1.0 - brake)
        self.w_core += self.core_lr * effective_delta
        self.w_core = F.normalize(self.w_core, p=2, dim=1)
        
        preds = torch.argmax(probs, dim=1)
        correct_mask = (preds == targets).float().unsqueeze(1)
        weighted_activity = sparse_h * correct_mask
        imp_update = torch.mm(target_oh.t(), weighted_activity) / x.shape[0]
        self.importance += self.consolidation_rate * imp_update
        self.importance = torch.clamp(self.importance, 0.0, 1.0)
        
        acc = (preds == targets).float().mean().item()
        return acc
    
    def get_memory_accuracy(self, x, targets):
        """Test memory retention using only CORE weights."""
        with torch.no_grad():
            sparse_h = self.get_sparse_h(x)
            logits = torch.mm(sparse_h, self.w_core.t())
            acc = (torch.argmax(logits, dim=1) == targets).float().mean().item()
        return acc
    
    def get_stats(self):
        """Get diagnostic statistics about the brain state."""
        return {
            'n_active_neurons': (self.neuron_utility > 0.1).sum().item(),
            'context_energy': self.w_context.abs().mean().item(),
            'core_energy': self.w_core.abs().mean().item(),
            'mean_importance': self.importance.mean().item(),
            'max_importance': self.importance.max().item()
        }
