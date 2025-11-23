"""
Core neural models implementing bio-inspired plasticity mechanisms.

This module contains the fundamental building blocks of the Triarchic Brain
architecture, which addresses the stability-plasticity dilemma through
a three-layer design.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18, ResNet18_Weights


class PretrainedEyes(nn.Module):
    """
    Visual feature extractor using frozen pretrained ResNet18.
    
    This component serves as the sensory input layer, analogous to the
    visual cortex in biological systems. By freezing weights, we ensure
    stable feature representations that don't drift during continual learning.
    
    Architecture:
    - ResNet18 (ImageNet pretrained)
    - Final FC layer replaced with Identity
    - L2-normalized 512-dimensional output
    
    Args:
        device: torch device to place the model on
    
    Returns:
        Normalized 512-dimensional feature vectors
    """
    
    def __init__(self, device='cpu'):
        super().__init__()
        self.device = device
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT).to(device)
        self.model.fc = nn.Identity()
        self.model.eval()
        
        # Freeze all parameters - this is our "sensory apparatus"
        for p in self.model.parameters():
            p.requires_grad = False
    
    def forward(self, x):
        """Extract and normalize visual features."""
        return F.normalize(self.model(x), p=2, dim=1)


class TriarchicBrain(nn.Module):
    """
    Three-layer adaptive neural architecture for continual learning.
    
    This architecture solves the stability-plasticity dilemma through three
    interacting components:
    
    1. **BASE (Structural Plasticity/Neurogenesis)**:
       - Sparse high-dimensional projection (512 → 8192)
       - Dynamic neuron generation when novel inputs detected
       - k-Winner-Take-All sparsity for orthogonal representations
       - Inspired by: Oja's Rule, Sparse Coding (Olshausen & Field, 1996)
    
    2. **CORE (Synaptic Consolidation)**:
       - Slow-learning weight matrix with importance-based protection
       - Synapses that contribute to correct predictions become "consolidated"
       - Protected weights resist overwriting during new task learning
       - Inspired by: Elastic Weight Consolidation (Kirkpatrick et al., 2017)
    
    3. **OVERLAY (Fast Context Adaptation)**:
       - Fast-learning, fast-decaying weight matrix
       - Handles temporary distributional shifts without permanent changes
       - Decays exponentially (half-life ~5 batches)
       - Inspired by: Fast Weights (Ba et al., 2016; Hinton & Plaut, 1987)
    
    Mathematical Formulation:
        h = sparse(projection @ x)              # Base: Sparse activation
        y = (W_core + W_context) @ h            # Combined output
        
        W_context += α_fast * Δw                # Fast update
        W_context *= decay_fast                 # Exponential decay
        
        W_core += α_slow * Δw * (1 - importance) # Protected update
        importance += β * correct_predictions    # Consolidation
    
    Args:
        in_size: Input feature dimension (typically 512 from ResNet)
        hidden_size: Sparse projection dimension (e.g., 8192)
        out_size: Number of output classes
        sparsity_k: Number of active neurons per sample (e.g., 32)
        device: torch device
    
    Key Hyperparameters:
        - Neurogenesis threshold: 0.85 (activation threshold for "novel" detection)
        - Context learning rate: 0.5 (fast adaptation)
        - Context decay: 0.8 per step (half-life ~3 steps)
        - Core learning rate: 0.05 (slow consolidation)
        - Consolidation rate: 2.0 (importance accumulation)
    
    References:
        - Liu, Z., et al. (2025). Intelligence Foundation Model. arXiv:2511.10119
        - Kirkpatrick, J., et al. (2017). Overcoming catastrophic forgetting in neural networks.
        - Ba, J., et al. (2016). Using Fast Weights to Attend to the Recent Past.
        - Olshausen, B. A., & Field, D. J. (1996). Emergence of simple-cell receptive fields.
    """
    
    def __init__(self, in_size=512, hidden_size=8192, out_size=10, sparsity_k=32, device='cpu'):
        super().__init__()
        
        self.in_size = in_size
        self.hidden_size = hidden_size
        self.out_size = out_size
        self.sparsity_k = sparsity_k
        self.device = device
        
        # === LAYER 1: BASE (Structural Plasticity) ===
        # Random orthogonal projection for sparse coding
        self.projection = torch.randn(hidden_size, in_size, device=device) * 0.01
        self.projection = F.normalize(self.projection, p=2, dim=1)
        
        # Utility tracker: measures neuron usage over time
        # Low utility neurons are candidates for neurogenesis
        self.neuron_utility = torch.zeros(hidden_size, device=device)
        
        # === LAYER 2: CORE (Long-Term Memory with Consolidation) ===
        # Slow-learning weights with synaptic protection
        self.w_core = torch.zeros(out_size, hidden_size, device=device)
        
        # Importance matrix: "synaptic consolidation strength"
        # High importance = protected from modification
        self.importance = torch.zeros(out_size, hidden_size, device=device)
        
        # === LAYER 3: OVERLAY (Short-Term Context) ===
        # Fast-learning, fast-decaying weights for temporary adaptation
        self.w_context = torch.zeros(out_size, hidden_size, device=device)
        
        # Hyperparameters
        self.neurogenesis_threshold = 0.85
        self.utility_decay = 0.995
        self.context_lr = 0.5
        self.context_decay = 0.8
        self.core_lr = 0.05
        self.consolidation_rate = 2.0
    
    def get_sparse_h(self, x):
        """
        Compute sparse hidden representation with dynamic neurogenesis.
        
        This implements the BASE layer functionality:
        1. Project input to high-dimensional space
        2. Check if input is "alien" (poorly represented)
        3. If alien, perform neurogenesis (reincarnate dead neurons)
        4. Apply k-Winner-Take-All sparsity
        5. Update neuron utility statistics
        
        Args:
            x: Input features [batch_size, in_size]
        
        Returns:
            Sparse normalized activations [batch_size, hidden_size]
        """
        # Project to high-dimensional space
        hidden = torch.mm(x, self.projection.t())
        
        # === NEUROGENESIS CHECK ===
        # If max activation is low, this input is "alien" to current representation
        max_val, _ = torch.max(hidden, dim=1)
        poorly_represented = (max_val < self.neurogenesis_threshold)
        
        if poorly_represented.any():
            # Find "dead" neurons (low historical utility)
            # Add noise to break ties and prevent systematic bias
            noisy_util = self.neuron_utility + torch.rand_like(self.neuron_utility) * 0.01
            n_needed = poorly_represented.sum().item()
            _, victim_indices = torch.topk(noisy_util, k=n_needed, largest=False)
            
            # REINCARNATE: Initialize dead neurons to match alien inputs
            self.projection[victim_indices] = x[poorly_represented].detach()
            self.projection = F.normalize(self.projection, p=2, dim=1)
            
            # Reset statistics for reborn neurons
            self.neuron_utility[victim_indices] = 1.0
            self.w_core[:, victim_indices] = 0.0
            self.w_context[:, victim_indices] = 0.0
            self.importance[:, victim_indices] = 0.0
            
            # Recompute activations with new projection
            hidden = torch.mm(x, self.projection.t())
        
        # === SPARSITY (k-Winner-Take-All) ===
        # Keep only top-k activations per sample
        vals, indices = torch.topk(hidden, k=self.sparsity_k, dim=1)
        threshold = vals[:, -1].unsqueeze(1)
        sparse_h = F.relu(hidden - threshold)
        
        # === UTILITY UPDATE (Homeostasis) ===
        # Track which neurons are being used
        self.neuron_utility *= self.utility_decay
        fired_mask = (sparse_h > 0).float().sum(dim=0)
        self.neuron_utility += (fired_mask > 0).float()
        
        return F.normalize(sparse_h, p=2, dim=1)
    
    def forward(self, x):
        """
        Forward pass combining Core and Context signals.
        
        Args:
            x: Input features [batch_size, in_size]
        
        Returns:
            logits: Class predictions [batch_size, out_size]
            sparse_h: Sparse activations [batch_size, hidden_size]
        """
        sparse_h = self.get_sparse_h(x)
        
        # Dual-frequency output: Stable + Plastic
        core_signal = torch.mm(sparse_h, self.w_core.t())
        context_signal = torch.mm(sparse_h, self.w_context.t())
        
        return core_signal + context_signal, sparse_h
    
    def update_batch(self, x, targets):
        """
        Perform one learning step with triarchic update rules.
        
        This implements the key plasticity mechanisms:
        1. Compute prediction error
        2. Update CONTEXT layer (fast, unprotected)
        3. Update CORE layer (slow, importance-protected)
        4. Consolidate important synapses
        
        Args:
            x: Input features [batch_size, in_size]
            targets: Target class indices [batch_size]
        
        Returns:
            acc: Batch accuracy (float)
        """
        sparse_h = self.get_sparse_h(x)
        
        # === PREDICTION ===
        core_logits = torch.mm(sparse_h, self.w_core.t())
        context_logits = torch.mm(sparse_h, self.w_context.t())
        total_logits = core_logits + context_logits
        probs = torch.softmax(total_logits, dim=1)
        
        # === ERROR COMPUTATION ===
        target_oh = F.one_hot(targets, num_classes=self.out_size).float()
        error = target_oh - probs  # [batch_size, out_size]
        
        # Hebbian-style delta (outer product of error and activity)
        delta = torch.mm(error.t(), sparse_h) / x.shape[0]
        
        # === LAYER 3: CONTEXT UPDATE (FAST & TRANSIENT) ===
        # High learning rate, no protection, strong decay
        # This layer absorbs immediate errors without permanent commitment
        self.w_context += self.context_lr * delta
        self.w_context *= self.context_decay  # Exponential decay
        
        # === LAYER 2: CORE UPDATE (SLOW & PROTECTED) ===
        # Apply synaptic brakes based on importance
        brake = torch.clamp(self.importance * 2.0, 0.0, 1.0)
        effective_delta = delta * (1.0 - brake)
        
        # Slow learning with normalization for stability
        self.w_core += self.core_lr * effective_delta
        self.w_core = F.normalize(self.w_core, p=2, dim=1)
        
        # === CONSOLIDATION (Locking Mechanism) ===
        # Synapses that contribute to correct predictions gain importance
        preds = torch.argmax(probs, dim=1)
        correct_mask = (preds == targets).float().unsqueeze(1)
        weighted_activity = sparse_h * correct_mask
        
        # Accumulate importance for correct predictions only
        imp_update = torch.mm(target_oh.t(), weighted_activity) / x.shape[0]
        self.importance += self.consolidation_rate * imp_update
        self.importance = torch.clamp(self.importance, 0.0, 1.0)
        
        # Return metrics
        acc = (preds == targets).float().mean().item()
        return acc
    
    def get_memory_accuracy(self, x, targets):
        """
        Test memory retention using only CORE weights (no context).
        
        This measures "true" long-term memory without temporary adaptations.
        
        Args:
            x: Input features [batch_size, in_size]
            targets: Target class indices [batch_size]
        
        Returns:
            acc: Accuracy using core memory only
        """
        with torch.no_grad():
            sparse_h = self.get_sparse_h(x)
            logits = torch.mm(sparse_h, self.w_core.t())
            acc = (torch.argmax(logits, dim=1) == targets).float().mean().item()
        return acc
    
    def get_stats(self):
        """
        Get diagnostic statistics about the brain state.
        
        Returns:
            dict with keys:
                - n_active_neurons: Number of neurons with non-zero utility
                - context_energy: Mean absolute value of context weights
                - core_energy: Mean absolute value of core weights
                - mean_importance: Average synaptic importance
                - max_importance: Maximum synaptic importance
        """
        return {
            'n_active_neurons': (self.neuron_utility > 0.1).sum().item(),
            'context_energy': self.w_context.abs().mean().item(),
            'core_energy': self.w_core.abs().mean().item(),
            'mean_importance': self.importance.mean().item(),
            'max_importance': self.importance.max().item()
        }

