"""
Training utilities for the Plastic Brain.

This module provides generic training orchestration:
- PhaseConfig: Configuration for training phases
- PlasticTrainer: Multi-phase training with different modes
- Utility functions for data transformations
"""

import torch
import torch.nn.functional as F
import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Callable, List, Dict, Any, Literal


@dataclass
class PhaseConfig:
    """
    Configuration for a single training phase.
    
    Args:
        name: Human-readable phase name
        n_epochs: Number of training epochs in this phase
        data_loader: PyTorch DataLoader for this phase
        mode: Training mode ("plastic" for normal, "panic" for storm adaptation)
        memorize: Whether to store samples in episodic memory
        transform_fn: Optional transformation applied to inputs (e.g., inversion)
        consolidate_after: Whether to consolidate features after this phase
        description: Optional description of what this phase tests
        neurogenesis: Control neurogenesis for this phase:
            - None: Use brain's default setting (no change)
            - True: Enable neurogenesis (recycle dead neurons)
            - False: Disable neurogenesis (preserve all neurons)
    
    Example:
        >>> phase = PhaseConfig(
        ...     name="Animals",
        ...     n_epochs=3,
        ...     data_loader=animal_loader,
        ...     mode="plastic",
        ...     memorize=True,
        ...     consolidate_after=True,
        ...     neurogenesis=True,  # Enable neuron recycling
        ...     description="Learn animal categories"
        ... )
    """
    name: str
    n_epochs: int
    data_loader: torch.utils.data.DataLoader
    mode: Literal["plastic", "panic"] = "plastic"
    memorize: bool = True
    transform_fn: Optional[Callable] = None
    consolidate_after: bool = False
    description: Optional[str] = None
    neurogenesis: Optional[bool] = None  # None = use brain default


@dataclass
class TrainingHistory:
    """Container for training metrics history."""
    step: List[int] = field(default_factory=list)
    epoch: List[int] = field(default_factory=list)
    phase: List[str] = field(default_factory=list)
    task_acc: List[float] = field(default_factory=list)
    memory_fill: List[float] = field(default_factory=list)
    layer_stats: List[Dict] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, List]:
        return {
            'step': self.step,
            'epoch': self.epoch,
            'phase': self.phase,
            'task_acc': self.task_acc,
            'memory_fill': self.memory_fill,
            'layer_stats': self.layer_stats
        }


class PlasticTrainer:
    """
    High-level trainer for PlasticBrain continual learning experiments.
    
    This class orchestrates multi-phase training with:
    - Phase transitions with different modes (plastic/panic)
    - Automatic consolidation between phases
    - Metric tracking (accuracy, memory fill, layer stats)
    - Support for any dataset via DataLoaders
    
    Example:
        >>> from plasticish import PlasticBrain, PlasticTrainer, PhaseConfig
        >>> 
        >>> # Create brain
        >>> brain = PlasticBrain(num_layers=4, device='cuda')
        >>> 
        >>> # Define phases
        >>> phases = [
        ...     PhaseConfig("Phase1", 3, loader1, mode="plastic", consolidate_after=True),
        ...     PhaseConfig("Phase2", 3, loader2, mode="plastic"),
        ...     PhaseConfig("Storm", 3, loader3, mode="panic", transform_fn=invert_tensor)
        ... ]
        >>> 
        >>> # Train
        >>> trainer = PlasticTrainer(brain, phases, device='cuda')
        >>> history = trainer.train()
    
    Args:
        brain: PlasticBrain instance
        phases: List of PhaseConfig defining the curriculum
        eval_loaders: Dict of name -> DataLoader for evaluation (optional)
        eval_interval: Steps between progress logs
        device: torch device
    """
    
    def __init__(
        self,
        brain,
        phases: List[PhaseConfig],
        eval_loaders: Optional[Dict[str, torch.utils.data.DataLoader]] = None,
        eval_interval: int = 100,
        device: str = 'cpu'
    ):
        self.brain = brain
        self.phases = phases
        self.eval_loaders = eval_loaders or {}
        self.eval_interval = eval_interval
        self.device = device
        
        self.history = TrainingHistory()
        self.phase_boundaries = []
    
    def train(self, verbose: bool = True) -> Dict[str, List]:
        """
        Run multi-phase training.
        
        Args:
            verbose: Whether to print progress
        
        Returns:
            history: Dictionary of tracked metrics
        """
        global_step = 0
        
        if verbose:
            print("=" * 60)
            print("🧠 PLASTIC BRAIN: CONTINUAL LEARNING")
            print("=" * 60)
            print(f"  Layers: {self.brain.num_layers}")
            print(f"  Hidden dim: {self.brain.hidden_dim}")
            print(f"  Sparsity k: {self.brain.sparsity_k}")
            print("=" * 60)
        
        for phase_idx, phase in enumerate(self.phases):
            if phase.neurogenesis is not None:
                self.brain.set_neurogenesis(phase.neurogenesis)
            
            if verbose:
                print(f"\n{'='*60}")
                print(f"PHASE {phase_idx + 1}: {phase.name.upper()}")
                neuro_str = "default" if phase.neurogenesis is None else ("ON" if phase.neurogenesis else "OFF")
                print(f"Mode: {phase.mode} | Epochs: {phase.n_epochs} | Memorize: {phase.memorize} | Neurogenesis: {neuro_str}")
                if phase.description:
                    print(f"Goal: {phase.description}")
                print(f"{'='*60}\n")
            
            if global_step > 0:
                self.phase_boundaries.append(global_step)
            
            for epoch in range(phase.n_epochs):
                self.brain.train()
                epoch_accs = []
                
                for batch_idx, (img, label) in enumerate(phase.data_loader):
                    img, label = img.to(self.device), label.to(self.device)
                    
                    if phase.transform_fn is not None:
                        img = phase.transform_fn(img)
                    
                    acc = self.brain.train_step(
                        img, label,
                        mode=phase.mode,
                        memorize=phase.memorize
                    )
                    epoch_accs.append(acc)
                    
                    self.history.step.append(global_step)
                    self.history.epoch.append(epoch)
                    self.history.phase.append(phase.name)
                    self.history.task_acc.append(acc)
                    self.history.memory_fill.append(self.brain.memory.get_fill_ratio())
                    self.history.layer_stats.append(self.brain.get_stats())
                    
                    if global_step % self.eval_interval == 0 and verbose:
                        stats = self.brain.get_stats()
                        print(f"  Step {global_step:>5} | Ep {epoch+1} | "
                              f"Acc: {acc:.2%} | "
                              f"Memory: {stats['memory_fill']:.1%}")
                    
                    global_step += 1
                
                if verbose:
                    mean_acc = np.mean(epoch_accs)
                    print(f"  → Epoch {epoch+1}/{phase.n_epochs} complete | "
                          f"Mean Acc: {mean_acc:.2%}")
            
            if verbose and hasattr(self.brain, 'get_neurogenesis_stats'):
                neuro_stats = self.brain.get_neurogenesis_stats()
                if neuro_stats['total_recycled'] > 0:
                    print(f"\n  > Neurogenesis: {neuro_stats['total_recycled']} neurons recycled")
            
            if phase.consolidate_after:
                if verbose:
                    print(f"\n  > Consolidating features...")
                stats = self.brain.consolidate()
                if verbose:
                    stat_str = " | ".join([f"L{k.split('_')[1]}:{v}" for k, v in stats.items()])
                    print(f"  > Mature neurons: {stat_str}")
            
            if self.eval_loaders and verbose:
                print(f"\n  > Evaluating...")
                for name, loader in self.eval_loaders.items():
                    acc = self.brain.evaluate(loader)
                    print(f"    {name}: {acc:.2f}%")
        
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}\n")
        
        return self.history.to_dict()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics per phase."""
        summary = {}
        
        for phase in self.phases:
            phase_mask = [p == phase.name for p in self.history.phase]
            phase_accs = [acc for acc, mask in zip(self.history.task_acc, phase_mask) if mask]
            
            if phase_accs:
                summary[phase.name] = {
                    'mean_accuracy': np.mean(phase_accs),
                    'final_accuracy': phase_accs[-1],
                    'initial_accuracy': phase_accs[0],
                    'improvement': phase_accs[-1] - phase_accs[0]
                }
        
        return summary
    
    def plot_results(self, figsize=(14, 10)):
        """Plot training results with multiple subplots."""
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        def moving_avg(data, window=25):
            if len(data) < window:
                return data
            ret = np.cumsum(data, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            return ret[window - 1:] / window
        
        steps = self.history.step
        
        # Plot 1: Accuracy
        ax = axs[0, 0]
        ax.plot(steps, self.history.task_acc, color='gray', alpha=0.3, label='Raw')
        if len(steps) >= 25:
            ax.plot(steps[24:], moving_avg(self.history.task_acc),
                    color='green', linewidth=2, label='Smoothed')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Learning Performance", fontweight='bold')
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 2: Memory Fill
        ax = axs[0, 1]
        ax.plot(steps, self.history.memory_fill, color='purple', linewidth=2)
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Episodic Memory Fill", fontweight='bold')
        ax.set_ylabel("Fill Ratio")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.3)
        
        # Plot 3: Mature Neurons per Layer
        ax = axs[1, 0]
        if self.history.layer_stats:
            num_layers = self.history.layer_stats[0]['num_layers']
            colors = plt.cm.viridis(np.linspace(0, 1, num_layers))
            
            for layer_idx in range(num_layers):
                mature_counts = [
                    stats['layers'][f'layer_{layer_idx}']['n_mature']
                    for stats in self.history.layer_stats
                ]
                ax.plot(steps, mature_counts, color=colors[layer_idx],
                        linewidth=2, label=f'Layer {layer_idx}')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Mature Neurons per Layer", fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Plot 4: Gate Means per Layer
        ax = axs[1, 1]
        if self.history.layer_stats:
            for layer_idx in range(num_layers):
                gate_means = [
                    stats['layers'][f'layer_{layer_idx}']['gate_mean']
                    for stats in self.history.layer_stats
                ]
                ax.plot(steps, gate_means, color=colors[layer_idx],
                        linewidth=2, label=f'Layer {layer_idx}')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Layer Gate Values", fontweight='bold')
        ax.set_ylabel("Mean Gate")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# === UTILITY FUNCTIONS (Generic) ===

def invert_colors(img: torch.Tensor) -> torch.Tensor:
    """
    Invert normalized image colors (for ImageNet-normalized tensors).
    
    Use this for images normalized with ImageNet stats.
    
    Args:
        img: Normalized image tensor
    
    Returns:
        Inverted image tensor
    """
    return img * -1.0


def invert_tensor(img: torch.Tensor) -> torch.Tensor:
    """
    Invert tensor values (1 - x).
    
    Use this for tensors in [0, 1] range before normalization.
    
    Args:
        img: Image tensor
    
    Returns:
        Inverted tensor
    """
    return 1.0 - img


def add_noise(img: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Add Gaussian noise to images.
    
    Args:
        img: Image tensor
        std: Noise standard deviation
    
    Returns:
        Noisy image tensor
    """
    noise = torch.randn_like(img) * std
    return img + noise


def blur_image(img: torch.Tensor, kernel_size: int = 5) -> torch.Tensor:
    """
    Apply Gaussian blur to images.
    
    Args:
        img: Image tensor [B, C, H, W] or [C, H, W]
    Returns:
        Blurred image tensor
    """
    was_3d = img.dim() == 3
    if was_3d:
        img = img.unsqueeze(0)
    
    # Simple box blur approximation
    padding = kernel_size // 2
    kernel = torch.ones(1, 1, kernel_size, kernel_size, device=img.device) / (kernel_size ** 2)
    
    blurred = []
    for c in range(img.shape[1]):
        blurred.append(F.conv2d(img[:, c:c+1], kernel, padding=padding))
    
    result = torch.cat(blurred, dim=1)
    return result.squeeze(0) if was_3d else result


# === LEGACY TRAINER (Backwards Compatibility) ===

class TriarchicTrainer:
    """
    Legacy trainer for TriarchicBrain.
    
    NOTE: For new projects, use PlasticTrainer with PlasticBrain.
    """
    
    def __init__(
        self,
        brain,
        eyes,
        phases: List[PhaseConfig],
        memory_test_loader: Optional[torch.utils.data.DataLoader] = None,
        eval_interval: int = 50,
        device: str = 'cpu'
    ):
        self.brain = brain
        self.eyes = eyes
        self.phases = phases
        self.memory_test_loader = memory_test_loader
        self.eval_interval = eval_interval
        self.device = device
        
        self.history = {
            'step': [],
            'phase': [],
            'task_acc': [],
            'memory_acc': [],
            'context_energy': [],
            'core_energy': [],
            'n_active_neurons': [],
            'mean_importance': []
        }
        self.phase_boundaries = []
        
    def train(self, verbose: bool = True) -> Dict[str, List]:
        """Run multi-phase training."""
        global_step = 0
        
        if verbose:
            print("=" * 60)
            print("TRIARCHIC BRAIN: CONTINUAL LEARNING SIMULATION")
            print("=" * 60)
        
        for phase_idx, phase in enumerate(self.phases):
            if verbose:
                print(f"\n{'='*60}")
                print(f"PHASE {phase_idx + 1}: {phase.name.upper()}")
                print(f"Duration: {phase.n_epochs} epochs")
                if phase.description:
                    print(f"Goal: {phase.description}")
                print(f"{'='*60}\n")
            
            if global_step > 0:
                self.phase_boundaries.append(global_step)
            
            iterator = iter(phase.data_loader)
            n_steps = phase.n_epochs * len(phase.data_loader)
            
            for step_in_phase in range(n_steps):
                try:
                    img, label = next(iterator)
                except StopIteration:
                    iterator = iter(phase.data_loader)
                    img, label = next(iterator)
                
                img, label = img.to(self.device), label.to(self.device)
                
                if phase.transform_fn is not None:
                    img = phase.transform_fn(img)
                
                features = self.eyes(img)
                task_acc = self.brain.update_batch(features, label)
                
                stats = self.brain.get_stats()
                
                self.history['step'].append(global_step)
                self.history['phase'].append(phase.name)
                self.history['task_acc'].append(task_acc)
                self.history['context_energy'].append(stats['context_energy'])
                self.history['core_energy'].append(stats['core_energy'])
                self.history['n_active_neurons'].append(stats['n_active_neurons'])
                self.history['mean_importance'].append(stats['mean_importance'])
                
                if global_step % self.eval_interval == 0:
                    if self.memory_test_loader is not None:
                        mem_acc = self._evaluate_memory()
                        self.history['memory_acc'].append(mem_acc)
                    else:
                        mem_acc = None
                    
                    if verbose:
                        mem_str = f"Memory: {mem_acc:.2%}" if mem_acc is not None else ""
                        print(f"Step {global_step:>4} | {phase.name:>10} | "
                              f"Task: {task_acc:.2%} | {mem_str} | "
                              f"Active Neurons: {stats['n_active_neurons']}")
                
                global_step += 1
        
        if verbose:
            print(f"\n{'='*60}")
            print("TRAINING COMPLETE")
            print(f"{'='*60}\n")
        
        return self.history
    
    def _evaluate_memory(self) -> float:
        """Evaluate memory retention on test set."""
        accs = []
        for img, label in self.memory_test_loader:
            img, label = img.to(self.device), label.to(self.device)
            features = self.eyes(img)
            acc = self.brain.get_memory_accuracy(features, label)
            accs.append(acc)
        return np.mean(accs)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get training summary statistics."""
        summary = {}
        
        for phase in self.phases:
            phase_mask = [p == phase.name for p in self.history['phase']]
            phase_accs = [acc for acc, mask in zip(self.history['task_acc'], phase_mask) if mask]
            
            summary[phase.name] = {
                'mean_accuracy': np.mean(phase_accs) if phase_accs else 0.0,
                'final_accuracy': phase_accs[-1] if phase_accs else 0.0,
                'initial_accuracy': phase_accs[0] if phase_accs else 0.0,
                'improvement': (phase_accs[-1] - phase_accs[0]) if phase_accs else 0.0
            }
        
        return summary
    
    def plot_results(self, figsize=(14, 10)):
        """Plot training results."""
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        def moving_avg(data, window=25):
            if len(data) < window:
                return data
            ret = np.cumsum(data, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            return ret[window - 1:] / window
        
        steps = self.history['step']
        
        ax = axs[0, 0]
        ax.plot(steps, self.history['task_acc'], color='gray', alpha=0.3, label='Raw')
        if len(steps) >= 25:
            ax.plot(steps[24:], moving_avg(self.history['task_acc']),
                    color='green', linewidth=2, label='Task Accuracy (MA)')
        
        if self.history['memory_acc']:
            mem_steps = [s for i, s in enumerate(steps) if i % self.eval_interval == 0]
            ax.plot(mem_steps[:len(self.history['memory_acc'])], 
                    self.history['memory_acc'],
                    color='purple', linewidth=3, marker='o', 
                    markersize=4, label='Core Memory')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Learning Performance", fontweight='bold')
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axs[0, 1]
        ax.plot(steps, self.history['context_energy'], 
                color='orange', linewidth=2, label='Context (Fast)')
        ax.plot(steps, self.history['core_energy'], 
                color='blue', linewidth=2, label='Core (Slow)')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Synaptic Energy", fontweight='bold')
        ax.set_ylabel("Mean |Weight|")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        ax = axs[1, 0]
        ax.plot(steps, self.history['n_active_neurons'], color='purple', linewidth=2)
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Active Neurons", fontweight='bold')
        ax.set_ylabel("Count")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.3)
        
        ax = axs[1, 1]
        ax.plot(steps, self.history['mean_importance'], color='brown', linewidth=2)
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Synaptic Importance", fontweight='bold')
        ax.set_ylabel("Mean Importance")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()
