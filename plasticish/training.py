"""
Training utilities for the Triarchic Brain.

This module provides high-level training orchestration, including:
- Multi-phase curriculum scheduling
- Data transformation utilities
- Metrics tracking and logging
"""

import torch
import numpy as np
from dataclasses import dataclass
from typing import Optional, Callable, List, Dict, Any


@dataclass
class PhaseConfig:
    """
    Configuration for a single training phase.
    
    Args:
        name: Human-readable phase name
        n_steps: Number of training steps in this phase
        data_loader: PyTorch DataLoader for this phase
        transform_fn: Optional transformation applied to inputs (e.g., inversion)
        description: Optional description of what this phase tests
    """
    name: str
    n_steps: int
    data_loader: torch.utils.data.DataLoader
    transform_fn: Optional[Callable] = None
    description: Optional[str] = None


class TriarchicTrainer:
    """
    High-level trainer for Triarchic Brain continual learning experiments.
    
    This class orchestrates multi-phase training with automatic:
    - Phase transitions
    - Metric tracking (accuracy, memory retention, neural statistics)
    - Periodic evaluation
    - Logging
    
    Example:
        >>> # Define training phases
        >>> phases = [
        ...     PhaseConfig("Animals", 1500, animal_loader),
        ...     PhaseConfig("Vehicles", 1000, vehicle_loader),
        ...     PhaseConfig("Storm", 1500, animal_loader, transform_fn=invert_colors)
        ... ]
        >>> 
        >>> # Initialize trainer
        >>> trainer = TriarchicTrainer(brain, eyes, phases, memory_test_loader)
        >>> 
        >>> # Run training
        >>> history = trainer.train()
        >>> trainer.plot_results()
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
        """
        Initialize trainer.
        
        Args:
            brain: TriarchicBrain instance
            eyes: PretrainedEyes instance
            phases: List of PhaseConfig defining the curriculum
            memory_test_loader: DataLoader for testing memory retention (optional)
            eval_interval: Steps between memory evaluations
            device: torch device
        """
        self.brain = brain
        self.eyes = eyes
        self.phases = phases
        self.memory_test_loader = memory_test_loader
        self.eval_interval = eval_interval
        self.device = device
        
        # History tracking
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
        
        # Phase boundaries for visualization
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
            print("TRIARCHIC BRAIN: CONTINUAL LEARNING SIMULATION")
            print("=" * 60)
        
        for phase_idx, phase in enumerate(self.phases):
            if verbose:
                print(f"\n{'='*60}")
                print(f"PHASE {phase_idx + 1}: {phase.name.upper()}")
                print(f"Duration: {phase.n_steps} steps")
                if phase.description:
                    print(f"Goal: {phase.description}")
                print(f"{'='*60}\n")
            
            # Record phase boundary
            if global_step > 0:
                self.phase_boundaries.append(global_step)
            
            # Create iterator
            iterator = iter(phase.data_loader)
            
            # Training loop for this phase
            for step_in_phase in range(phase.n_steps):
                # Get batch
                try:
                    img, label = next(iterator)
                except StopIteration:
                    iterator = iter(phase.data_loader)
                    img, label = next(iterator)
                
                img, label = img.to(self.device), label.to(self.device)
                
                # Apply phase-specific transformation (e.g., color inversion)
                if phase.transform_fn is not None:
                    img = phase.transform_fn(img)
                
                # Training step
                features = self.eyes(img)
                task_acc = self.brain.update_batch(features, label)
                
                # Get brain statistics
                stats = self.brain.get_stats()
                
                # Record metrics
                self.history['step'].append(global_step)
                self.history['phase'].append(phase.name)
                self.history['task_acc'].append(task_acc)
                self.history['context_energy'].append(stats['context_energy'])
                self.history['core_energy'].append(stats['core_energy'])
                self.history['n_active_neurons'].append(stats['n_active_neurons'])
                self.history['mean_importance'].append(stats['mean_importance'])
                
                # Periodic memory evaluation
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
        """
        Evaluate memory retention on test set.
        
        Uses only CORE weights to test "true" long-term memory
        without temporary context adaptations.
        
        Returns:
            Average accuracy across memory test set
        """
        accs = []
        for img, label in self.memory_test_loader:
            img, label = img.to(self.device), label.to(self.device)
            features = self.eyes(img)
            acc = self.brain.get_memory_accuracy(features, label)
            accs.append(acc)
        return np.mean(accs)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Get training summary statistics.
        
        Returns:
            Dictionary with summary metrics per phase
        """
        summary = {}
        
        for phase in self.phases:
            phase_mask = [p == phase.name for p in self.history['phase']]
            phase_accs = [acc for acc, mask in zip(self.history['task_acc'], phase_mask) if mask]
            
            summary[phase.name] = {
                'mean_accuracy': np.mean(phase_accs),
                'final_accuracy': phase_accs[-1] if phase_accs else 0.0,
                'initial_accuracy': phase_accs[0] if phase_accs else 0.0,
                'improvement': (phase_accs[-1] - phase_accs[0]) if phase_accs else 0.0
            }
        
        return summary
    
    def plot_results(self, figsize=(14, 10)):
        """
        Plot training results with multiple subplots.
        
        Creates a comprehensive visualization showing:
        1. Task accuracy and memory retention over time
        2. Context vs Core energy (plasticity indicators)
        3. Active neuron count (capacity usage)
        4. Synaptic importance (consolidation strength)
        
        Args:
            figsize: Figure size tuple
        """
        import matplotlib.pyplot as plt
        
        fig, axs = plt.subplots(2, 2, figsize=figsize)
        
        # Helper function for moving average
        def moving_avg(data, window=25):
            ret = np.cumsum(data, dtype=float)
            ret[window:] = ret[window:] - ret[:-window]
            return ret[window - 1:] / window
        
        steps = self.history['step']
        
        # === Plot 1: Accuracy ===
        ax = axs[0, 0]
        ax.plot(steps, self.history['task_acc'], color='gray', alpha=0.3, label='Raw')
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
        
        ax.set_title("Learning Performance: Plasticity vs Stability", fontweight='bold')
        ax.set_ylabel("Accuracy")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # === Plot 2: Energy (Context vs Core) ===
        ax = axs[0, 1]
        ax.plot(steps, self.history['context_energy'], 
                color='orange', linewidth=2, label='Context (Fast)')
        ax.plot(steps, self.history['core_energy'], 
                color='blue', linewidth=2, label='Core (Slow)')
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Synaptic Energy: Dual-Frequency Dynamics", fontweight='bold')
        ax.set_ylabel("Mean |Weight|")
        ax.set_xlabel("Training Steps")
        ax.legend()
        ax.grid(alpha=0.3)
        
        # === Plot 3: Active Neurons ===
        ax = axs[1, 0]
        ax.plot(steps, self.history['n_active_neurons'], 
                color='purple', linewidth=2)
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Neural Capacity: Neurogenesis & Sparsity", fontweight='bold')
        ax.set_ylabel("Active Neurons")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.3)
        
        # === Plot 4: Importance ===
        ax = axs[1, 1]
        ax.plot(steps, self.history['mean_importance'], 
                color='brown', linewidth=2)
        
        for boundary in self.phase_boundaries:
            ax.axvline(x=boundary, color='red', linestyle='--', alpha=0.6)
        
        ax.set_title("Synaptic Consolidation Strength", fontweight='bold')
        ax.set_ylabel("Mean Importance")
        ax.set_xlabel("Training Steps")
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.show()


# === UTILITY FUNCTIONS ===

def invert_colors(img):
    """
    Invert normalized image colors (simulate distributional shift).
    
    This creates a "storm" effect where colors are inverted,
    testing the brain's ability to adapt via the context layer
    without destroying core memory.
    
    Args:
        img: Normalized image tensor
    
    Returns:
        Inverted image tensor
    """
    return img * -1.0


def add_noise(img, std=0.1):
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


def create_cifar10_loaders(
    batch_size=64,
    data_root='./data',
    num_workers=0
):
    """
    Create standard CIFAR-10 data loaders for continual learning experiments.
    
    Splits CIFAR-10 into:
    - Animals: Bird, Cat, Deer, Dog, Frog, Horse (classes 2-7)
    - Vehicles: Plane, Car, Ship, Truck (classes 0, 1, 8, 9)
    
    Args:
        batch_size: Batch size for training
        data_root: Directory to store/load CIFAR-10
        num_workers: Number of data loading workers
    
    Returns:
        Tuple of (animal_loader, vehicle_loader, memory_test_loader, full_dataset)
    """
    import torchvision
    import torchvision.transforms as transforms
    
    # ResNet normalization
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Load dataset
    dataset = torchvision.datasets.CIFAR10(
        root=data_root,
        train=True,
        download=True,
        transform=transform
    )
    
    # Split by semantic category
    animal_indices = [i for i, t in enumerate(dataset.targets) if t in [2, 3, 4, 5, 6, 7]]
    vehicle_indices = [i for i, t in enumerate(dataset.targets) if t in [0, 1, 8, 9]]
    
    # Create loaders
    animal_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, animal_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    vehicle_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, vehicle_indices),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    
    # Memory test set (small subset of animals for quick evaluation)
    memory_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(dataset, animal_indices[:500]),
        batch_size=100,
        shuffle=False,
        num_workers=num_workers
    )
    
    return animal_loader, vehicle_loader, memory_test_loader, dataset

