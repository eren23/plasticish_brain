"""
Visualization tools for understanding Triarchic Brain behavior.

This module provides detailed visualizations of:
- Sparse neural activity patterns
- Core vs Context weight contributions
- Prediction confidence and correctness
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def denormalize_imagenet(tensor):
    """
    Reverse ImageNet/ResNet normalization for display.
    
    Converts normalized tensors back to [0, 1] range for visualization.
    
    Args:
        tensor: Normalized image tensor [C, H, W]
    
    Returns:
        Denormalized numpy array [H, W, C] in [0, 1] range
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    img = tensor.cpu().permute(1, 2, 0).numpy()
    img = std * img + mean
    return np.clip(img, 0, 1)


class TriarchicVisualizer:
    """
    Comprehensive visualization suite for Triarchic Brain analysis.
    
    This class creates detailed visualizations showing:
    1. Input images with predictions
    2. Sparse neural activity heatmaps
    3. Synaptic contribution analysis (Core vs Context)
    
    The visualizations help understand:
    - Which neurons fire for which inputs
    - How Core (stable) and Context (plastic) layers contribute
    - How the brain responds to different types of inputs
    
    Example:
        >>> visualizer = TriarchicVisualizer(brain, eyes)
        >>> visualizer.visualize_thinking(
        ...     dataset,
        ...     num_samples=5,
        ...     class_names=['Cat', 'Dog', ...],
        ...     title="Normal Conditions"
        ... )
    """
    
    def __init__(self, brain, eyes, device='cpu'):
        """
        Initialize visualizer.
        
        Args:
            brain: TriarchicBrain instance
            eyes: PretrainedEyes instance
            device: torch device
        """
        self.brain = brain
        self.eyes = eyes
        self.device = device
    
    def visualize_thinking(
        self,
        dataset,
        num_samples=5,
        class_names=None,
        transform_fn=None,
        title=None,
        figsize=None
    ):
        """
        Visualize detailed brain activity for sample inputs.
        
        Creates a multi-panel visualization for each sample showing:
        - Panel 1: The input image with prediction
        - Panel 2: Sparse neural activity heatmap
        - Panel 3: Synaptic contribution breakdown (Core vs Context)
        
        This visualization reveals:
        - How the brain represents different inputs
        - Whether stable (Core) or plastic (Context) weights dominate
        - Sparsity patterns (which neurons fire)
        
        Args:
            dataset: PyTorch Dataset to sample from
            num_samples: Number of samples to visualize
            class_names: List of class names (defaults to CIFAR-10)
            transform_fn: Optional transformation (e.g., color inversion)
            title: Overall figure title
            figsize: Figure size (defaults to automatic)
        """
        # Default CIFAR-10 classes
        if class_names is None:
            class_names = [
                'Plane', 'Car', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
            ]
        
        # Setup data loader
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=1, shuffle=True
        )
        
        # Collect samples
        samples = []
        iter_data = iter(loader)
        for _ in range(num_samples):
            img, lbl = next(iter_data)
            if transform_fn is not None:
                img = transform_fn(img)
            samples.append((img, lbl))
        
        # Create figure
        if figsize is None:
            figsize = (16, 4 * num_samples)
        fig = plt.figure(figsize=figsize)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        for i, (img, lbl) in enumerate(samples):
            img, lbl = img.to(self.device), lbl.item()
            
            # === INSTRUMENTED FORWARD PASS ===
            with torch.no_grad():
                # Extract features
                feats = self.eyes(img)
                
                # Get sparse representation
                sparse_h = self.brain.get_sparse_h(feats)  # [1, hidden_size]
                
                # Decompose prediction into Core and Context
                core_logits = torch.mm(sparse_h, self.brain.w_core.t())
                context_logits = torch.mm(sparse_h, self.brain.w_context.t())
                total_logits = core_logits + context_logits
                
                probs = torch.softmax(total_logits, dim=1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0, pred_idx].item()
            
            # === EXTRACT ANALYSIS DATA ===
            
            # Active neurons
            active_indices = torch.nonzero(sparse_h.squeeze()).flatten()
            n_active = len(active_indices)
            
            # Get weights for predicted class only
            # (these are the "votes" that led to this prediction)
            relevant_core = self.brain.w_core[pred_idx, active_indices]
            relevant_context = self.brain.w_context[pred_idx, active_indices]
            
            # Sort by total contribution magnitude
            total_contribution = relevant_core + relevant_context
            sorted_vals, sorted_idx = torch.sort(
                total_contribution.abs(), descending=True
            )
            
            # Keep top 10 contributors for visualization
            top_k = min(10, len(sorted_idx))
            top_10_idx = sorted_idx[:top_k]
            top_core = relevant_core[top_10_idx].cpu().numpy()
            top_context = relevant_context[top_10_idx].cpu().numpy()
            
            # === CREATE VISUALIZATIONS ===
            
            # Panel 1: Input Image
            ax_img = fig.add_subplot(num_samples, 3, (i * 3) + 1)
            ax_img.imshow(denormalize_imagenet(img.squeeze()))
            
            status = "✓ CORRECT" if lbl == pred_idx else "✗ WRONG"
            color = "green" if lbl == pred_idx else "red"
            title_text = (f"{status}\n"
                         f"True: {class_names[lbl]}\n"
                         f"Pred: {class_names[pred_idx]} ({confidence:.1%})")
            ax_img.set_title(title_text, color=color, fontweight='bold')
            ax_img.axis('off')
            
            # Panel 2: Sparse Activity Heatmap
            ax_scan = fig.add_subplot(num_samples, 3, (i * 3) + 2)
            activity_map = sparse_h.squeeze().cpu().numpy()
            
            # Reshape to square grid for visualization
            # Pad to perfect square
            side = int(np.ceil(np.sqrt(self.brain.hidden_size)))
            padded_map = np.zeros(side * side)
            padded_map[:len(activity_map)] = activity_map
            activity_grid = padded_map.reshape(side, side)
            
            sns.heatmap(
                activity_grid,
                ax=ax_scan,
                cbar=False,
                cmap="magma",
                xticklabels=False,
                yticklabels=False
            )
            ax_scan.set_title(
                f"Sparse Neural Activity\n{n_active} / {self.brain.hidden_size} "
                f"neurons fired ({n_active/self.brain.hidden_size*100:.2f}%)"
            )
            
            # Panel 3: Synaptic Contribution (Core vs Context)
            ax_w = fig.add_subplot(num_samples, 3, (i * 3) + 3)
            indices = np.arange(top_k)
            
            # Stacked bar chart showing Core (bottom) and Context (top)
            ax_w.bar(
                indices, top_core,
                label='Core (Stable Memory)',
                color='#1f77b4',  # Blue
                alpha=0.8
            )
            ax_w.bar(
                indices, top_context,
                bottom=top_core,
                label='Context (Fast Adaptation)',
                color='#ff7f0e',  # Orange
                alpha=0.9
            )
            
            ax_w.axhline(0, color='black', linewidth=0.5)
            ax_w.set_title(
                f"Top {top_k} Synaptic Contributors\n"
                "Blue = Stable | Orange = Plastic"
            )
            ax_w.set_xlabel("Neuron Rank (by contribution)")
            ax_w.set_ylabel("Weight Magnitude")
            ax_w.legend(loc='upper right', fontsize='small')
            ax_w.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def compare_conditions(
        self,
        dataset,
        num_samples=4,
        class_names=None
    ):
        """
        Compare brain behavior under normal vs perturbed conditions.
        
        Creates side-by-side visualizations showing:
        1. Normal inputs: Should see Core (blue) dominance
        2. Perturbed inputs (inverted): Should see Context (orange) compensation
        
        This demonstrates the dual-frequency mechanism in action.
        
        Args:
            dataset: PyTorch Dataset to sample from
            num_samples: Number of samples per condition
            class_names: List of class names
        """
        print("=" * 70)
        print("COMPARING NORMAL vs PERTURBED CONDITIONS")
        print("=" * 70)
        
        print("\n--- CONDITION 1: Normal Inputs ---")
        print("Expectation: Predictions driven by CORE (blue bars)")
        print("This shows stable, consolidated knowledge.\n")
        
        self.visualize_thinking(
            dataset,
            num_samples=num_samples,
            class_names=class_names,
            transform_fn=None,
            title="NORMAL CONDITIONS: Core Memory Dominance"
        )
        
        print("\n" + "=" * 70)
        print("--- CONDITION 2: Inverted Colors (The Storm) ---")
        print("Expectation: CONTEXT (orange bars) compensates for distortion")
        print("Core memory stays intact while Context adapts.\n")
        
        self.visualize_thinking(
            dataset,
            num_samples=num_samples,
            class_names=class_names,
            transform_fn=lambda x: x * -1.0,  # Invert colors
            title="PERTURBED CONDITIONS: Context Adaptation"
        )
    
    def plot_weight_distribution(self):
        """
        Plot the distribution of Core and Context weights.
        
        Useful for understanding:
        - Weight sparsity patterns
        - Relative magnitude of stable vs plastic components
        """
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        # Core weights
        core_flat = self.brain.w_core.flatten().cpu().numpy()
        axs[0].hist(core_flat, bins=100, color='blue', alpha=0.7)
        axs[0].set_title("Core Weight Distribution (Stable Memory)")
        axs[0].set_xlabel("Weight Value")
        axs[0].set_ylabel("Frequency")
        axs[0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        # Context weights
        context_flat = self.brain.w_context.flatten().cpu().numpy()
        axs[1].hist(context_flat, bins=100, color='orange', alpha=0.7)
        axs[1].set_title("Context Weight Distribution (Fast Adaptation)")
        axs[1].set_xlabel("Weight Value")
        axs[1].set_ylabel("Frequency")
        axs[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_importance_heatmap(self, class_names=None):
        """
        Visualize the importance matrix (synaptic consolidation).
        
        Shows which synapses are "locked" (high importance) vs
        still plastic (low importance).
        
        Args:
            class_names: List of class names for y-axis labels
        """
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.brain.out_size)]
        
        importance = self.brain.importance.cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(
            importance,
            cmap='YlOrRd',
            cbar_kws={'label': 'Importance (Consolidation Strength)'},
            yticklabels=class_names,
            xticklabels=False
        )
        plt.title("Synaptic Importance Matrix: Which Weights Are Protected?")
        plt.xlabel(f"Hidden Neurons (0 - {self.brain.hidden_size})")
        plt.ylabel("Output Classes")
        plt.tight_layout()
        plt.show()

