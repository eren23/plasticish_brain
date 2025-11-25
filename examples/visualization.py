"""
Visualization tools for understanding Plastic Brain behavior.

This module provides detailed visualizations of:
- Multi-layer sparse neural activity patterns
- Layer-by-layer thought tracing
- Functional connectivity (Hebbian assemblies)
- Neuron specialization analysis
- Core vs Context weight contributions (legacy)
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, List, Dict, Any


def denormalize_imagenet(tensor: torch.Tensor) -> np.ndarray:
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


class PlasticVisualizer:
    """
    Comprehensive visualization suite for PlasticBrain multi-layer analysis.
    
    This class creates detailed visualizations showing:
    1. Layer-by-layer thought propagation
    2. Sparse neural activity patterns per layer
    3. Functional connectivity (co-activation) analysis
    4. Neuron specialization (what triggers each neuron)
    5. Signal flow through the network
    
    Example:
        >>> visualizer = PlasticVisualizer(brain)
        >>> visualizer.trace_thought(storm_loader)
        >>> visualizer.bio_debug_suite(normal_loader, storm_loader)
    """
    
    def __init__(self, brain, device: str = 'cpu', class_names: Optional[List[str]] = None):
        """
        Initialize visualizer.
        
        Args:
            brain: PlasticBrain instance
            device: torch device
            class_names: List of class names (defaults to CIFAR-10)
        """
        self.brain = brain
        self.device = device
        self.class_names = class_names or [
            'Plane', 'Car', 'Bird', 'Cat', 'Deer',
            'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
        ]
    
    def trace_thought(
        self,
        loader: torch.utils.data.DataLoader,
        sample_idx: int = 0,
        use_panic_mode: bool = True,
        show_input: bool = True
    ):
        """
        Trace a single thought through all layers with detailed visualization.
        
        Creates a multi-panel figure showing:
        - Per-layer sparse concept activations
        - Per-layer associative recall (decoder output)
        - Signal flow statistics (input/output magnitude)
        
        Args:
            loader: DataLoader to get sample from
            sample_idx: Index of sample in batch to trace
            use_panic_mode: Whether to use panic mode (boosted signals)
            show_input: Whether to display the input image
        """
        print("\n" + "=" * 50)
        print("      🧠 SINGLE THOUGHT TRACE (LAYER BY LAYER)")
        print("=" * 50)
        
        self.brain.eval()
        
        # Get sample
        images, labels = next(iter(loader))
        img_tensor = images[sample_idx:sample_idx+1].to(self.device)
        label_idx = labels[sample_idx].item()
        label_name = self.class_names[label_idx]
        
        print(f"Target: {label_name}")
        
        # Trace through layers
        traces = []
        
        with torch.no_grad():
            # Eye: Get features
            feat = self.brain.eye(img_tensor)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            current_thought = feat
            
            # Brain layers
            for i, layer in enumerate(self.brain.brain):
                # Normalize input
                x_norm = F.normalize(current_thought, p=2, dim=1)
                
                # Encoder (pre-sparsity)
                hidden = layer.encoder(x_norm)
                if use_panic_mode:
                    hidden = hidden * 2.0
                
                # Sparsity
                top_val, top_idx = torch.topk(hidden, layer.k, dim=1)
                sparse_code = torch.zeros_like(hidden).scatter_(1, top_idx, top_val)
                
                # Decoder (imagination)
                imagination = layer.decoder(sparse_code)
                
                # Output (residual)
                gated_imagination = layer.gate.view(1, -1) * imagination
                next_thought = current_thought + gated_imagination
                
                traces.append({
                    "layer": i,
                    "input_mag": current_thought.norm().item(),
                    "sparse_code": sparse_code.cpu().numpy().flatten(),
                    "imagination": imagination.cpu().numpy().flatten(),
                    "gate_avg": layer.gate.mean().item(),
                    "output_mag": next_thought.norm().item()
                })
                
                current_thought = next_thought
            
            # Memory lookup
            final_thought = self.brain.bn(current_thought)
            logits = self.brain.memory(final_thought)
            best_idx = logits.argmax().item()
            prediction = self.class_names[best_idx]
            confidence = logits.max().item()
        
        print(f"Prediction: {prediction} (Conf: {confidence:.2f})")
        color_pred = 'green' if prediction == label_name else 'red'
        
        # Visualization
        num_layers = len(traces)
        fig, axes = plt.subplots(num_layers, 3, figsize=(18, 3.5 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, trace in enumerate(traces):
            # Col 1: Sparse Code
            ax_code = axes[i, 0]
            active_neurons = trace['sparse_code']
            ax_code.plot(active_neurons, color='purple', linewidth=1)
            ax_code.set_title(f"L{i} Sparse Concepts (k={self.brain.sparsity_k})")
            ax_code.set_ylabel("Activation")
            ymax = active_neurons.max() if active_neurons.max() > 0 else 1.0
            ax_code.set_ylim(0, ymax * 1.1)
            ax_code.grid(True, alpha=0.3)
            
            # Col 2: Imagination
            ax_img = axes[i, 1]
            ax_img.plot(trace['imagination'], color='orange', alpha=0.8)
            ax_img.set_title(f"L{i} Associative Recall (Decoder)")
            ax_img.set_ylabel("Feature Vector Value")
            ax_img.grid(True, alpha=0.3)
            
            # Col 3: Signal Flow
            ax_stat = axes[i, 2]
            ax_stat.bar(["In Mag", "Out Mag"],
                       [trace['input_mag'], trace['output_mag']],
                       color=['blue', 'green'])
            ax_stat.set_title(f"L{i} Signal Flow (Gate: {trace['gate_avg']:.4f})")
            ax_stat.set_ylim(0, max(trace['input_mag'], trace['output_mag']) * 1.3)
            
            diff = trace['output_mag'] - trace['input_mag']
            ax_stat.text(0.5, trace['output_mag'] * 1.05, f"Change: {diff:+.2f}",
                        ha='center', va='bottom', fontweight='bold', fontsize=10)
        
        plt.suptitle(f"True: {label_name} → Predicted: {prediction}",
                    y=1.02, fontsize=16, color=color_pred)
        plt.tight_layout()
        plt.show()
        
        # Show input image
        if show_input:
            plt.figure(figsize=(4, 4))
            img_show = img_tensor.squeeze(0).cpu()
            if use_panic_mode:
                img_show = 1.0 - img_show  # Re-invert for display
            img_show = denormalize_imagenet(img_show)
            plt.imshow(img_show)
            plt.title(f"Input: {label_name}")
            plt.axis('off')
            plt.show()
    
    def trace_thought_matrix(
        self,
        loader: torch.utils.data.DataLoader,
        sample_idx: int = 0,
        use_panic_mode: bool = True
    ):
        """
        Visualize thought as 2D matrix grids for each layer.
        
        Shows sparse patterns as heatmaps for intuitive understanding
        of layer-by-layer processing.
        
        Args:
            loader: DataLoader to get sample from
            sample_idx: Index of sample in batch
            use_panic_mode: Whether to use panic mode
        """
        print("\n" + "=" * 50)
        print("      🧠 2D MATRIX BRAIN ACTIVITY SCAN")
        print("=" * 50)
        
        self.brain.eval()
        
        images, labels = next(iter(loader))
        img_tensor = images[sample_idx:sample_idx+1].to(self.device)
        label_idx = labels[sample_idx].item()
        label_name = self.class_names[label_idx]
        
        print(f"Subject: {label_name}")
        
        traces = []
        
        with torch.no_grad():
            feat = self.brain.eye(img_tensor)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            current_thought = feat
            
            for i, layer in enumerate(self.brain.brain):
                x_norm = F.normalize(current_thought, p=2, dim=1)
                
                hidden = layer.encoder(x_norm)
                if use_panic_mode:
                    hidden = hidden * 2.0
                
                top_val, top_idx = torch.topk(hidden, layer.k, dim=1)
                sparse_code = torch.zeros_like(hidden).scatter_(1, top_idx, top_val)
                
                imagination = layer.decoder(sparse_code)
                gated_imagination = layer.gate.view(1, -1) * imagination
                next_thought = current_thought + gated_imagination
                
                # Calculate grid sizes
                hidden_size = layer.hidden_dim
                in_size = layer.in_dim
                sparse_side = int(np.ceil(np.sqrt(hidden_size)))
                dense_side = int(np.ceil(np.sqrt(in_size)))
                
                sparse_flat = sparse_code.cpu().numpy().flatten()
                sparse_padded = np.zeros(sparse_side * sparse_side)
                sparse_padded[:len(sparse_flat)] = sparse_flat
                
                dense_flat = imagination.cpu().numpy().flatten()
                dense_padded = np.zeros(dense_side * dense_side)
                dense_padded[:len(dense_flat)] = dense_flat
                
                traces.append({
                    "layer": i,
                    "sparse_map": sparse_padded.reshape(sparse_side, sparse_side),
                    "dense_map": dense_padded.reshape(dense_side, dense_side),
                    "input_mag": current_thought.norm().item(),
                    "output_mag": next_thought.norm().item()
                })
                
                current_thought = next_thought
            
            final_thought = self.brain.bn(current_thought)
            logits = self.brain.memory(final_thought)
            prediction = self.class_names[logits.argmax().item()]
            confidence = logits.max().item()
        
        print(f"Prediction: {prediction} (Confidence: {confidence:.2f})")
        
        num_layers = len(traces)
        fig, axes = plt.subplots(num_layers, 3, figsize=(18, 4 * num_layers))
        
        if num_layers == 1:
            axes = axes.reshape(1, -1)
        
        for i, trace in enumerate(traces):
            # Col 1: Sparse Thought Matrix
            ax_sparse = axes[i, 0]
            im1 = ax_sparse.imshow(trace['sparse_map'], cmap='inferno', aspect='auto')
            ax_sparse.set_title(f"Layer {i} Thought Pattern\n(Sparse Concept Grid)")
            ax_sparse.axis('off')
            plt.colorbar(im1, ax=ax_sparse, fraction=0.046, pad=0.04)
            
            # Col 2: Dense Output Matrix
            ax_dense = axes[i, 1]
            limit = np.max(np.abs(trace['dense_map'])) + 1e-6
            im2 = ax_dense.imshow(trace['dense_map'], cmap='RdBu_r',
                                  vmin=-limit, vmax=limit, aspect='auto')
            ax_dense.set_title(f"Layer {i} Contribution\n(Feature Grid)")
            ax_dense.axis('off')
            plt.colorbar(im2, ax=ax_dense, fraction=0.046, pad=0.04)
            
            # Col 3: Signal Power
            ax_stat = axes[i, 2]
            diff = trace['output_mag'] - trace['input_mag']
            color = 'green' if diff > 0 else 'red'
            
            ax_stat.bar(["Input Power", "Output Power"],
                       [trace['input_mag'], trace['output_mag']],
                       color=['gray', color])
            ax_stat.set_title(f"Layer {i} Signal Amplification")
            ax_stat.set_ylim(0, max(trace['input_mag'], trace['output_mag']) * 1.2)
            ax_stat.text(0.5, trace['output_mag'], f"{diff:+.2f}",
                        ha='center', va='bottom', fontweight='bold', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        
        # Show input
        plt.figure(figsize=(3, 3))
        img_show = img_tensor.squeeze(0).cpu()
        if use_panic_mode:
            img_show = 1.0 - img_show
        img_show = denormalize_imagenet(img_show)
        plt.imshow(img_show)
        plt.title(f"Input: {label_name}")
        plt.axis('off')
        plt.show()
    
    def bio_debug_suite(
        self,
        normal_loader: torch.utils.data.DataLoader,
        storm_loader: Optional[torch.utils.data.DataLoader] = None,
        layer_idx: int = 0
    ):
        """
        Comprehensive biological debugging suite inspired by fMRI analysis.
        
        Creates multiple visualizations:
        1. Functional Connectivity (co-activation heatmap)
        2. Neuron Specialization (what images activate each neuron)
        3. Sparse Code Dynamics (Normal vs Storm/Panic)
        
        Args:
            normal_loader: DataLoader for normal images
            storm_loader: Optional DataLoader for inverted/storm images
            layer_idx: Which layer to analyze (default: 0)
        """
        print("\n" + "=" * 50)
        print("      🧬 PLASTIC BRAIN fMRI SCAN")
        print("=" * 50)
        
        self.brain.eval()
        layer = self.brain.brain[layer_idx]
        
        # 1. Data Collection
        print(f"[1] Collecting Neural Activity from Layer {layer_idx}...")
        
        images, labels = next(iter(normal_loader))
        images = images.to(self.device)
        
        with torch.no_grad():
            feat = self.brain.eye(images)
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            x_norm = F.normalize(feat, p=2, dim=1)
            
            hidden = layer.encoder(x_norm)
            top_val, top_idx = torch.topk(hidden, layer.k, dim=1)
            sparse_act = torch.zeros_like(hidden).scatter_(1, top_idx, top_val)
            
            freqs = layer.freq_count.cpu().numpy()
            mature = layer.maturity_mask.cpu().numpy()
        
        # 2. Connectivity Map
        print("[2] Visualizing Functional Connectivity (Co-Activation)...")
        
        most_active_idx = np.argsort(freqs)[::-1][:50].copy()
        neuron_traces = sparse_act[:, most_active_idx].cpu().numpy()
        corr_matrix = np.corrcoef(neuron_traces.T + 1e-8)
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, cmap="coolwarm", center=0, square=True,
                   xticklabels=False, yticklabels=False)
        plt.title(f"Functional Connectivity (Top 50 Neurons, Layer {layer_idx})\n"
                 "Red = Hebbian Assemblies (Wire Together)")
        plt.xlabel("Neuron ID (Ranked by Activity)")
        plt.ylabel("Neuron ID")
        plt.show()
        
        print("  > Red squares indicate 'Assemblies' - neurons that fire together.")
        
        # 3. Neuron Specialization
        print(f"\n[3] Feature Visualization: What activates Layer {layer_idx} neurons?")
        
        batch_sum = sparse_act.sum(dim=0)
        candidates = ((batch_sum > 0) & (layer.maturity_mask == 1)).nonzero().view(-1)
        
        if len(candidates) < 5:
            print("  > Not enough active mature neurons for visualization.")
        else:
            selected_neurons = candidates[torch.randperm(len(candidates))[:5]]
            
            fig, axes = plt.subplots(5, 6, figsize=(12, 10))
            
            for row_idx, neur_idx in enumerate(selected_neurons):
                neur_id = neur_idx.item()
                
                activations = sparse_act[:, neur_id]
                top_acts, top_img_indices = torch.topk(activations, 5)
                
                ax_info = axes[row_idx, 0]
                ax_info.text(0.1, 0.5, f"Neuron #{neur_id}\nProtected\nMax: {top_acts[0]:.2f}",
                            fontsize=12)
                ax_info.axis('off')
                
                for col_idx, img_idx in enumerate(top_img_indices):
                    ax = axes[row_idx, col_idx + 1]
                    img = images[img_idx]
                    img = denormalize_imagenet(img)
                    
                    label = labels[img_idx].item()
                    name = self.class_names[label]
                    
                    ax.imshow(img)
                    ax.set_title(f"{name}\n{top_acts[col_idx]:.2f}", fontsize=9)
                    ax.axis('off')
            
            plt.suptitle(f"What do Layer {layer_idx} Protected Neurons 'See'?", y=1.02)
            plt.tight_layout()
            plt.show()
        
        # 4. Sparse Code Dynamics (if storm loader provided)
        if storm_loader is not None:
            print(f"\n[4] Sparse Code Comparison: Normal vs Inverted")
            
            storm_images, _ = next(iter(storm_loader))
            storm_images = storm_images.to(self.device)
            
            with torch.no_grad():
                feat_s = self.brain.eye(storm_images)
                if feat_s.dim() == 1:
                    feat_s = feat_s.unsqueeze(0)
                hidden_s = layer.encoder(F.normalize(feat_s, p=2, dim=1))
                hidden_panic = hidden_s * 2.0
            
            idx = 0
            plt.figure(figsize=(15, 4))
            
            plt.subplot(1, 3, 1)
            plt.plot(hidden[idx].cpu().numpy(), color='green', alpha=0.7)
            plt.title("Normal Image: Neural Output")
            plt.ylim(-0.1, 0.5)
            plt.ylabel("Activation")
            
            plt.subplot(1, 3, 2)
            plt.plot(hidden_s[idx].cpu().numpy(), color='red', alpha=0.7)
            plt.title("Inverted Image (Raw)")
            plt.ylim(-0.1, 0.5)
            plt.xlabel("Neuron Index")
            
            plt.subplot(1, 3, 3)
            plt.plot(hidden_panic[idx].cpu().numpy(), color='purple', alpha=0.7)
            plt.title("Inverted Image (Panic Mode)")
            plt.ylim(-0.1, 1.0)
            
            plt.tight_layout()
            plt.show()
            
            print("  > Panic Mode amplifies weak signals above the sparsity threshold.")
    
    def visualize_inference(
        self,
        loader: torch.utils.data.DataLoader,
        num_samples: int = 20,
        include_inverted: bool = True,
        invert_fn: Optional[callable] = None
    ):
        """
        Visual inference test showing predictions on sample images.
        
        Args:
            loader: DataLoader for normal images
            num_samples: Total number of samples to show
            include_inverted: Whether to include inverted versions
            invert_fn: Custom inversion function (default: 1 - x)
        """
        print("\n" + "=" * 50)
        print("      🔮 VISUAL INFERENCE TEST")
        print("=" * 50)
        
        self.brain.eval()
        
        if invert_fn is None:
            invert_fn = lambda x: 1.0 - x
        
        # Collect samples
        all_images = []
        all_labels = []
        all_types = []
        
        samples_per_type = num_samples // 2 if include_inverted else num_samples
        
        # Normal samples
        for img, lbl in loader:
            for i in range(min(len(img), samples_per_type - len(all_images))):
                all_images.append(img[i])
                all_labels.append(lbl[i].item())
                all_types.append("Normal")
            if len(all_images) >= samples_per_type:
                break
        
        # Inverted samples
        if include_inverted:
            for img, lbl in loader:
                for i in range(min(len(img), num_samples - len(all_images))):
                    all_images.append(invert_fn(img[i]))
                    all_labels.append(lbl[i].item())
                    all_types.append("Inverted")
                if len(all_images) >= num_samples:
                    break
        
        # Stack and predict
        x_all = torch.stack(all_images).to(self.device)
        
        with torch.no_grad():
            preds = self.brain.predict(x_all).cpu()
        
        # Plot
        cols = 5
        rows = (len(all_images) + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))
        axes = axes.flatten()
        
        correct = 0
        for i in range(len(all_images)):
            ax = axes[i]
            img = all_images[i]
            
            # Invert back for display if needed
            if all_types[i] == "Inverted":
                img = 1.0 - img
            
            img = denormalize_imagenet(img)
            
            true_name = self.class_names[all_labels[i]]
            pred_name = self.class_names[preds[i].item()]
            is_correct = preds[i].item() == all_labels[i]
            if is_correct:
                correct += 1
            
            color = 'green' if is_correct else 'red'
            
            ax.imshow(img)
            ax.set_title(f"{all_types[i]}\nT:{true_name}\nP:{pred_name}",
                        color=color, fontsize=10)
            ax.axis('off')
        
        # Hide unused axes
        for i in range(len(all_images), len(axes)):
            axes[i].axis('off')
        
        accuracy = 100 * correct / len(all_images)
        plt.suptitle(f"Inference Results: {accuracy:.1f}% Accuracy ({correct}/{len(all_images)})",
                    y=1.02, fontsize=16)
        plt.tight_layout()
        plt.show()
    
    def plot_layer_comparison(self):
        """
        Compare statistics across all layers.
        
        Creates bar charts showing:
        - Mature neuron counts
        - Gate values
        - Frequency statistics
        """
        num_layers = len(self.brain.brain)
        
        layer_stats = [layer.get_stats() for layer in self.brain.brain]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Mature neurons
        ax = axes[0]
        mature = [s['n_mature'] for s in layer_stats]
        ax.bar(range(num_layers), mature, color='purple')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Count")
        ax.set_title("Mature (Protected) Neurons")
        ax.set_xticks(range(num_layers))
        
        # Gate values
        ax = axes[1]
        gates = [s['gate_mean'] for s in layer_stats]
        ax.bar(range(num_layers), gates, color='orange')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Mean Gate")
        ax.set_title("Layer Gate Values")
        ax.set_xticks(range(num_layers))
        
        # Frequency
        ax = axes[2]
        freq_mean = [s['freq_mean'] for s in layer_stats]
        freq_max = [s['freq_max'] for s in layer_stats]
        x = np.arange(num_layers)
        width = 0.35
        ax.bar(x - width/2, freq_mean, width, label='Mean', color='blue')
        ax.bar(x + width/2, freq_max, width, label='Max', color='green')
        ax.set_xlabel("Layer")
        ax.set_ylabel("Frequency")
        ax.set_title("Neuron Activation Frequency")
        ax.set_xticks(range(num_layers))
        ax.legend()
        
        plt.tight_layout()
        plt.show()


# === LEGACY VISUALIZER (Backwards Compatibility) ===

class TriarchicVisualizer:
    """
    Legacy visualization suite for TriarchicBrain analysis.
    
    NOTE: For new projects, use PlasticVisualizer with PlasticBrain.
    """
    
    def __init__(self, brain, eyes, device='cpu'):
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
        """Visualize detailed brain activity for sample inputs."""
        if class_names is None:
            class_names = [
                'Plane', 'Car', 'Bird', 'Cat', 'Deer',
                'Dog', 'Frog', 'Horse', 'Ship', 'Truck'
            ]
        
        loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
        
        samples = []
        iter_data = iter(loader)
        for _ in range(num_samples):
            img, lbl = next(iter_data)
            if transform_fn is not None:
                img = transform_fn(img)
            samples.append((img, lbl))
        
        if figsize is None:
            figsize = (16, 4 * num_samples)
        fig = plt.figure(figsize=figsize)
        
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
        
        for i, (img, lbl) in enumerate(samples):
            img, lbl = img.to(self.device), lbl.item()
            
            with torch.no_grad():
                feats = self.eyes(img)
                sparse_h = self.brain.get_sparse_h(feats)
                
                core_logits = torch.mm(sparse_h, self.brain.w_core.t())
                context_logits = torch.mm(sparse_h, self.brain.w_context.t())
                total_logits = core_logits + context_logits
                
                probs = torch.softmax(total_logits, dim=1)
                pred_idx = torch.argmax(probs).item()
                confidence = probs[0, pred_idx].item()
            
            active_indices = torch.nonzero(sparse_h.squeeze()).flatten()
            n_active = len(active_indices)
            
            relevant_core = self.brain.w_core[pred_idx, active_indices]
            relevant_context = self.brain.w_context[pred_idx, active_indices]
            
            total_contribution = relevant_core + relevant_context
            sorted_vals, sorted_idx = torch.sort(total_contribution.abs(), descending=True)
            
            top_k = min(10, len(sorted_idx))
            top_10_idx = sorted_idx[:top_k]
            top_core = relevant_core[top_10_idx].cpu().numpy()
            top_context = relevant_context[top_10_idx].cpu().numpy()
            
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
            
            # Panel 2: Sparse Activity
            ax_scan = fig.add_subplot(num_samples, 3, (i * 3) + 2)
            activity_map = sparse_h.squeeze().cpu().numpy()
            
            side = int(np.ceil(np.sqrt(self.brain.hidden_size)))
            padded_map = np.zeros(side * side)
            padded_map[:len(activity_map)] = activity_map
            activity_grid = padded_map.reshape(side, side)
            
            sns.heatmap(activity_grid, ax=ax_scan, cbar=False, cmap="magma",
                       xticklabels=False, yticklabels=False)
            ax_scan.set_title(f"Sparse Neural Activity\n{n_active} / {self.brain.hidden_size} fired")
            
            # Panel 3: Synaptic Contribution
            ax_w = fig.add_subplot(num_samples, 3, (i * 3) + 3)
            indices = np.arange(top_k)
            
            ax_w.bar(indices, top_core, label='Core (Stable)', color='#1f77b4', alpha=0.8)
            ax_w.bar(indices, top_context, bottom=top_core,
                    label='Context (Fast)', color='#ff7f0e', alpha=0.9)
            
            ax_w.axhline(0, color='black', linewidth=0.5)
            ax_w.set_title(f"Top {top_k} Synaptic Contributors")
            ax_w.set_xlabel("Neuron Rank")
            ax_w.set_ylabel("Weight")
            ax_w.legend(loc='upper right', fontsize='small')
            ax_w.grid(alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()
    
    def compare_conditions(self, dataset, num_samples=4, class_names=None):
        """Compare brain behavior under normal vs perturbed conditions."""
        print("=" * 70)
        print("COMPARING NORMAL vs PERTURBED CONDITIONS")
        print("=" * 70)
        
        print("\n--- CONDITION 1: Normal Inputs ---")
        self.visualize_thinking(dataset, num_samples=num_samples,
                               class_names=class_names, transform_fn=None,
                               title="NORMAL CONDITIONS")
        
        print("\n--- CONDITION 2: Inverted Colors ---")
        self.visualize_thinking(dataset, num_samples=num_samples,
            class_names=class_names,
                               transform_fn=lambda x: x * -1.0,
                               title="PERTURBED CONDITIONS")
    
    def plot_weight_distribution(self):
        """Plot the distribution of Core and Context weights."""
        fig, axs = plt.subplots(1, 2, figsize=(12, 4))
        
        core_flat = self.brain.w_core.flatten().cpu().numpy()
        axs[0].hist(core_flat, bins=100, color='blue', alpha=0.7)
        axs[0].set_title("Core Weight Distribution")
        axs[0].set_xlabel("Weight Value")
        axs[0].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        context_flat = self.brain.w_context.flatten().cpu().numpy()
        axs[1].hist(context_flat, bins=100, color='orange', alpha=0.7)
        axs[1].set_title("Context Weight Distribution")
        axs[1].set_xlabel("Weight Value")
        axs[1].axvline(0, color='black', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        plt.show()
    
    def plot_importance_heatmap(self, class_names=None):
        """Visualize the importance matrix."""
        if class_names is None:
            class_names = [f"Class {i}" for i in range(self.brain.out_size)]
        
        importance = self.brain.importance.cpu().numpy()
        
        plt.figure(figsize=(12, 6))
        sns.heatmap(importance, cmap='YlOrRd',
                   cbar_kws={'label': 'Importance'},
                   yticklabels=class_names, xticklabels=False)
        plt.title("Synaptic Importance Matrix")
        plt.xlabel(f"Hidden Neurons (0 - {self.brain.hidden_size})")
        plt.ylabel("Output Classes")
        plt.tight_layout()
        plt.show()
