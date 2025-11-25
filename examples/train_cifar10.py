"""
Example: Training Triarchic Brain on CIFAR-10 (Legacy API)

This script uses the legacy TriarchicBrain API for backwards compatibility.
For new projects, see train_multilayer.py which uses the newer PlasticBrain.

Three-Phase Continual Learning:
1. Animals: Build core memory via neurogenesis and consolidation
2. Vehicles: Expand capacity without forgetting animals
3. Storm: Adapt to inverted colors using context layer only

Run:
    python examples/train_cifar10.py
"""

import torch
from plasticish import PretrainedEyes, TriarchicBrain, TriarchicTrainer, PhaseConfig
from plasticish.training import invert_colors

# Import CIFAR-10 utilities
from cifar10_utils import create_cifar10_loaders, print_loader_info


def main():
    # === CONFIGURATION ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Architecture
    INPUT_DIM = 512
    EXPANSION_DIM = 8192
    SPARSITY_K = 32
    NUM_CLASSES = 10
    
    # Training
    BATCH_SIZE = 64
    PHASE_1_EPOCHS = 3
    PHASE_2_EPOCHS = 2
    PHASE_3_EPOCHS = 3
    
    print(f"Using device: {DEVICE}")
    print(f"Architecture: {INPUT_DIM} → {EXPANSION_DIM} (sparse) → {NUM_CLASSES}")
    print(f"Sparsity: {SPARSITY_K}/{EXPANSION_DIM} = {SPARSITY_K/EXPANSION_DIM*100:.2f}%\n")
    
    # === DATA LOADING ===
    print("Loading CIFAR-10...")
    loaders = create_cifar10_loaders(batch_size=BATCH_SIZE, data_root='./data')
    print_loader_info(loaders)
    
    # Create memory test loader (subset of animals for quick eval)
    memory_test_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(loaders['train_full'], range(500)),
        batch_size=100,
        shuffle=False
    )
    
    # === MODEL INITIALIZATION ===
    print("\nInitializing Triarchic Brain...")
    eyes = PretrainedEyes(device=DEVICE)
    brain = TriarchicBrain(
        in_size=INPUT_DIM,
        hidden_size=EXPANSION_DIM,
        out_size=NUM_CLASSES,
        sparsity_k=SPARSITY_K,
        device=DEVICE
    )
    
    print(f"Eyes: ResNet18 (frozen)")
    print(f"Brain projection: {brain.projection.numel()} parameters")
    
    # === DEFINE CURRICULUM ===
    phases = [
        PhaseConfig(
            name="Animals",
            n_epochs=PHASE_1_EPOCHS,
            data_loader=loaders['train_animals'],
            description="Build core knowledge via neurogenesis and consolidation"
        ),
        PhaseConfig(
            name="Vehicles",
            n_epochs=PHASE_2_EPOCHS,
            data_loader=loaders['train_vehicles'],
            description="Expand capacity without catastrophic forgetting"
        ),
        PhaseConfig(
            name="Storm",
            n_epochs=PHASE_3_EPOCHS,
            data_loader=loaders['train_animals'],
            transform_fn=invert_colors,
            description="Adapt to inverted colors using context layer"
        )
    ]
    
    # === TRAINING ===
    trainer = TriarchicTrainer(
        brain=brain,
        eyes=eyes,
        phases=phases,
        memory_test_loader=memory_test_loader,
        eval_interval=50,
        device=DEVICE
    )
    
    print("\nStarting continual learning simulation...\n")
    history = trainer.train(verbose=True)
    
    # === RESULTS ===
    print("\n" + "=" * 70)
    print("TRAINING SUMMARY")
    print("=" * 70)
    
    summary = trainer.get_summary()
    for phase_name, stats in summary.items():
        print(f"\n{phase_name}:")
        print(f"  Initial Accuracy: {stats['initial_accuracy']:.2%}")
        print(f"  Final Accuracy: {stats['final_accuracy']:.2%}")
        print(f"  Mean Accuracy: {stats['mean_accuracy']:.2%}")
    
    # Final brain state
    final_stats = brain.get_stats()
    print(f"\nFinal Brain State:")
    print(f"  Active Neurons: {final_stats['n_active_neurons']} / {EXPANSION_DIM}")
    print(f"  Core Energy: {final_stats['core_energy']:.4f}")
    print(f"  Context Energy: {final_stats['context_energy']:.4f}")
    print(f"  Mean Importance: {final_stats['mean_importance']:.3f}")
    
    # === VISUALIZATION ===
    print("\nGenerating plots...")
    trainer.plot_results(figsize=(14, 10))
    
    # Detailed visualization
    from plasticish import TriarchicVisualizer
    visualizer = TriarchicVisualizer(brain, eyes, device=DEVICE)
    visualizer.compare_conditions(
        dataset=loaders['test_full'],
        num_samples=4,
        class_names=loaders['class_names']
    )
    
    print("\n" + "=" * 70)
    print("EXPERIMENT COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
