"""
Example: Training PlasticBrain with Multi-Layer Architecture

This script demonstrates the new pluggable multi-layer architecture:
1. Create a PlasticBrain with configurable layers
2. Dynamically add/remove layers
3. Train with different modes (plastic/panic)
4. Visualize multi-layer activity

Three-Phase Continual Learning:
- Phase 1 (Vehicles): Learn vehicle categories
- Phase 2 (Animals): Learn animal categories without forgetting vehicles
- Phase 3 (Storm): Adapt to inverted images using panic mode

Run:
    python examples/train_multilayer.py
"""

import torch
from plasticish import PlasticBrain, PlasticTrainer, PlasticVisualizer, PhaseConfig
from plasticish.training import invert_colors

# Import CIFAR-10 utilities from examples
from cifar10_utils import create_cifar10_loaders, print_loader_info, CLASS_NAMES


def main():
    # === CONFIGURATION ===
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Architecture
    NUM_LAYERS = 4
    HIDDEN_DIM = 4096
    SPARSITY_K = 128
    LR_BRAIN = 0.04
    
    # Memory
    MEMORY_SIZE = 60000
    K_NEIGHBORS = 50
    
    # Training
    BATCH_SIZE = 256
    NUM_EPOCHS = 3
    
    print(f"🧠 Plastic Brain Training")
    print(f"   Device: {DEVICE}")
    print(f"   Layers: {NUM_LAYERS}")
    print(f"   Hidden: {HIDDEN_DIM}")
    print(f"   Sparsity: {SPARSITY_K}")
    print()
    
    # === DATA LOADING ===
    print("Loading CIFAR-10...")
    loaders = create_cifar10_loaders(
        batch_size=BATCH_SIZE,
        data_root='./data',
        include_storm=True
    )
    print_loader_info(loaders)
    
    # === MODEL CREATION ===
    print("\nInitializing PlasticBrain...")
    brain = PlasticBrain(
        in_dim=512,
        hidden_dim=HIDDEN_DIM,
        sparsity_k=SPARSITY_K,
        num_layers=NUM_LAYERS,
        memory_size=MEMORY_SIZE,
        k_neighbors=K_NEIGHBORS,
        num_classes=10,
        lr_brain=LR_BRAIN,
        device=DEVICE
    )
    
    print(f"  Created brain with {brain.num_layers} layers")
    print(f"  Memory capacity: {MEMORY_SIZE}")
    
    # === DEMONSTRATE PLUGGABLE LAYERS ===
    print("\n--- Layer Management Demo ---")
    print(f"  Initial layers: {brain.num_layers}")
    
    # Add a layer
    brain.add_layer()
    print(f"  After add_layer(): {brain.num_layers}")
    
    # Remove a layer
    brain.remove_layer(0)
    print(f"  After remove_layer(0): {brain.num_layers}")
    
    # Reset to original
    while brain.num_layers > NUM_LAYERS:
        brain.remove_layer(-1)
    while brain.num_layers < NUM_LAYERS:
        brain.add_layer()
    print(f"  Reset to: {brain.num_layers} layers")
    
    # === DEFINE TRAINING PHASES ===
    phases = [
        PhaseConfig(
            name="Vehicles",
            n_epochs=NUM_EPOCHS,
            data_loader=loaders['train_vehicles'],
            mode="plastic",
            memorize=True,
            consolidate_after=True,
            description="Learn vehicle categories (Plane, Car, Ship, Truck)"
        ),
        PhaseConfig(
            name="Animals",
            n_epochs=NUM_EPOCHS,
            data_loader=loaders['train_animals'],
            mode="plastic",
            memorize=True,
            consolidate_after=True,
            description="Learn animal categories without forgetting vehicles"
        ),
        PhaseConfig(
            name="Storm",
            n_epochs=NUM_EPOCHS,
            data_loader=loaders['storm_vehicles'],
            mode="panic",
            memorize=True,
            consolidate_after=False,
            description="Adapt to inverted images using panic mode"
        ),
    ]
    
    # === TRAINING ===
    trainer = PlasticTrainer(
        brain=brain,
        phases=phases,
        eval_loaders={
            'Vehicles': loaders['test_vehicles'],
            'Animals': loaders['test_animals'],
        },
        eval_interval=50,
        device=DEVICE
    )
    
    print("\n" + "=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)
    
    history = trainer.train(verbose=True)
    
    # === RESULTS ===
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    
    summary = trainer.get_summary()
    for phase_name, stats in summary.items():
        print(f"\n{phase_name}:")
        print(f"  Initial: {stats['initial_accuracy']:.2%}")
        print(f"  Final:   {stats['final_accuracy']:.2%}")
        print(f"  Mean:    {stats['mean_accuracy']:.2%}")
    
    # Final evaluation
    print("\n--- Final Evaluation ---")
    for name, loader in [('Vehicles', loaders['test_vehicles']), 
                         ('Animals', loaders['test_animals']),
                         ('Storm', loaders['storm_vehicles'])]:
        acc = brain.evaluate(loader)
        print(f"  {name}: {acc:.2f}%")
    
    # Brain stats
    stats = brain.get_stats()
    print(f"\n--- Brain State ---")
    print(f"  Memory fill: {stats['memory_fill']:.1%}")
    print(f"  Layers: {stats['num_layers']}")
    for i, layer_stats in stats['layers'].items():
        print(f"    {i}: {layer_stats['n_mature']} mature, gate={layer_stats['gate_mean']:.4f}")
    
    # === VISUALIZATION ===
    print("\n--- Generating Plots ---")
    trainer.plot_results()
    
    # Multi-layer visualization
    print("\n--- Layer-by-Layer Analysis ---")
    visualizer = PlasticVisualizer(brain, device=DEVICE, class_names=CLASS_NAMES)
    
    # Trace a single thought through layers
    visualizer.trace_thought(loaders['storm_vehicles'], use_panic_mode=True)
    
    # Compare layers
    visualizer.plot_layer_comparison()
    
    # Bio debug suite
    visualizer.bio_debug_suite(
        loaders['test_vehicles'],
        loaders['storm_vehicles'],
        layer_idx=0
    )
    
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()

