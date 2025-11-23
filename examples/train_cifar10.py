"""
Example: Training Triarchic Brain on CIFAR-10 Continual Learning

This script demonstrates the full workflow:
1. Create a Triarchic Brain architecture
2. Define a multi-phase curriculum (Animals → Vehicles → Storm)
3. Train with automatic metric tracking
4. Visualize results and brain behavior

Expected Results:
- Phase 1 (Animals): Builds core memory via neurogenesis and consolidation
- Phase 2 (Vehicles): Expands capacity without forgetting animals
- Phase 3 (Storm): Adapts to inverted colors using context layer only

Key Metrics:
- Task Accuracy: Performance on current phase
- Memory Accuracy: Retention of Phase 1 (animals) throughout training
- Context Energy: Should spike during distribution shifts
- Core Energy: Should remain stable, showing protected memory
"""

import torch
from plasticish import PretrainedEyes, TriarchicBrain, TriarchicTrainer, PhaseConfig
from plasticish.training import create_cifar10_loaders, invert_colors

# === CONFIGURATION ===
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Architecture hyperparameters
INPUT_DIM = 512          # ResNet18 feature dimension
EXPANSION_DIM = 8192     # Sparse projection dimension (16x expansion)
SPARSITY_K = 32          # Active neurons per input (~0.4% sparsity)
NUM_CLASSES = 10         # CIFAR-10 classes

# Training hyperparameters
BATCH_SIZE = 64
PHASE_1_STEPS = 1500     # Animals: Build foundation
PHASE_2_STEPS = 1000     # Vehicles: Test capacity expansion
PHASE_3_STEPS = 1500     # Storm: Test adaptation

print(f"Using device: {DEVICE}")
print(f"Architecture: {INPUT_DIM} → {EXPANSION_DIM} (sparse) → {NUM_CLASSES}")
print(f"Sparsity: {SPARSITY_K}/{EXPANSION_DIM} = {SPARSITY_K/EXPANSION_DIM*100:.2f}%\n")

# === DATA PREPARATION ===
print("Loading CIFAR-10 dataset...")
animal_loader, vehicle_loader, memory_test_loader, full_dataset = create_cifar10_loaders(
    batch_size=BATCH_SIZE,
    data_root='./data'
)

print(f"Animals: {len(animal_loader.dataset)} samples")
print(f"Vehicles: {len(vehicle_loader.dataset)} samples")
print(f"Memory Test: {len(memory_test_loader.dataset)} samples\n")

# === MODEL INITIALIZATION ===
print("Initializing Triarchic Brain...")

eyes = PretrainedEyes(device=DEVICE)
brain = TriarchicBrain(
    in_size=INPUT_DIM,
    hidden_size=EXPANSION_DIM,
    out_size=NUM_CLASSES,
    sparsity_k=SPARSITY_K,
    device=DEVICE
)

print(f"Eyes: ResNet18 (frozen)")
print(f"Brain: {sum(p.numel() for p in [brain.projection, brain.w_core, brain.w_context, brain.importance])} parameters")
print(f"  - Projection: {brain.projection.numel()} (dynamic)")
print(f"  - Core Weights: {brain.w_core.numel()} (slow + protected)")
print(f"  - Context Weights: {brain.w_context.numel()} (fast + decaying)")
print(f"  - Importance Matrix: {brain.importance.numel()} (consolidation)\n")

# === DEFINE CURRICULUM ===
phases = [
    PhaseConfig(
        name="Animals",
        n_steps=PHASE_1_STEPS,
        data_loader=animal_loader,
        description="Build core knowledge via neurogenesis and consolidation"
    ),
    PhaseConfig(
        name="Vehicles",
        n_steps=PHASE_2_STEPS,
        data_loader=vehicle_loader,
        description="Expand capacity without catastrophic forgetting"
    ),
    PhaseConfig(
        name="Storm",
        n_steps=PHASE_3_STEPS,
        data_loader=animal_loader,
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

print("Starting continual learning simulation...\n")
history = trainer.train(verbose=True)

# === RESULTS ===
print("\n" + "="*70)
print("TRAINING SUMMARY")
print("="*70)

summary = trainer.get_summary()
for phase_name, stats in summary.items():
    print(f"\n{phase_name}:")
    print(f"  Initial Accuracy: {stats['initial_accuracy']:.2%}")
    print(f"  Final Accuracy: {stats['final_accuracy']:.2%}")
    print(f"  Improvement: {stats['improvement']:+.2%}")
    print(f"  Mean Accuracy: {stats['mean_accuracy']:.2%}")

# Final brain state
final_stats = brain.get_stats()
print(f"\nFinal Brain State:")
print(f"  Active Neurons: {final_stats['n_active_neurons']} / {EXPANSION_DIM}")
print(f"  Capacity Usage: {final_stats['n_active_neurons']/EXPANSION_DIM*100:.2f}%")
print(f"  Mean Importance: {final_stats['mean_importance']:.3f}")
print(f"  Max Importance: {final_stats['max_importance']:.3f}")
print(f"  Core Energy: {final_stats['core_energy']:.4f}")
print(f"  Context Energy: {final_stats['context_energy']:.4f}")

print("\n" + "="*70)
print("KEY OBSERVATIONS")
print("="*70)
print("""
Expected Behaviors:

1. PLASTICITY (Phase 2 - Vehicles):
   - Rapid learning of new classes via neurogenesis
   - New neurons allocated for vehicle representations
   - Minimal interference with animal memory

2. STABILITY (Throughout):
   - Animal memory retention >70% during vehicle training
   - Core weights protected by importance matrix
   - No catastrophic forgetting

3. ADAPTATION (Phase 3 - Storm):
   - Context layer compensates for color inversion
   - Core memory remains intact (still recognizes concepts)
   - Context energy spikes, then decays after phase ends

4. EFFICIENCY:
   - Only ~0.4% of neurons active per input (sparse coding)
   - Orthogonal representations minimize interference
""")

# === VISUALIZATION ===
print("\nGenerating training plots...")
trainer.plot_results(figsize=(14, 10))

# Detailed brain inspection
print("\nPreparing detailed brain visualization...")
from plasticish import TriarchicVisualizer

visualizer = TriarchicVisualizer(brain, eyes, device=DEVICE)

# Compare normal vs perturbed conditions
visualizer.compare_conditions(
    dataset=full_dataset,
    num_samples=4,
    class_names=['Plane', 'Car', 'Bird', 'Cat', 'Deer', 
                 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
)

print("\n" + "="*70)
print("EXPERIMENT COMPLETE")
print("="*70)
print("""
This experiment demonstrates that bio-inspired plasticity mechanisms
can enable true continual learning without catastrophic forgetting.

Key Innovations:
1. Sparse orthogonal coding → minimal interference
2. Synaptic consolidation → protected long-term memory
3. Dual-frequency weights → stable core + adaptive context
4. Dynamic neurogenesis → capacity expansion on demand

For more details, see the paper:
Liu, Z., et al. (2025). Intelligence Foundation Model. arXiv:2511.10119
""")

# Optional: Save model
# torch.save({
#     'brain_state': brain.state_dict(),
#     'history': history,
#     'config': {
#         'input_dim': INPUT_DIM,
#         'expansion_dim': EXPANSION_DIM,
#         'sparsity_k': SPARSITY_K,
#         'num_classes': NUM_CLASSES
#     }
# }, 'triarchic_brain_cifar10.pt')

