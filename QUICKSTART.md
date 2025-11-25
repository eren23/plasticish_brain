# Quick Start Guide

## Installation (30 seconds)

```bash
# Clone and install
git clone https://github.com/yourusername/plasticish_brain.git
cd plasticish_brain
pip install -r requirements.txt
```

## Minimal Example (v0.2.0 - PlasticBrain)

```python
from plasticish import PlasticBrain, PlasticTrainer, PhaseConfig
from examples.cifar10_utils import create_cifar10_loaders

device = 'cuda'  # or 'cpu'
loaders = create_cifar10_loaders(batch_size=256)

# Create multi-layer plastic brain
brain = PlasticBrain(num_layers=4, device=device)

# Define training phases
phases = [PhaseConfig("Vehicles", 3, loaders['train_vehicles'], mode="plastic")]

# Train
trainer = PlasticTrainer(brain, phases, device=device)
trainer.train()
trainer.plot_results()
```

## Run Complete Examples

```bash
# NEW: Multi-layer architecture (recommended)
python examples/train_multilayer.py

# LEGACY: Triarchic brain example
python examples/train_cifar10.py
```

This will:

1. Download CIFAR-10 automatically
2. Train on Vehicles → Animals → Storm (inverted colors)
3. Display accuracy graphs showing no catastrophic forgetting
4. Visualize multi-layer brain activity and memory usage

## What You'll See

### Training Output

```
==========================================================
TRIARCHIC BRAIN: CONTINUAL LEARNING SIMULATION
==========================================================

==========================================================
PHASE 1: ANIMALS
Duration: 1500 steps
==========================================================

Step    0 | Animals    | Task: 15.00% | Memory: 15.00%
Step   50 | Animals    | Task: 45.00% | Memory: 42.00%
Step  100 | Animals    | Task: 67.00% | Memory: 65.00%
...
Step 1450 | Animals    | Task: 76.00% | Memory: 75.00%

==========================================================
PHASE 2: VEHICLES
Duration: 1000 steps
==========================================================

Step 1500 | Vehicles   | Task: 35.00% | Memory: 71.00%  ← Memory retained!
Step 1550 | Vehicles   | Task: 72.00% | Memory: 70.00%
Step 2000 | Vehicles   | Task: 83.00% | Memory: 70.00%

==========================================================
PHASE 3: STORM
Duration: 1500 steps
==========================================================

Step 2500 | Storm!!    | Task: 42.00% | Memory: 69.00%  ← Adapting...
Step 2550 | Storm!!    | Task: 68.00% | Memory: 68.00%
Step 3500 | Storm!!    | Task: 78.00% | Memory: 68.00%
```

### Key Observation

**Memory accuracy stays ~70% throughout all phases!** This proves:

- No catastrophic forgetting
- Capacity expansion (animals + vehicles learned)
- Adaptation without permanent damage (storm phase)

## Understand the Architecture (3 diagrams)

### 1. Data Flow (v0.2.0 - PlasticBrain)

```
Image (3×224×224)
    ↓
PretrainedEyes (ResNet18, frozen)
    ↓
Features (512)
    ↓
┌─────────────────────────────────────┐
│         PLASTIC BRAIN               │
│  ┌─────────────────────────────────┐│
│  │   NeuromodulatedBlock (L0)     ││
│  │   Encoder → TopK → Decoder      ││
│  │   + Maturity + Neurogenesis     ││
│  └─────────────┬───────────────────┘│
│                ↓ (residual)         │
│  ┌─────────────────────────────────┐│
│  │   NeuromodulatedBlock (L1-N)   ││
│  │   ... (pluggable layers) ...    ││
│  └─────────────┬───────────────────┘│
│                ↓                    │
│  ┌─────────────────────────────────┐│
│  │   EpisodicMemoryBank (KNN)     ││
│  │   Store → Query → Soft Vote     ││
│  └─────────────────────────────────┘│
└──────────────┬──────────────────────┘
               ↓
        Output Logits (10 classes)
```

### 2. Learning Dynamics

```
                  Prediction from Memory
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
        Correct?                    Wrong?
            ↓                           ↓
      Reward = +1                Reward = -0.2
            ↓                           ↓
            └─────────────┬─────────────┘
                          ↓
               Modulated Hebbian Update
                    dW = R * Post * Pre
                          ↓
            ┌─────────────┴─────────────┐
            ↓                           ↓
      Mature Neurons              Plastic Neurons
      (Protected)                 (Full Update)
```

### 3. Three-Phase Experiment

```
Phase 1: VEHICLES (4 classes) - mode="plastic"
────────────────────────────────
[Plane] [Car] [Ship] [Truck]
    ↓
Layers: Learn vehicle features
Memory: Store vehicle examples
→ consolidate_after=True (protect neurons)

Phase 2: ANIMALS (6 classes) - mode="plastic"
────────────────────────────────
[Cat] [Dog] [Deer] [Bird] [Frog] [Horse]
    ↓
Neurogenesis: Recycle unused neurons
Memory: Add animal examples
Vehicles: Still remembered!

Phase 3: STORM (inverted colors) - mode="panic"
────────────────────────────────
[Inverted Vehicle Images]
    ↓
Panic Mode: Boost weak signals ×2
Neurons: Aggressive adaptation
Memory: Adapts quickly, no permanent damage
```

## Customize (5 examples)

### 1. Larger Brain with More Layers

```python
brain = PlasticBrain(
    num_layers=6,           # More layers
    hidden_dim=8192,        # Larger hidden dimension
    sparsity_k=256,         # More active neurons
    memory_size=100000,     # Larger memory
    device='cuda'
)
```

### 2. Dynamic Layer Management

```python
# Add a layer at runtime
brain.add_layer()

# Remove a specific layer
brain.remove_layer(2)

# Freeze a layer (disable plasticity)
brain.freeze_layer(0)
```

### 3. Custom Training Phases

```python
from plasticish.training import add_noise, blur_image

phases = [
    PhaseConfig("Clean", 3, clean_loader, mode="plastic", consolidate_after=True),
    PhaseConfig("Noisy", 3, clean_loader, mode="panic", transform_fn=add_noise),
    PhaseConfig("Blurry", 2, clean_loader, mode="panic", transform_fn=blur_image),
]
```

### 4. Multi-Layer Visualization

```python
from plasticish import PlasticVisualizer

visualizer = PlasticVisualizer(brain, device='cuda')

# Trace a thought through all layers
visualizer.trace_thought(storm_loader, use_panic_mode=True)

# 2D matrix visualization
visualizer.trace_thought_matrix(loader)

# Bio-inspired fMRI-style analysis
visualizer.bio_debug_suite(normal_loader, storm_loader, layer_idx=0)

# Compare layer statistics
visualizer.plot_layer_comparison()
```

### 5. Legacy TriarchicBrain Customization

```python
from plasticish import TriarchicBrain, TriarchicVisualizer

brain = TriarchicBrain(512, 16384, 10, 64, device='cuda')
brain.context_lr = 0.8      # Higher → faster adaptation
brain.core_lr = 0.02        # Lower → slower changes

visualizer = TriarchicVisualizer(brain, eyes, device='cuda')
visualizer.compare_conditions(dataset, num_samples=4)
```

## Next Steps

1. **Experiment**: Try `examples/train_multilayer.py` with different configurations
2. **Read**: `README.md` for theoretical background and architecture details
3. **Explore**: `plasticish/models.py` for implementation details
4. **Notebook**: Try `examples/plasticish_brain_example.ipynb` for interactive learning
5. **Extend**: Create your own phases, transforms, and layer configurations
6. **Contribute**: Add new features, benchmarks, or applications

## Common Issues

### Import Error

```python
ModuleNotFoundError: No module named 'plasticish'
```

**Solution**: Run `pip install -e .` from the project root

### CUDA Out of Memory

**Solution**: Reduce batch size or hidden size

```python
brain = TriarchicBrain(512, 4096, 10, 16, device='cuda')  # Smaller
```

### Slow Training

**Solution**: Use GPU if available

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

## Project Structure

```
plasticish_brain/
├── plasticish/                     # Core Library
│   ├── __init__.py                 # Public API (v0.2.0)
│   ├── models.py                   # PlasticBrain, NeuromodulatedBlock, etc.
│   ├── training.py                 # PlasticTrainer, PhaseConfig, utilities
│   └── visualization.py            # PlasticVisualizer, analysis tools
│
├── examples/                       # Usage Examples
│   ├── train_multilayer.py         # PlasticBrain example (RECOMMENDED)
│   ├── train_cifar10.py            # Legacy TriarchicBrain example
│   ├── cifar10_utils.py            # Data loading utilities
│   └── plasticish_brain_example.ipynb  # Jupyter notebook
│
├── assets/                         # Documentation images
├── data/                           # Auto-downloaded datasets
├── README.md                       # Full documentation
├── PROJECT_OVERVIEW.md             # Project overview
└── QUICKSTART.md                   # This file
```

## Resources

- **Full Documentation**: `README.md`
- **Project Overview**: `PROJECT_OVERVIEW.md`
- **Multi-Layer Example**: `examples/train_multilayer.py`
- **Legacy Example**: `examples/train_cifar10.py`
- **Interactive Notebook**: `examples/plasticish_brain_example.ipynb`
- **Theory & References**: See README.md References section

## Performance Expectations

### CIFAR-10 (RTX 3090)

- Training: ~10 minutes (4000 steps)
- Memory: ~2GB GPU
- Final accuracies:
  - Animals: 75%
  - Vehicles: 83%
  - Memory retention: 70%

### CIFAR-10 (CPU)

- Training: ~45 minutes
- Memory: ~4GB RAM
- Same accuracy (just slower)

---

**Ready to start?** Run:

```bash
python examples/train_cifar10.py
```

Happy experimenting!
