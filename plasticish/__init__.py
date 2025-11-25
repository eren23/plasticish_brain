"""
Plasticish Brain: A Bio-Inspired Continual Learning Library

This library implements a modular neural architecture for lifelong learning
without catastrophic forgetting, inspired by biological neural plasticity.

v0.2.0 - Multi-Layer Pluggable Architecture:
- NeuromodulatedBlock: Pluggable plastic layers with local Hebbian learning
- EpisodicMemoryBank: KNN-based hippocampal memory system
- PlasticBrain: Full network with dynamic layer management
- PlasticTrainer: Multi-phase training with plastic/panic modes
- PlasticVisualizer: Multi-layer activity visualization

Key Features:
- Pluggable layers: Add/remove/replace layers dynamically
- Training modes: "plastic" (normal) and "panic" (storm adaptation)
- Self-supervised reward: Learning from memory predictions
- Neurogenesis: Dead neurons are recycled for new concepts
- Maturity protection: Frequently used neurons become consolidated
- Episodic memory: One-shot learning via KNN retrieval

Quick Start:
    >>> from plasticish import PlasticBrain, PlasticTrainer, PhaseConfig
    >>> 
    >>> # Create model
    >>> brain = PlasticBrain(num_layers=4, device='cuda')
    >>> 
    >>> # Define training phases
    >>> phases = [
    ...     PhaseConfig("Phase1", 3, train_loader, mode="plastic"),
    ...     PhaseConfig("Phase2", 3, other_loader, mode="panic"),
    ... ]
    >>> 
    >>> # Train
    >>> trainer = PlasticTrainer(brain, phases, device='cuda')
    >>> history = trainer.train()

See examples/ directory for complete CIFAR-10 training examples.
"""

# === MODELS ===
from .models import (
    PretrainedEyes,
    NeuromodulatedBlock,
    EpisodicMemoryBank,
    PlasticBrain,
)

# === TRAINING ===
from .training import (
    PhaseConfig,
    TrainingHistory,
    PlasticTrainer,
    # Generic utilities
    invert_colors,
    invert_tensor,
    add_noise,
    blur_image,
)

# === VISUALIZATION ===
from .visualization import (
    PlasticVisualizer,
    denormalize_imagenet,
)

# === LEGACY SUPPORT (v0.1.x) ===
from .models import TriarchicBrain
from .training import TriarchicTrainer
from .visualization import TriarchicVisualizer

__version__ = "0.2.0"

__all__ = [
    # Core Models
    "PretrainedEyes",
    "NeuromodulatedBlock",
    "EpisodicMemoryBank",
    "PlasticBrain",
    
    # Training
    "PhaseConfig",
    "TrainingHistory",
    "PlasticTrainer",
    
    # Visualization
    "PlasticVisualizer",
    "denormalize_imagenet",
    
    # Utilities
    "invert_colors",
    "invert_tensor",
    "add_noise",
    "blur_image",
    
    # Legacy
    "TriarchicBrain",
    "TriarchicTrainer",
    "TriarchicVisualizer",
]
