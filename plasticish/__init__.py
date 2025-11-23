"""
Plasticish Brain: A Bio-Inspired Continual Learning Library

This library implements a triarchic neural architecture for lifelong learning
without catastrophic forgetting, inspired by biological neural plasticity.

Key Components:
- PretrainedEyes: Feature extraction using frozen pretrained models
- TriarchicBrain: Three-layer adaptive architecture with:
  1. Structural neurogenesis (capacity expansion)
  2. Synaptic consolidation (stable long-term memory)
  3. Fast context adaptation (temporary plasticity)
"""

from .models import PretrainedEyes, TriarchicBrain
from .training import TriarchicTrainer, PhaseConfig
from .visualization import TriarchicVisualizer

__version__ = "0.1.0"
__all__ = [
    "PretrainedEyes",
    "TriarchicBrain",
    "TriarchicTrainer",
    "PhaseConfig",
    "TriarchicVisualizer"
]

