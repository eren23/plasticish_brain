"""
Plasticish Brain Examples

This package contains example scripts and utilities for training
the Plastic Brain architecture on various datasets.

Available Examples:
- train_multilayer.py: New PlasticBrain with pluggable layers (recommended)
- train_cifar10.py: Legacy TriarchicBrain example

Utilities:
- cifar10_utils: CIFAR-10 data loading and preprocessing
"""

from .cifar10_utils import (
    create_cifar10_loaders,
    create_subset_loader,
    get_transforms,
    get_inverse_normalize,
    print_loader_info,
    CLASS_NAMES,
    VEHICLE_CLASSES,
    ANIMAL_CLASSES,
)

__all__ = [
    "create_cifar10_loaders",
    "create_subset_loader",
    "get_transforms",
    "get_inverse_normalize",
    "print_loader_info",
    "CLASS_NAMES",
    "VEHICLE_CLASSES",
    "ANIMAL_CLASSES",
]
