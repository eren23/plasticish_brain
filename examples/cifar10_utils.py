"""
CIFAR-10 Dataset Utilities for Plasticish Brain Examples

This module provides CIFAR-10 specific data loading and preprocessing
for the continual learning examples.
"""

import torch
import torchvision
import torchvision.transforms as transforms
from typing import Dict, List, Any, Optional


# CIFAR-10 Class Information
CLASS_NAMES = ['Plane', 'Car', 'Bird', 'Cat', 'Deer',
               'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

VEHICLE_CLASSES = [0, 1, 8, 9]  # Plane, Car, Ship, Truck
ANIMAL_CLASSES = [2, 3, 4, 5, 6, 7]  # Bird, Cat, Deer, Dog, Frog, Horse


class InvertTensor:
    """Transform that inverts tensor values (1 - x)."""
    def __call__(self, img):
        return 1.0 - img


def get_transforms(for_storm: bool = False):
    """
    Get image transforms for CIFAR-10.
    
    Args:
        for_storm: If True, include inversion before normalization
    
    Returns:
        torchvision.transforms.Compose
    """
    base = [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ]
    
    if for_storm:
        base.append(InvertTensor())
    
    base.append(transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ))
    
    return transforms.Compose(base)


def get_inverse_normalize():
    """Get transform to reverse ImageNet normalization for display."""
    return transforms.Normalize(
        mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
        std=[1/0.229, 1/0.224, 1/0.225]
    )


def create_subset_loader(
    dataset,
    class_indices: List[int],
    batch_size: int = 64,
    shuffle: bool = True,
    num_workers: int = 0
):
    """
    Create a DataLoader for a subset of classes.
    
    Args:
        dataset: Full CIFAR-10 dataset
        class_indices: List of class indices to include
        batch_size: Batch size
        shuffle: Whether to shuffle
        num_workers: Number of data loading workers
    
    Returns:
        DataLoader
    """
    targets = torch.tensor(dataset.targets)
    mask = torch.tensor([t.item() in class_indices for t in targets])
    subset = torch.utils.data.Subset(dataset, mask.nonzero().view(-1))
    return torch.utils.data.DataLoader(
        subset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers
    )


def create_cifar10_loaders(
    batch_size: int = 64,
    data_root: str = './data',
    num_workers: int = 0,
    include_storm: bool = True
) -> Dict[str, Any]:
    """
    Create all standard CIFAR-10 data loaders for continual learning.
    
    Returns loaders for:
    - Vehicles (train/test)
    - Animals (train/test)  
    - Storm/Inverted vehicles (optional)
    - Mixed (all classes)
    
    Args:
        batch_size: Batch size for all loaders
        data_root: Directory to download/load CIFAR-10
        num_workers: Number of data loading workers
        include_storm: Whether to create storm (inverted) loaders
    
    Returns:
        Dictionary with all loaders and metadata:
        {
            'train_vehicles': DataLoader,
            'train_animals': DataLoader,
            'test_vehicles': DataLoader,
            'test_animals': DataLoader,
            'test_mixed': DataLoader,
            'storm_vehicles': DataLoader (if include_storm),
            'train_full': Dataset,
            'test_full': Dataset,
            'class_names': List[str],
            'vehicle_classes': List[int],
            'animal_classes': List[int],
        }
    """
    # Load datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, 
        transform=get_transforms(for_storm=False)
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True,
        transform=get_transforms(for_storm=False)
    )
    
    result = {
        # Training loaders
        'train_vehicles': create_subset_loader(train_dataset, VEHICLE_CLASSES, batch_size, True, num_workers),
        'train_animals': create_subset_loader(train_dataset, ANIMAL_CLASSES, batch_size, True, num_workers),
        
        # Test loaders
        'test_vehicles': create_subset_loader(test_dataset, VEHICLE_CLASSES, batch_size, False, num_workers),
        'test_animals': create_subset_loader(test_dataset, ANIMAL_CLASSES, batch_size, False, num_workers),
        'test_mixed': create_subset_loader(test_dataset, VEHICLE_CLASSES + ANIMAL_CLASSES, batch_size, False, num_workers),
        
        # Full datasets
        'train_full': train_dataset,
        'test_full': test_dataset,
        
        # Metadata
        'class_names': CLASS_NAMES,
        'vehicle_classes': VEHICLE_CLASSES,
        'animal_classes': ANIMAL_CLASSES,
    }
    
    # Storm loader (inverted vehicles)
    if include_storm:
        storm_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=get_transforms(for_storm=True)
        )
        result['storm_vehicles'] = create_subset_loader(
            storm_dataset, VEHICLE_CLASSES, batch_size, True, num_workers
        )
    
    return result


def print_loader_info(loaders: Dict[str, Any]):
    """Print information about the created loaders."""
    print("=" * 50)
    print("CIFAR-10 Data Loaders")
    print("=" * 50)
    
    for key, value in loaders.items():
        if isinstance(value, torch.utils.data.DataLoader):
            print(f"  {key}: {len(value.dataset)} samples")
        elif key == 'class_names':
            print(f"  Classes: {value}")
    
    print("=" * 50)

