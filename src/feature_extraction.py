"""
Feature Extraction Module using Pre-trained CNNs
=================================================

This module extracts deep features from thermographic images using
pre-trained CNN architectures (EfficientNet-B0, ResNet50).

The extracted features are then used with an SVM classifier for
the final classification task.

Key Functions:
    - extract_features: Extract features from images using CNN
    - get_feature_extractor: Get CNN model configured for extraction
"""

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
from typing import Tuple, List, Optional, Union
from tqdm import tqdm
import albumentations as A

from .preprocessing import (
    IMAGENET_MEAN,
    IMAGENET_STD,
    get_augmentation_transform,
    apply_augmentation
)


# Feature dimensions for each architecture
FEATURE_DIMS = {
    'efficientnet_b0': 1280,
    'resnet50': 2048,
}


def get_device() -> torch.device:
    """
    Get the best available device for computation.

    Priority: MPS (Apple Silicon) > CUDA > CPU

    Returns:
        torch.device object
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Apple Silicon GPU)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"Using CUDA: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def get_feature_extractor(
    model_name: str = 'efficientnet_b0',
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, int]:
    """
    Get a pre-trained CNN configured as a feature extractor.

    The classification head is removed, leaving only the feature
    extraction layers that output a feature vector.

    Args:
        model_name: 'efficientnet_b0' or 'resnet50'
        device: Computation device (auto-detected if None)

    Returns:
        model: Feature extractor model
        num_features: Dimension of output feature vector
    """
    if device is None:
        device = get_device()

    if model_name == 'efficientnet_b0':
        # Load pre-trained EfficientNet-B0
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1
        base_model = models.efficientnet_b0(weights=weights)

        # Remove classifier, keep features + avgpool
        # EfficientNet structure: features -> avgpool -> classifier
        model = nn.Sequential(
            base_model.features,
            base_model.avgpool,
            nn.Flatten()
        )
        num_features = FEATURE_DIMS['efficientnet_b0']

    elif model_name == 'resnet50':
        # Load pre-trained ResNet50
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        base_model = models.resnet50(weights=weights)

        # Remove the final FC layer
        # ResNet structure: conv layers -> avgpool -> fc
        modules = list(base_model.children())[:-1]  # Remove fc layer
        model = nn.Sequential(*modules, nn.Flatten())
        num_features = FEATURE_DIMS['resnet50']

    else:
        raise ValueError(f"Unknown model: {model_name}. "
                        f"Supported: {list(FEATURE_DIMS.keys())}")

    # Move to device and set to evaluation mode
    model = model.to(device)
    model.eval()

    print(f"Loaded {model_name} feature extractor")
    print(f"  Output features: {num_features}")
    print(f"  Device: {device}")

    return model, num_features


def preprocess_for_cnn(
    images: np.ndarray,
    already_normalized: bool = True
) -> torch.Tensor:
    """
    Prepare images for CNN input.

    Args:
        images: Images array (N, H, W, 3) or (H, W, 3)
        already_normalized: Whether ImageNet normalization was already applied

    Returns:
        Tensor ready for CNN input (N, 3, H, W)
    """
    # Handle single image
    if images.ndim == 3:
        images = np.expand_dims(images, axis=0)

    # Convert from (N, H, W, C) to (N, C, H, W)
    images = np.transpose(images, (0, 3, 1, 2))

    # Convert to tensor
    tensor = torch.from_numpy(images).float()

    # Apply ImageNet normalization if not already done
    if not already_normalized:
        mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

    return tensor


def extract_features(
    images: np.ndarray,
    model_name: str = 'efficientnet_b0',
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    augment: bool = False,
    num_augmentations: int = 5,
    already_normalized: bool = True,
    show_progress: bool = True
) -> np.ndarray:
    """
    Extract features from images using a pre-trained CNN.

    This function handles batching and optional test-time augmentation.

    Args:
        images: Array of preprocessed images (N, H, W, 3)
        model_name: CNN architecture to use
        device: Computation device
        batch_size: Batch size for processing
        augment: Whether to apply augmentation (for training data)
        num_augmentations: Number of augmented versions per image (if augment=True)
        already_normalized: Whether ImageNet normalization was applied
        show_progress: Whether to show progress bar

    Returns:
        features: Array of extracted features (N, num_features) or
                 (N * num_augmentations, num_features) if augment=True
    """
    if device is None:
        device = get_device()

    # Get feature extractor
    model, num_features = get_feature_extractor(model_name, device)

    # Get augmentation transform if needed
    transform = get_augmentation_transform() if augment else None

    all_features = []
    all_indices = []

    # Prepare images (with augmentation if requested)
    if augment and transform is not None:
        print(f"Applying {num_augmentations} augmentations per image...")
        augmented_images = []
        augmented_indices = []

        for idx, img in enumerate(images):
            # Always include original
            augmented_images.append(img)
            augmented_indices.append(idx)

            # Add augmented versions
            for _ in range(num_augmentations - 1):
                aug_img = apply_augmentation(img, transform)
                augmented_images.append(aug_img)
                augmented_indices.append(idx)

        images_to_process = np.array(augmented_images)
        all_indices = augmented_indices
    else:
        images_to_process = images
        all_indices = list(range(len(images)))

    # Process in batches
    num_samples = len(images_to_process)
    num_batches = (num_samples + batch_size - 1) // batch_size

    iterator = range(0, num_samples, batch_size)
    if show_progress:
        iterator = tqdm(iterator, total=num_batches, desc=f"Extracting {model_name} features")

    with torch.no_grad():
        for start_idx in iterator:
            end_idx = min(start_idx + batch_size, num_samples)
            batch_images = images_to_process[start_idx:end_idx]

            # Prepare batch for CNN
            batch_tensor = preprocess_for_cnn(batch_images, already_normalized)
            batch_tensor = batch_tensor.to(device)

            # Extract features
            batch_features = model(batch_tensor)

            # Move to CPU and convert to numpy
            batch_features = batch_features.cpu().numpy()
            all_features.append(batch_features)

    features = np.vstack(all_features)

    print(f"Extracted features shape: {features.shape}")

    return features, all_indices if augment else features


def extract_features_with_augmentation(
    images: np.ndarray,
    labels: np.ndarray,
    model_name: str = 'efficientnet_b0',
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    num_augmentations: int = 5,
    already_normalized: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract features with data augmentation for training data.

    Each image is augmented multiple times, and features are extracted
    from all versions. Labels are replicated accordingly.

    Args:
        images: Training images array (N, H, W, 3)
        labels: Training labels array (N,)
        model_name: CNN architecture
        device: Computation device
        batch_size: Batch size for processing
        num_augmentations: Number of versions per image (including original)
        already_normalized: Whether ImageNet normalization was applied

    Returns:
        features: Augmented features (N * num_augmentations, num_features)
        labels_aug: Replicated labels (N * num_augmentations,)
    """
    features, indices = extract_features(
        images,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        augment=True,
        num_augmentations=num_augmentations,
        already_normalized=already_normalized
    )

    # Replicate labels according to augmentation indices
    labels_aug = labels[indices]

    print(f"Original: {len(labels)} samples")
    print(f"After augmentation: {len(labels_aug)} samples")

    return features, labels_aug


def extract_features_simple(
    images: np.ndarray,
    model_name: str = 'efficientnet_b0',
    device: Optional[torch.device] = None,
    batch_size: int = 32,
    already_normalized: bool = True
) -> np.ndarray:
    """
    Simple feature extraction without augmentation.

    Use this for validation and test sets.

    Args:
        images: Images array (N, H, W, 3)
        model_name: CNN architecture
        device: Computation device
        batch_size: Batch size for processing
        already_normalized: Whether ImageNet normalization was applied

    Returns:
        features: Extracted features (N, num_features)
    """
    return extract_features(
        images,
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        augment=False,
        already_normalized=already_normalized
    )


def compare_feature_extractors(
    images: np.ndarray,
    device: Optional[torch.device] = None,
    batch_size: int = 32
) -> dict:
    """
    Extract features using all available architectures for comparison.

    Args:
        images: Images array (N, H, W, 3)
        device: Computation device
        batch_size: Batch size for processing

    Returns:
        Dictionary mapping model names to their extracted features
    """
    results = {}

    for model_name in FEATURE_DIMS.keys():
        print(f"\n{'='*50}")
        print(f"Extracting features with {model_name}")
        print('='*50)

        features = extract_features_simple(
            images,
            model_name=model_name,
            device=device,
            batch_size=batch_size
        )
        results[model_name] = features

    return results
