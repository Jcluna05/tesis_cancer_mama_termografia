"""
Preprocessing Module for Thermographic Breast Images
=====================================================

This module handles image preprocessing specifically designed for
thermographic images, including CLAHE normalization to handle
different color palettes (Ironbow, Rainbow, grayscale).

Key Functions:
    - preprocess_image: Full preprocessing pipeline for single image
    - load_dataset: Load and preprocess entire dataset
    - apply_clahe: Apply CLAHE normalization
    - get_data_splits: Stratified train/val/test split
"""

import os
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Union
from sklearn.model_selection import train_test_split
import albumentations as A
from tqdm import tqdm


# ImageNet normalization constants (required for transfer learning)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD = np.array([0.229, 0.224, 0.225])


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Apply CLAHE (Contrast Limited Adaptive Histogram Equalization).

    CLAHE improves local contrast without saturating the image,
    which is especially useful for thermographic images with
    different color palettes.

    Args:
        image: Input image (BGR or grayscale)
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization

    Returns:
        CLAHE-enhanced image
    """
    # Convert to LAB color space if color image
    if len(image.shape) == 3 and image.shape[2] == 3:
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l_channel, a_channel, b_channel = cv2.split(lab)

        # Apply CLAHE to L channel
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        l_clahe = clahe.apply(l_channel)

        # Merge channels back
        lab_clahe = cv2.merge([l_clahe, a_channel, b_channel])
        result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    else:
        # Grayscale image
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        result = clahe.apply(image)

    return result


def preprocess_image(
    image_path: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe_norm: bool = True,
    normalize_imagenet: bool = True
) -> np.ndarray:
    """
    Full preprocessing pipeline for thermographic images.

    Pipeline steps:
    1. Load image
    2. Apply CLAHE for contrast normalization
    3. Resize to target size
    4. Convert to RGB (3 channels)
    5. Normalize to [0, 1] range
    6. Optionally apply ImageNet normalization

    Args:
        image_path: Path to the image file
        target_size: Output size (height, width)
        apply_clahe_norm: Whether to apply CLAHE normalization
        normalize_imagenet: Whether to apply ImageNet mean/std normalization

    Returns:
        Preprocessed image as numpy array (H, W, 3)
    """
    # Load image using OpenCV (BGR format)
    image = cv2.imread(str(image_path))

    if image is None:
        raise ValueError(f"Could not load image: {image_path}")

    # Apply CLAHE normalization
    if apply_clahe_norm:
        image = apply_clahe(image)

    # Resize to target size
    image = cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)

    # Convert BGR to RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Normalize to [0, 1]
    image = image.astype(np.float32) / 255.0

    # Apply ImageNet normalization if requested
    if normalize_imagenet:
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image


def load_dataset(
    data_dir: Union[str, Path],
    target_size: Tuple[int, int] = (224, 224),
    apply_clahe_norm: bool = True,
    normalize_imagenet: bool = True
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Load and preprocess the entire dataset.

    Expected directory structure:
    data_dir/
    ├── healthy/
    │   ├── image1.png
    │   └── ...
    └── sick/
        ├── image1.png
        └── ...

    Args:
        data_dir: Root directory containing 'healthy' and 'sick' folders
        target_size: Output image size
        apply_clahe_norm: Whether to apply CLAHE
        normalize_imagenet: Whether to apply ImageNet normalization

    Returns:
        images: Array of preprocessed images (N, H, W, 3)
        labels: Array of labels (0=healthy, 1=sick)
        paths: List of original image paths
    """
    data_dir = Path(data_dir)

    images = []
    labels = []
    paths = []

    # Class mapping
    class_map = {'healthy': 0, 'sick': 1}

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} not found")
            continue

        # Get all image files
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))
            image_files.extend(list(class_dir.glob(ext.upper())))

        print(f"Loading {len(image_files)} images from '{class_name}'...")

        for img_path in tqdm(image_files, desc=f"Processing {class_name}"):
            try:
                img = preprocess_image(
                    img_path,
                    target_size=target_size,
                    apply_clahe_norm=apply_clahe_norm,
                    normalize_imagenet=normalize_imagenet
                )
                images.append(img)
                labels.append(label)
                paths.append(str(img_path))
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                continue

    return np.array(images), np.array(labels), paths


def get_data_splits(
    X: np.ndarray,
    y: np.ndarray,
    paths: Optional[List[str]] = None,
    test_size: float = 0.30,
    val_ratio: float = 0.50,
    random_state: int = 42
) -> Dict:
    """
    Perform stratified train/validation/test split.

    CRITICAL: This split must be done BEFORE any data augmentation
    to avoid data leakage.

    Default split: 70% train, 15% validation, 15% test

    Args:
        X: Feature array (images or extracted features)
        y: Labels array
        paths: Optional list of image paths
        test_size: Proportion for val+test combined (default 0.30)
        val_ratio: Proportion of test_size for validation (default 0.50)
        random_state: Random seed for reproducibility

    Returns:
        Dictionary containing all splits
    """
    # First split: train vs (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # Second split: val vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=random_state,
        stratify=y_temp
    )

    splits = {
        'X_train': X_train,
        'X_val': X_val,
        'X_test': X_test,
        'y_train': y_train,
        'y_val': y_val,
        'y_test': y_test,
    }

    # Handle paths if provided
    if paths is not None:
        paths = np.array(paths)
        idx_train, idx_temp = train_test_split(
            np.arange(len(y)),
            test_size=test_size,
            random_state=random_state,
            stratify=y
        )
        idx_val, idx_test = train_test_split(
            idx_temp,
            test_size=val_ratio,
            random_state=random_state,
            stratify=y[idx_temp]
        )
        splits['paths_train'] = paths[idx_train].tolist()
        splits['paths_val'] = paths[idx_val].tolist()
        splits['paths_test'] = paths[idx_test].tolist()

    # Print split statistics
    print("\nDataset Split Statistics:")
    print(f"  Training:   {len(y_train):4d} samples "
          f"(healthy: {sum(y_train==0)}, sick: {sum(y_train==1)})")
    print(f"  Validation: {len(y_val):4d} samples "
          f"(healthy: {sum(y_val==0)}, sick: {sum(y_val==1)})")
    print(f"  Test:       {len(y_test):4d} samples "
          f"(healthy: {sum(y_test==0)}, sick: {sum(y_test==1)})")

    return splits


def get_augmentation_transform(augment: bool = True) -> Optional[A.Compose]:
    """
    Get augmentation transform for training data.

    IMPORTANT: Only applied to training data, NEVER to validation/test.
    Uses conservative augmentation suitable for medical images.

    Args:
        augment: Whether to return augmentation transform

    Returns:
        Albumentations Compose transform or None
    """
    if not augment:
        return None

    return A.Compose([
        A.HorizontalFlip(p=0.5),
        A.Rotate(limit=15, p=0.5, border_mode=cv2.BORDER_REFLECT_101),
        A.RandomBrightnessContrast(
            brightness_limit=0.1,
            contrast_limit=0.1,
            p=0.3
        ),
        A.GaussNoise(var_limit=(10.0, 30.0), p=0.2),
    ])


def apply_augmentation(
    image: np.ndarray,
    transform: Optional[A.Compose] = None
) -> np.ndarray:
    """
    Apply augmentation transform to a single image.

    Args:
        image: Input image (H, W, 3), can be normalized
        transform: Albumentations transform

    Returns:
        Augmented image
    """
    if transform is None:
        return image

    # If image is ImageNet normalized, denormalize first
    was_normalized = image.min() < 0  # ImageNet normalized images have negative values

    if was_normalized:
        image = image * IMAGENET_STD + IMAGENET_MEAN
        image = np.clip(image, 0, 1)

    # Convert to uint8 for albumentations
    image_uint8 = (image * 255).astype(np.uint8)

    # Apply augmentation
    augmented = transform(image=image_uint8)['image']

    # Convert back to float32 [0, 1]
    image = augmented.astype(np.float32) / 255.0

    # Re-apply ImageNet normalization if it was normalized before
    if was_normalized:
        image = (image - IMAGENET_MEAN) / IMAGENET_STD

    return image


def get_dataset_info(data_dir: Union[str, Path]) -> Dict:
    """
    Get information about the dataset without loading images.

    Args:
        data_dir: Root directory of the dataset

    Returns:
        Dictionary with dataset statistics
    """
    data_dir = Path(data_dir)

    info = {
        'healthy_count': 0,
        'sick_count': 0,
        'total_count': 0,
        'healthy_files': [],
        'sick_files': [],
    }

    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

    for class_name in ['healthy', 'sick']:
        class_dir = data_dir / class_name
        if class_dir.exists():
            files = []
            for ext in image_extensions:
                files.extend(list(class_dir.glob(ext)))
                files.extend(list(class_dir.glob(ext.upper())))

            info[f'{class_name}_count'] = len(files)
            info[f'{class_name}_files'] = [str(f) for f in files]

    info['total_count'] = info['healthy_count'] + info['sick_count']

    return info
