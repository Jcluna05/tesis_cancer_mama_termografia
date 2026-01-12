#!/usr/bin/env python3
"""
Generate Test Images List
==========================

This script generates a list of the 41 test images used in the experiments.
It recreates the exact train/val/test split using the same random_state=42
that was used in the original notebook.

Usage:
    python scripts/generate_test_images_list.py

Output:
    results/test_images_list.txt - List of test image names with their labels

Requirements:
    - Dataset DMR-IR must be available in data/raw/DMR-IR/
    - Or set DATASET_PATH environment variable to the dataset location
"""

import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from sklearn.model_selection import train_test_split

# Configuration (same as in the notebook)
RANDOM_STATE = 42
TEST_SIZE = 0.30
VAL_RATIO = 0.50


def get_image_paths(data_dir: Path) -> tuple:
    """
    Get all image paths and labels from the dataset.

    Returns:
        paths: List of image paths
        labels: Array of labels (0=healthy, 1=sick)
    """
    paths = []
    labels = []

    class_map = {'healthy': 0, 'sick': 1}
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp', '*.tif', '*.tiff']

    for class_name, label in class_map.items():
        class_dir = data_dir / class_name

        if not class_dir.exists():
            print(f"Warning: Directory {class_dir} not found")
            continue

        # Get all image files
        image_files = []
        for ext in image_extensions:
            image_files.extend(list(class_dir.glob(ext)))
            image_files.extend(list(class_dir.glob(ext.upper())))

        # Sort for reproducibility
        image_files = sorted(image_files)

        for img_path in image_files:
            paths.append(str(img_path))
            labels.append(label)

    return paths, np.array(labels)


def get_test_split_indices(n_samples: int, labels: np.ndarray) -> np.ndarray:
    """
    Recreate the exact test split indices using the same parameters as the notebook.
    """
    # First split: train vs (val + test)
    idx_train, idx_temp = train_test_split(
        np.arange(n_samples),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels
    )

    # Second split: val vs test
    idx_val, idx_test = train_test_split(
        idx_temp,
        test_size=VAL_RATIO,
        random_state=RANDOM_STATE,
        stratify=labels[idx_temp]
    )

    return idx_test, labels[idx_test]


def main():
    # Determine dataset path
    default_path = PROJECT_ROOT / 'data' / 'raw' / 'DMR-IR'
    dataset_path = Path(os.environ.get('DATASET_PATH', default_path))

    print("=" * 60)
    print("GENERATING TEST IMAGES LIST")
    print("=" * 60)
    print(f"\nDataset path: {dataset_path}")

    if not dataset_path.exists():
        print(f"\nERROR: Dataset not found at {dataset_path}")
        print("\nTo use this script, either:")
        print("  1. Copy the DMR-IR dataset to data/raw/DMR-IR/")
        print("  2. Set DATASET_PATH environment variable:")
        print("     export DATASET_PATH=/path/to/DMR-IR")
        sys.exit(1)

    # Get all image paths and labels
    print("\nLoading image paths...")
    paths, labels = get_image_paths(dataset_path)

    print(f"Total images found: {len(paths)}")
    print(f"  - Healthy: {sum(labels == 0)}")
    print(f"  - Sick: {sum(labels == 1)}")

    if len(paths) != 272:
        print(f"\nWARNING: Expected 272 images, found {len(paths)}")
        print("The test split may not match the original experiment.")

    # Get test split indices
    print("\nRecreating test split with random_state=42...")
    test_indices, test_labels = get_test_split_indices(len(paths), labels)

    print(f"Test set size: {len(test_indices)}")
    print(f"  - Healthy: {sum(test_labels == 0)}")
    print(f"  - Sick: {sum(test_labels == 1)}")

    # Prepare output
    output_dir = PROJECT_ROOT / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / 'test_images_list.txt'

    # Generate the list
    print(f"\nWriting to: {output_file}")

    with open(output_file, 'w') as f:
        f.write("# Test Images List - DMR-IR Dataset\n")
        f.write("# Generated with random_state=42 (same as experimentos.ipynb)\n")
        f.write(f"# Total: {len(test_indices)} images\n")
        f.write("#\n")
        f.write("# Format: label,image_name\n")
        f.write("# Labels: healthy=0, sick=1\n")
        f.write("#" + "=" * 59 + "\n\n")

        # Group by class for better readability
        healthy_images = []
        sick_images = []

        for idx in test_indices:
            img_path = Path(paths[idx])
            img_name = img_path.name
            label = labels[idx]

            if label == 0:
                healthy_images.append(img_name)
            else:
                sick_images.append(img_name)

        # Write healthy images
        f.write(f"# HEALTHY ({len(healthy_images)} images)\n")
        for img_name in sorted(healthy_images):
            f.write(f"healthy,{img_name}\n")

        f.write(f"\n# SICK ({len(sick_images)} images)\n")
        for img_name in sorted(sick_images):
            f.write(f"sick,{img_name}\n")

    print("\nDone!")
    print(f"\nTest images list saved to: {output_file}")

    # Also print summary to console
    print("\n" + "=" * 60)
    print("TEST SET SUMMARY")
    print("=" * 60)
    print(f"\nHealthy ({len(healthy_images)} images):")
    for img in sorted(healthy_images)[:5]:
        print(f"  - {img}")
    if len(healthy_images) > 5:
        print(f"  ... and {len(healthy_images) - 5} more")

    print(f"\nSick ({len(sick_images)} images):")
    for img in sorted(sick_images)[:5]:
        print(f"  - {img}")
    if len(sick_images) > 5:
        print(f"  ... and {len(sick_images) - 5} more")


if __name__ == '__main__':
    main()
