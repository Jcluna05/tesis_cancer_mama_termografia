"""
Breast Cancer Detection using Thermographic Images
====================================================

This package implements a classification pipeline for breast thermography
using pre-trained CNNs (EfficientNet-B0, ResNet50) as feature extractors
combined with SVM classifiers.

Modules:
    - preprocessing: Image preprocessing with CLAHE normalization
    - feature_extraction: CNN-based feature extraction
    - models: SVM classifier training and hyperparameter tuning
    - evaluation: Model evaluation metrics and analysis
    - visualization: Publication-quality figures generation

Author: Thesis Project - Master's in Applied Artificial Intelligence
"""

from . import preprocessing
from . import feature_extraction
from . import models
from . import evaluation
from . import visualization

__version__ = "1.0.0"
__author__ = "Thesis Project"
