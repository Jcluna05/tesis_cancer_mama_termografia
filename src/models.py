"""
Classification Models Module
============================

This module implements the SVM classifiers that work with
features extracted from pre-trained CNNs.

Key Functions:
    - train_svm_classifier: Train SVM with hyperparameter tuning
    - create_pipeline: Create sklearn pipeline with scaling
    - save_model: Save trained model to disk
    - load_model: Load model from disk
"""

import numpy as np
import joblib
from pathlib import Path
from typing import Tuple, Dict, Optional, Any
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import warnings


def create_pipeline(
    kernel: str = 'rbf',
    class_weight: str = 'balanced',
    probability: bool = True
) -> Pipeline:
    """
    Create a sklearn pipeline with StandardScaler and SVM.

    The pipeline ensures that scaling is applied consistently
    during training and inference.

    Args:
        kernel: SVM kernel type ('rbf', 'linear', 'poly')
        class_weight: Weight strategy for imbalanced classes
        probability: Whether to enable probability estimates

    Returns:
        sklearn Pipeline object
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            kernel=kernel,
            class_weight=class_weight,
            probability=probability,
            random_state=42
        ))
    ])

    return pipeline


def train_svm_classifier(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    param_grid: Optional[Dict] = None,
    cv_folds: int = 5,
    scoring: str = 'f1',
    n_jobs: int = -1,
    verbose: int = 1
) -> Tuple[Pipeline, Dict, Dict]:
    """
    Train SVM classifier with hyperparameter tuning.

    Uses GridSearchCV to find optimal hyperparameters based on
    cross-validation performance.

    IMPORTANT: class_weight='balanced' is used by default to handle
    the imbalanced dataset (177 healthy vs 95 sick).

    Args:
        X_train: Training features (N, num_features)
        y_train: Training labels (N,)
        X_val: Validation features (optional)
        y_val: Validation labels (optional)
        param_grid: Hyperparameter grid for search
        cv_folds: Number of cross-validation folds
        scoring: Metric for optimization
        n_jobs: Number of parallel jobs
        verbose: Verbosity level

    Returns:
        best_model: Trained pipeline with best parameters
        best_params: Best hyperparameters found
        cv_results: Cross-validation results summary
    """
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.001, 0.01, 0.1, 1],
            'svm__kernel': ['rbf']  # RBF typically works best for image features
        }

    # Create base pipeline
    pipeline = create_pipeline()

    # Setup cross-validation
    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)

    # Grid search
    print(f"Starting GridSearchCV with {cv_folds}-fold cross-validation...")
    print(f"Optimization metric: {scoring}")
    print(f"Parameter grid: {param_grid}")

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        return_train_score=True
    )

    # Suppress convergence warnings during grid search
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        grid_search.fit(X_train, y_train)

    # Get best model and parameters
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_

    # Compile CV results
    cv_results = {
        'best_score': grid_search.best_score_,
        'best_params': best_params,
        'cv_scores_mean': grid_search.cv_results_['mean_test_score'],
        'cv_scores_std': grid_search.cv_results_['std_test_score'],
    }

    print(f"\nBest parameters: {best_params}")
    print(f"Best CV score ({scoring}): {grid_search.best_score_:.4f}")

    # Evaluate on validation set if provided
    if X_val is not None and y_val is not None:
        val_score = best_model.score(X_val, y_val)
        print(f"Validation accuracy: {val_score:.4f}")
        cv_results['val_accuracy'] = val_score

    return best_model, best_params, cv_results


def train_multiple_kernels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    cv_folds: int = 5,
    scoring: str = 'f1'
) -> Dict[str, Tuple[Pipeline, Dict]]:
    """
    Train SVM classifiers with different kernels for comparison.

    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        cv_folds: Number of CV folds
        scoring: Optimization metric

    Returns:
        Dictionary mapping kernel name to (model, params) tuple
    """
    kernel_configs = {
        'rbf': {
            'svm__C': [0.1, 1, 10, 100],
            'svm__gamma': ['scale', 'auto', 0.01, 0.1],
            'svm__kernel': ['rbf']
        },
        'linear': {
            'svm__C': [0.01, 0.1, 1, 10, 100],
            'svm__kernel': ['linear']
        },
        'poly': {
            'svm__C': [0.1, 1, 10],
            'svm__gamma': ['scale', 'auto'],
            'svm__degree': [2, 3],
            'svm__kernel': ['poly']
        }
    }

    results = {}

    for kernel_name, param_grid in kernel_configs.items():
        print(f"\n{'='*50}")
        print(f"Training SVM with {kernel_name} kernel")
        print('='*50)

        model, params, _ = train_svm_classifier(
            X_train, y_train,
            X_val, y_val,
            param_grid=param_grid,
            cv_folds=cv_folds,
            scoring=scoring
        )

        results[kernel_name] = (model, params)

    return results


def quick_train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 1.0,
    gamma: str = 'scale',
    kernel: str = 'rbf'
) -> Pipeline:
    """
    Quick training without hyperparameter search.

    Useful for rapid prototyping or when optimal parameters
    are already known.

    Args:
        X_train: Training features
        y_train: Training labels
        C: Regularization parameter
        gamma: Kernel coefficient
        kernel: Kernel type

    Returns:
        Trained pipeline
    """
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svm', SVC(
            C=C,
            gamma=gamma,
            kernel=kernel,
            class_weight='balanced',
            probability=True,
            random_state=42
        ))
    ])

    pipeline.fit(X_train, y_train)

    return pipeline


def save_model(
    model: Pipeline,
    filepath: str,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save trained model to disk.

    Args:
        model: Trained pipeline
        filepath: Path to save the model
        metadata: Optional metadata to save alongside
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    save_data = {
        'model': model,
        'metadata': metadata or {}
    }

    joblib.dump(save_data, filepath)
    print(f"Model saved to: {filepath}")


def load_model(filepath: str) -> Tuple[Pipeline, Dict]:
    """
    Load trained model from disk.

    Args:
        filepath: Path to the saved model

    Returns:
        model: Loaded pipeline
        metadata: Associated metadata
    """
    filepath = Path(filepath)

    if not filepath.exists():
        raise FileNotFoundError(f"Model file not found: {filepath}")

    save_data = joblib.load(filepath)

    model = save_data['model']
    metadata = save_data.get('metadata', {})

    print(f"Model loaded from: {filepath}")

    return model, metadata


def get_model_info(model: Pipeline) -> Dict:
    """
    Extract information about a trained model.

    Args:
        model: Trained pipeline

    Returns:
        Dictionary with model information
    """
    svm = model.named_steps['svm']
    scaler = model.named_steps['scaler']

    info = {
        'kernel': svm.kernel,
        'C': svm.C,
        'gamma': svm.gamma if hasattr(svm, 'gamma') else None,
        'class_weight': svm.class_weight,
        'n_support_vectors': svm.n_support_.tolist() if hasattr(svm, 'n_support_') else None,
        'total_support_vectors': sum(svm.n_support_) if hasattr(svm, 'n_support_') else None,
        'scaler_mean_shape': scaler.mean_.shape if hasattr(scaler, 'mean_') else None,
    }

    return info
