"""
Evaluation Module for Model Assessment
======================================

This module provides comprehensive evaluation metrics for
binary classification models, specifically tailored for
medical diagnosis applications.

Key Metrics:
    - Accuracy
    - Precision
    - Recall (Sensitivity)
    - Specificity
    - F1-Score
    - AUC-ROC

Key Functions:
    - evaluate_model: Complete model evaluation
    - calculate_metrics: Compute all metrics
    - get_roc_data: Get ROC curve data
    - generate_classification_report: Full text report
"""

import numpy as np
from typing import Dict, Tuple, Any, Optional
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve,
    average_precision_score
)
import json
from pathlib import Path


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_pred_proba: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Calculate all classification metrics.

    For medical diagnosis, Recall (Sensitivity) and Specificity
    are particularly important:
    - Sensitivity: Ability to correctly identify sick patients
    - Specificity: Ability to correctly identify healthy patients

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities for positive class

    Returns:
        Dictionary of all metrics
    """
    # Basic metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),  # Sensitivity
        'f1_score': f1_score(y_true, y_pred, zero_division=0),
    }

    # Confusion matrix for specificity calculation
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Specificity = TN / (TN + FP)
    metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # Additional confusion matrix values
    metrics['true_negatives'] = int(tn)
    metrics['false_positives'] = int(fp)
    metrics['false_negatives'] = int(fn)
    metrics['true_positives'] = int(tp)

    # AUC-ROC if probabilities are available
    if y_pred_proba is not None:
        metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_pred_proba)

    return metrics


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = "Model"
) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
    """
    Complete model evaluation on test set.

    Args:
        model: Trained classifier with predict and predict_proba methods
        X_test: Test features
        y_test: Test labels
        model_name: Name for display purposes

    Returns:
        metrics: Dictionary of all metrics
        y_pred: Predicted labels
        y_pred_proba: Predicted probabilities
    """
    # Get predictions
    y_pred = model.predict(X_test)

    # Get probabilities if available
    y_pred_proba = None
    if hasattr(model, 'predict_proba'):
        y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Calculate metrics
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)

    # Print results
    print(f"\n{'='*50}")
    print(f"Evaluation Results: {model_name}")
    print('='*50)
    print(f"  Accuracy:    {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)")
    print(f"  Precision:   {metrics['precision']:.4f} ({metrics['precision']*100:.2f}%)")
    print(f"  Recall:      {metrics['recall']:.4f} ({metrics['recall']*100:.2f}%)")
    print(f"  Specificity: {metrics['specificity']:.4f} ({metrics['specificity']*100:.2f}%)")
    print(f"  F1-Score:    {metrics['f1_score']:.4f} ({metrics['f1_score']*100:.2f}%)")
    if 'auc_roc' in metrics:
        print(f"  AUC-ROC:     {metrics['auc_roc']:.4f}")
    print()
    print(f"Confusion Matrix:")
    print(f"  TN={metrics['true_negatives']:3d}  FP={metrics['false_positives']:3d}")
    print(f"  FN={metrics['false_negatives']:3d}  TP={metrics['true_positives']:3d}")

    return metrics, y_pred, y_pred_proba


def get_roc_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get ROC curve data for plotting.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        fpr: False positive rates
        tpr: True positive rates (Sensitivity)
        thresholds: Decision thresholds
        auc: Area under the ROC curve
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc = roc_auc_score(y_true, y_pred_proba)

    return fpr, tpr, thresholds, auc


def get_precision_recall_data(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Get Precision-Recall curve data for plotting.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities

    Returns:
        precision: Precision values
        recall: Recall values
        thresholds: Decision thresholds
        ap: Average precision score
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    ap = average_precision_score(y_true, y_pred_proba)

    return precision, recall, thresholds, ap


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list = None
) -> str:
    """
    Generate a detailed classification report.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        target_names: Names for each class

    Returns:
        Formatted classification report string
    """
    if target_names is None:
        target_names = ['Healthy', 'Sick']

    return classification_report(
        y_true, y_pred,
        target_names=target_names,
        digits=4
    )


def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics_to_compare: list = None
) -> str:
    """
    Generate a comparison table for multiple models.

    Args:
        results: Dictionary mapping model names to their metrics
        metrics_to_compare: List of metric names to include

    Returns:
        Formatted comparison table as string
    """
    if metrics_to_compare is None:
        metrics_to_compare = [
            'accuracy', 'precision', 'recall',
            'specificity', 'f1_score', 'auc_roc'
        ]

    # Build table
    header = "| Metric        |"
    separator = "|---------------|"

    for model_name in results.keys():
        header += f" {model_name:^20} |"
        separator += "----------------------|"

    rows = [header, separator]

    metric_display_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall (Sens.)',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC'
    }

    for metric in metrics_to_compare:
        display_name = metric_display_names.get(metric, metric)
        row = f"| {display_name:<13} |"

        for model_name, metrics in results.items():
            value = metrics.get(metric, 0)
            if metric == 'auc_roc':
                row += f" {value:^20.4f} |"
            else:
                row += f" {value*100:^19.2f}% |"

        rows.append(row)

    return '\n'.join(rows)


def save_metrics(
    metrics: Dict[str, Any],
    filepath: str,
    model_name: str = None
) -> None:
    """
    Save metrics to JSON file.

    Args:
        metrics: Dictionary of metrics
        filepath: Path to save file
        model_name: Optional model name for grouping
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types
    clean_metrics = {}
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            clean_metrics[key] = float(value)
        elif isinstance(value, np.ndarray):
            clean_metrics[key] = value.tolist()
        else:
            clean_metrics[key] = value

    save_data = clean_metrics
    if model_name:
        save_data = {model_name: clean_metrics}

    # Load existing data if file exists
    if filepath.exists():
        with open(filepath, 'r') as f:
            existing_data = json.load(f)
        if model_name:
            existing_data[model_name] = clean_metrics
            save_data = existing_data

    with open(filepath, 'w') as f:
        json.dump(save_data, f, indent=2)

    print(f"Metrics saved to: {filepath}")


def load_metrics(filepath: str) -> Dict:
    """
    Load metrics from JSON file.

    Args:
        filepath: Path to metrics file

    Returns:
        Dictionary of metrics
    """
    with open(filepath, 'r') as f:
        return json.load(f)


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find optimal classification threshold.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'youden', 'sensitivity')

    Returns:
        optimal_threshold: Best threshold value
        best_score: Score at optimal threshold
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    optimal_threshold = 0.5

    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)

        if metric == 'f1':
            score = f1_score(y_true, y_pred)
        elif metric == 'youden':
            # Youden's J statistic = Sensitivity + Specificity - 1
            tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        elif metric == 'sensitivity':
            score = recall_score(y_true, y_pred)
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if score > best_score:
            best_score = score
            optimal_threshold = threshold

    return optimal_threshold, best_score


def bootstrap_confidence_interval(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric_func: callable,
    n_iterations: int = 1000,
    confidence_level: float = 0.95,
    random_state: int = 42
) -> Tuple[float, float, float]:
    """
    Calculate bootstrap confidence interval for a metric.

    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric_func: Function that takes (y_true, y_pred_proba) and returns metric
        n_iterations: Number of bootstrap iterations
        confidence_level: Confidence level (default 0.95 for 95% CI)
        random_state: Random seed

    Returns:
        mean: Mean metric value
        lower: Lower bound of confidence interval
        upper: Upper bound of confidence interval
    """
    np.random.seed(random_state)
    n_samples = len(y_true)
    scores = []

    for _ in range(n_iterations):
        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_proba_boot = y_pred_proba[indices]

        try:
            score = metric_func(y_true_boot, y_proba_boot)
            scores.append(score)
        except:
            continue

    scores = np.array(scores)

    alpha = 1 - confidence_level
    lower = np.percentile(scores, alpha/2 * 100)
    upper = np.percentile(scores, (1 - alpha/2) * 100)
    mean = np.mean(scores)

    return mean, lower, upper
