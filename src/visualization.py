"""
Visualization Module for Thesis Figures
========================================

This module generates publication-quality figures for the thesis
document, including confusion matrices, ROC curves, metric
comparisons, and feature visualizations.

All figures are saved in both PNG (300 DPI) and PDF formats
for compatibility with different document systems.

Key Functions:
    - plot_confusion_matrix: Confusion matrix heatmap
    - plot_roc_curves: ROC curves comparison
    - plot_metrics_comparison: Bar chart of metrics
    - plot_feature_distribution: t-SNE/PCA visualization
    - plot_preprocessing_examples: Before/after preprocessing
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings


# Set style for publication-quality figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.titlesize': 18,
    'figure.dpi': 100,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Color palette
COLORS = {
    'efficientnet': '#2ecc71',  # Green
    'resnet': '#3498db',        # Blue
    'healthy': '#27ae60',       # Dark green
    'sick': '#e74c3c',          # Red
    'reference': '#7f8c8d',     # Gray
}


def save_figure(
    fig: plt.Figure,
    filename: str,
    output_dir: str = 'results/figures',
    formats: List[str] = ['png', 'pdf']
) -> None:
    """
    Save figure in multiple formats.

    Args:
        fig: Matplotlib figure object
        filename: Base filename (without extension)
        output_dir: Output directory
        formats: List of formats to save
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, format=fmt, dpi=300, bbox_inches='tight')
        print(f"Saved: {filepath}")


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
    class_names: List[str] = ['Healthy', 'Sick'],
    normalize: bool = True,
    figsize: Tuple[int, int] = (8, 6),
    cmap: str = 'Blues',
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Plot confusion matrix as heatmap.

    Shows both absolute values and percentages for clarity.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        model_name: Name for the title
        class_names: Names of classes
        normalize: Whether to show percentages
        figsize: Figure size
        cmap: Colormap
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=figsize)

    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        # Create annotation strings with both count and percentage
        annotations = np.empty_like(cm, dtype=object)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                annotations[i, j] = f'{cm[i, j]}\n({cm_normalized[i, j]*100:.1f}%)'

        sns.heatmap(
            cm_normalized, annot=annotations, fmt='',
            cmap=cmap, ax=ax, vmin=0, vmax=1,
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Proportion'}
        )
    else:
        sns.heatmap(
            cm, annot=True, fmt='d',
            cmap=cmap, ax=ax,
            xticklabels=class_names, yticklabels=class_names
        )

    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title(f'Confusion Matrix - {model_name}')

    plt.tight_layout()

    # Save figure
    filename = f"confusion_matrix_{model_name.lower().replace(' ', '_').replace('+', '_')}"
    save_figure(fig, filename, output_dir)

    return fig


def plot_roc_curves(
    roc_data: Dict[str, Tuple[np.ndarray, np.ndarray, float]],
    figsize: Tuple[int, int] = (8, 8),
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.

    Args:
        roc_data: Dictionary mapping model names to (fpr, tpr, auc) tuples
        figsize: Figure size
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=figsize)

    colors = list(COLORS.values())

    for idx, (model_name, (fpr, tpr, auc)) in enumerate(roc_data.items()):
        color = colors[idx % len(colors)]
        ax.plot(
            fpr, tpr,
            color=color,
            linewidth=2,
            label=f'{model_name} (AUC = {auc:.4f})'
        )

    # Reference diagonal line
    ax.plot(
        [0, 1], [0, 1],
        color=COLORS['reference'],
        linestyle='--',
        linewidth=1.5,
        label='Random Classifier'
    )

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title('ROC Curves Comparison')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)

    # Add perfect classifier point
    ax.plot(0, 1, 'g*', markersize=15, label='Perfect Classifier')

    plt.tight_layout()

    save_figure(fig, 'roc_curves_comparison', output_dir)

    return fig


def plot_metrics_comparison(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_plot: List[str] = None,
    figsize: Tuple[int, int] = (12, 6),
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Plot grouped bar chart comparing metrics across models.

    Args:
        metrics_dict: Dictionary mapping model names to metric dictionaries
        metrics_to_plot: List of metrics to include
        figsize: Figure size
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']

    metric_labels = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall\n(Sensitivity)',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC'
    }

    model_names = list(metrics_dict.keys())
    n_models = len(model_names)
    n_metrics = len(metrics_to_plot)

    x = np.arange(n_metrics)
    width = 0.35 if n_models == 2 else 0.25

    fig, ax = plt.subplots(figsize=figsize)

    colors = [COLORS['efficientnet'], COLORS['resnet'], '#9b59b6', '#f39c12']

    for idx, model_name in enumerate(model_names):
        offset = (idx - (n_models - 1) / 2) * width
        values = [metrics_dict[model_name].get(m, 0) * 100 for m in metrics_to_plot]

        bars = ax.bar(
            x + offset, values, width,
            label=model_name,
            color=colors[idx % len(colors)],
            edgecolor='black',
            linewidth=0.5
        )

        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.annotate(
                f'{value:.1f}%',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha='center', va='bottom',
                fontsize=9
            )

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score (%)')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels.get(m, m) for m in metrics_to_plot])
    ax.legend(loc='upper right')
    ax.set_ylim(0, 110)
    ax.grid(True, axis='y', alpha=0.3)

    plt.tight_layout()

    save_figure(fig, 'metrics_comparison', output_dir)

    return fig


def plot_feature_distribution(
    features: np.ndarray,
    labels: np.ndarray,
    method: str = 'tsne',
    model_name: str = "Features",
    figsize: Tuple[int, int] = (10, 8),
    random_state: int = 42,
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Visualize feature distribution using dimensionality reduction.

    Args:
        features: Feature array (N, num_features)
        labels: Class labels (N,)
        method: 'tsne' or 'pca'
        model_name: Name for the title
        figsize: Figure size
        random_state: Random seed
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    print(f"Computing {method.upper()} projection...")

    if method == 'tsne':
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            reducer = TSNE(
                n_components=2,
                random_state=random_state,
                perplexity=min(30, len(features) - 1),
                max_iter=1000
            )
            projected = reducer.fit_transform(features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=random_state)
        projected = reducer.fit_transform(features)
    else:
        raise ValueError(f"Unknown method: {method}")

    fig, ax = plt.subplots(figsize=figsize)

    # Plot each class
    class_names = ['Healthy', 'Sick']
    colors = [COLORS['healthy'], COLORS['sick']]
    markers = ['o', 's']

    for class_idx in [0, 1]:
        mask = labels == class_idx
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            c=colors[class_idx],
            marker=markers[class_idx],
            s=60,
            alpha=0.7,
            edgecolors='white',
            linewidth=0.5,
            label=class_names[class_idx]
        )

    ax.set_xlabel(f'{method.upper()} Component 1')
    ax.set_ylabel(f'{method.upper()} Component 2')
    ax.set_title(f'Feature Distribution - {model_name}\n({method.upper()} Projection)')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    filename = f"feature_distribution_{model_name.lower().replace(' ', '_').replace('+', '_')}_{method}"
    save_figure(fig, filename, output_dir)

    return fig


def plot_preprocessing_examples(
    original_images: List[np.ndarray],
    processed_images: List[np.ndarray],
    labels: List[int],
    n_examples: int = 4,
    figsize: Tuple[int, int] = (12, 8),
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Show before/after preprocessing examples.

    Args:
        original_images: List of original images
        processed_images: List of preprocessed images
        labels: Class labels for each image
        n_examples: Number of examples to show
        figsize: Figure size
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    n_examples = min(n_examples, len(original_images))

    fig, axes = plt.subplots(2, n_examples, figsize=figsize)

    class_names = ['Healthy', 'Sick']

    for i in range(n_examples):
        # Original image
        ax_orig = axes[0, i]
        if original_images[i].dtype == np.float32:
            img_orig = np.clip(original_images[i], 0, 1)
        else:
            img_orig = original_images[i]
        ax_orig.imshow(img_orig)
        ax_orig.set_title(f'{class_names[labels[i]]}\n(Original)')
        ax_orig.axis('off')

        # Processed image
        ax_proc = axes[1, i]
        img_proc = processed_images[i]
        # Denormalize if ImageNet normalized
        if img_proc.min() < 0:
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img_proc = img_proc * std + mean
        img_proc = np.clip(img_proc, 0, 1)
        ax_proc.imshow(img_proc)
        ax_proc.set_title('Preprocessed\n(CLAHE + Normalized)')
        ax_proc.axis('off')

    plt.suptitle('Image Preprocessing Examples', fontsize=16, y=1.02)
    plt.tight_layout()

    save_figure(fig, 'preprocessing_examples', output_dir)

    return fig


def plot_class_distribution(
    labels: np.ndarray,
    split_name: str = "Dataset",
    figsize: Tuple[int, int] = (8, 6),
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Plot class distribution as bar chart.

    Args:
        labels: Class labels array
        split_name: Name for the title
        figsize: Figure size
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    class_names = ['Healthy', 'Sick']
    counts = [np.sum(labels == 0), np.sum(labels == 1)]

    fig, ax = plt.subplots(figsize=figsize)

    bars = ax.bar(
        class_names, counts,
        color=[COLORS['healthy'], COLORS['sick']],
        edgecolor='black',
        linewidth=1
    )

    # Add value labels
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        ax.annotate(
            f'{count}\n({count/sum(counts)*100:.1f}%)',
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha='center', va='bottom',
            fontsize=12
        )

    ax.set_xlabel('Class')
    ax.set_ylabel('Number of Samples')
    ax.set_title(f'Class Distribution - {split_name}')
    ax.set_ylim(0, max(counts) * 1.2)

    plt.tight_layout()

    filename = f"class_distribution_{split_name.lower().replace(' ', '_')}"
    save_figure(fig, filename, output_dir)

    return fig


def plot_training_results_summary(
    results: Dict[str, Dict],
    output_dir: str = 'results/figures'
) -> plt.Figure:
    """
    Create a summary figure with multiple subplots.

    Args:
        results: Dictionary with all model results
        output_dir: Directory to save figure

    Returns:
        Matplotlib figure
    """
    fig = plt.figure(figsize=(16, 12))

    # Create grid
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)

    model_names = list(results.keys())

    # Subplot 1: Metrics comparison
    ax1 = fig.add_subplot(gs[0, 0])
    metrics = ['accuracy', 'precision', 'recall', 'specificity', 'f1_score']
    x = np.arange(len(metrics))
    width = 0.35

    for idx, model_name in enumerate(model_names):
        offset = (idx - 0.5) * width
        values = [results[model_name]['metrics'].get(m, 0) * 100 for m in metrics]
        ax1.bar(x + offset, values, width, label=model_name)

    ax1.set_xticks(x)
    ax1.set_xticklabels(['Acc', 'Prec', 'Recall', 'Spec', 'F1'])
    ax1.set_ylabel('Score (%)')
    ax1.set_title('Performance Metrics')
    ax1.legend()
    ax1.set_ylim(0, 100)

    # Subplot 2: ROC curves
    ax2 = fig.add_subplot(gs[0, 1])
    colors = [COLORS['efficientnet'], COLORS['resnet']]

    for idx, model_name in enumerate(model_names):
        if 'roc_data' in results[model_name]:
            fpr, tpr, auc = results[model_name]['roc_data']
            ax2.plot(fpr, tpr, color=colors[idx],
                    label=f'{model_name} (AUC={auc:.3f})')

    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves')
    ax2.legend(loc='lower right')

    # Subplot 3 & 4: Confusion matrices
    for idx, model_name in enumerate(model_names):
        ax = fig.add_subplot(gs[1, idx])
        if 'confusion_matrix' in results[model_name]:
            cm = results[model_name]['confusion_matrix']
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Healthy', 'Sick'],
                       yticklabels=['Healthy', 'Sick'])
            ax.set_title(f'Confusion Matrix\n{model_name}')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')

    plt.suptitle('Model Evaluation Summary', fontsize=16, y=1.02)

    save_figure(fig, 'training_summary', output_dir)

    return fig


def create_latex_table(
    metrics_dict: Dict[str, Dict[str, float]],
    metrics_to_include: List[str] = None,
    caption: str = "Model Comparison Results",
    label: str = "tab:results"
) -> str:
    """
    Generate LaTeX table code for the thesis.

    Args:
        metrics_dict: Dictionary mapping model names to metrics
        metrics_to_include: List of metrics to include
        caption: Table caption
        label: LaTeX label

    Returns:
        LaTeX table code as string
    """
    if metrics_to_include is None:
        metrics_to_include = [
            'accuracy', 'precision', 'recall',
            'specificity', 'f1_score', 'auc_roc'
        ]

    metric_names = {
        'accuracy': 'Accuracy',
        'precision': 'Precision',
        'recall': 'Recall (Sensitivity)',
        'specificity': 'Specificity',
        'f1_score': 'F1-Score',
        'auc_roc': 'AUC-ROC'
    }

    model_names = list(metrics_dict.keys())
    n_models = len(model_names)

    # Build LaTeX code
    latex = []
    latex.append("\\begin{table}[htbp]")
    latex.append("\\centering")
    latex.append(f"\\caption{{{caption}}}")
    latex.append(f"\\label{{{label}}}")
    latex.append("\\begin{tabular}{l" + "c" * n_models + "}")
    latex.append("\\toprule")

    # Header row
    header = "\\textbf{Metric}"
    for model in model_names:
        header += f" & \\textbf{{{model}}}"
    header += " \\\\"
    latex.append(header)
    latex.append("\\midrule")

    # Data rows
    for metric in metrics_to_include:
        row = metric_names.get(metric, metric)
        for model in model_names:
            value = metrics_dict[model].get(metric, 0)
            if metric == 'auc_roc':
                row += f" & {value:.4f}"
            else:
                row += f" & {value*100:.2f}\\%"
        row += " \\\\"
        latex.append(row)

    latex.append("\\bottomrule")
    latex.append("\\end{tabular}")
    latex.append("\\end{table}")

    return '\n'.join(latex)
