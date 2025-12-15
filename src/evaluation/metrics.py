"""
Evaluation metrics and visualization utilities.
"""
import numpy as np
import pandas as pd
from typing import List, Dict, Optional, Tuple, Union
from collections import Counter


def evaluate_predictions(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Dict[str, float]:
    """
    Compute evaluation metrics for predictions.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of all possible labels
    
    Returns:
        Dictionary with accuracy, macro/weighted F1, etc.
    """
    from sklearn.metrics import (
        accuracy_score, 
        f1_score, 
        precision_score, 
        recall_score,
        classification_report
    )
    
    # Filter out None predictions
    valid_indices = [i for i, p in enumerate(y_pred) if p is not None]
    y_true_valid = [y_true[i] for i in valid_indices]
    y_pred_valid = [y_pred[i] for i in valid_indices]
    
    coverage = len(valid_indices) / len(y_true) if y_true else 0.0
    
    if not y_true_valid:
        return {
            'accuracy': 0.0,
            'f1_macro': 0.0,
            'f1_weighted': 0.0,
            'precision_macro': 0.0,
            'recall_macro': 0.0,
            'coverage': coverage
        }
    
    metrics = {
        'accuracy': accuracy_score(y_true_valid, y_pred_valid),
        'f1_macro': f1_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'f1_weighted': f1_score(y_true_valid, y_pred_valid, average='weighted', zero_division=0),
        'precision_macro': precision_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'recall_macro': recall_score(y_true_valid, y_pred_valid, average='macro', zero_division=0),
        'coverage': coverage
    }
    
    return metrics


def get_classification_report(
    y_true: List[str],
    y_pred: List[str],
    output_format: str = 'dict'
) -> Union[str, Dict]:
    """
    Generate detailed classification report.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        output_format: 'dict' or 'string'
    
    Returns:
        Classification report as dict or string
    """
    from sklearn.metrics import classification_report
    
    return classification_report(
        y_true, y_pred, 
        output_dict=(output_format == 'dict'),
        zero_division=0
    )


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names (for ordering)
    
    Returns:
        Tuple of (confusion_matrix, label_names)
    """
    from sklearn.metrics import confusion_matrix
    
    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    return cm, labels


def plot_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: Optional[List[str]] = None,
    title: str = "Confusion Matrix",
    figsize: Tuple[int, int] = (10, 8),
    normalize: bool = True,
    save_path: Optional[str] = None
):
    """
    Plot confusion matrix with matplotlib/seaborn.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        labels: Optional list of label names
        title: Plot title
        figsize: Figure size
        normalize: Whether to normalize by row (true labels)
        save_path: Optional path to save the figure
    
    Returns:
        matplotlib figure
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    
    cm, label_names = compute_confusion_matrix(y_true, y_pred, labels)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        cm = np.nan_to_num(cm)  # Replace NaN with 0
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        cm, 
        annot=True, 
        fmt=fmt, 
        cmap='Blues',
        xticklabels=label_names,
        yticklabels=label_names,
        ax=ax
    )
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title(title)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def compare_models(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['accuracy', 'f1_macro', 'f1_weighted']
) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        results: Dictionary of {model_name: metrics_dict}
        metrics: List of metric names to include
    
    Returns:
        DataFrame comparing models
    """
    comparison = []
    
    for model_name, model_metrics in results.items():
        row = {'model': model_name}
        row.update({m: model_metrics.get(m, 0.0) for m in metrics})
        comparison.append(row)
    
    df = pd.DataFrame(comparison)
    df = df.sort_values('f1_macro', ascending=False)
    
    return df


def analyze_errors(
    texts: List[str],
    y_true: List[str],
    y_pred: List[str],
    n_samples: int = 10
) -> pd.DataFrame:
    """
    Analyze misclassified examples.
    
    Args:
        texts: Input texts
        y_true: Ground truth labels
        y_pred: Predicted labels
        n_samples: Number of error samples to return
    
    Returns:
        DataFrame with error analysis
    """
    errors = []
    
    for text, true, pred in zip(texts, y_true, y_pred):
        if true != pred:
            errors.append({
                'text': text[:100] + '...' if len(text) > 100 else text,
                'true_label': true,
                'predicted_label': pred
            })
    
    df = pd.DataFrame(errors)
    
    if len(df) > n_samples:
        df = df.sample(n=n_samples, random_state=42)
    
    return df
