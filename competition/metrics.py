"""Metrics module for HeteroShot challenge.

This module contains evaluation metrics used for scoring submissions.
Customize this file to change the evaluation metric.
"""

from typing import List


def macro_f1(y_true: List[int], y_pred: List[int]) -> float:
    """Compute Macro-averaged F1 score.
    
    Macro-F1 gives equal weight to all classes regardless of their frequency.
    This is important for imbalanced datasets where minority classes
    should be given equal importance.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Macro-averaged F1 score between 0 and 1
        
    Example:
        >>> macro_f1([0, 0, 1, 1, 2], [0, 0, 1, 0, 2])
        0.777...
    """
    labels = sorted(set(y_true))
    scores = []
    
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        
        # F1 = 2 * (precision * recall) / (precision + recall)
        # Simplified: F1 = 2*TP / (2*TP + FP + FN)
        denom = (2 * tp) + fp + fn
        f1 = (2 * tp) / denom if denom > 0 else 0.0
        scores.append(f1)
    
    return sum(scores) / len(scores) if scores else 0.0


def accuracy(y_true: List[int], y_pred: List[int]) -> float:
    """Compute accuracy score.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy score between 0 and 1
    """
    if len(y_true) == 0:
        return 0.0
    correct = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp)
    return correct / len(y_true)


def precision_per_class(y_true: List[int], y_pred: List[int], label: int) -> float:
    """Compute precision for a specific class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label: Target class label
        
    Returns:
        Precision score for the specified class
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall_per_class(y_true: List[int], y_pred: List[int], label: int) -> float:
    """Compute recall for a specific class.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        label: Target class label
        
    Returns:
        Recall score for the specified class
    """
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


# Default metric used for scoring
DEFAULT_METRIC = macro_f1
METRIC_NAME = "Macro-F1"
