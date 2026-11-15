"""
src/losses.py - Loss functions and evaluation metrics.
"""

from typing import Dict
import numpy as np


def categorical_crossentropy(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Categorical cross-entropy loss for training with Softmax output."""
    pass


def binary_crossentropy(
    p: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Binary cross-entropy error function as defined in subject IV.4."""
    pass


def compute_accuracy(y_pred_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate accuracy score."""
    pass


def compute_classification_metrics(
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Calculate confusion matrix, precision, recall, specificity, and F1-score."""
    pass
