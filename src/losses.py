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
    """Categorical cross-entropy loss for training with Softmax output.

    Formula:
        L = - 1/N * sum(y_true * log(y_pred + eps))
    """
    y_pred_clipped = np.clip(y_pred, eps, 1.0 - eps)
    loss = -np.sum(y_true * np.log(y_pred_clipped)) / y_pred.shape[0]
    return float(loss)


def binary_crossentropy(
    p: np.ndarray,
    y: np.ndarray,
    eps: float = 1e-15,
) -> float:
    """Binary cross-entropy error function as defined in subject IV.4:
        E = - 1/N * sum(yn * log(pn) + (1 - yn) * log(1 - pn))

    Args:
        p: (N,) or (N, 1) predicted probability of positive class (Malignant)
        y: (N,) or (N, 1) binary target labels (0 for Benign, 1 for Malignant)
    """
    p_flat = p.reshape(-1)
    y_flat = y.reshape(-1)
    p_clipped = np.clip(p_flat, eps, 1.0 - eps)
    loss = -np.mean(y_flat * np.log(p_clipped) + (1.0 - y_flat) * np.log(1.0 - p_clipped))
    return float(loss)


def compute_accuracy(y_pred_prob: np.ndarray, y_true: np.ndarray) -> float:
    """Calculate classification accuracy score."""
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        preds = np.argmax(y_pred_prob, axis=1)
    else:
        preds = (y_pred_prob.reshape(-1) >= 0.5).astype(int)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        targets = np.argmax(y_true, axis=1)
    else:
        targets = y_true.reshape(-1).astype(int)

    return float(np.mean(preds == targets))


def compute_classification_metrics(
    y_pred_prob: np.ndarray,
    y_true: np.ndarray,
) -> Dict[str, float]:
    """Calculate full confusion matrix, precision, recall, specificity, and F1-score."""
    if y_pred_prob.ndim == 2 and y_pred_prob.shape[1] > 1:
        preds = np.argmax(y_pred_prob, axis=1)
    else:
        preds = (y_pred_prob.reshape(-1) >= 0.5).astype(int)

    if y_true.ndim == 2 and y_true.shape[1] > 1:
        targets = np.argmax(y_true, axis=1)
    else:
        targets = y_true.reshape(-1).astype(int)

    tp = int(np.sum((preds == 1) & (targets == 1)))
    tn = int(np.sum((preds == 0) & (targets == 0)))
    fp = int(np.sum((preds == 1) & (targets == 0)))
    fn = int(np.sum((preds == 0) & (targets == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    accuracy = (tp + tn) / len(targets) if len(targets) > 0 else 0.0

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "specificity": float(specificity),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }
