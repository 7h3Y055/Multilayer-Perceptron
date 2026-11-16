"""
src/dataset.py - Data loading, preprocessing, Z-score standardization, and dataset splitting.
"""

import csv
from typing import Tuple, Optional
import numpy as np


class StandardScaler:
    """Standardizes features by removing the mean and scaling to unit variance.

    Formula:
        X_norm = (X - mu) / (sigma + eps)
    """

    def __init__(self, eps: float = 1e-8):
        self.mean: Optional[np.ndarray] = None
        self.std: Optional[np.ndarray] = None
        self.eps = eps

    def fit(self, X: np.ndarray) -> "StandardScaler":
        """Compute the mean and std to be used for later scaling."""
        self.mean = np.mean(X, axis=0, keepdims=True)
        self.std = np.std(X, axis=0, keepdims=True)
        # Prevent division by zero for constant features
        self.std = np.where(self.std == 0, 1.0, self.std)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Perform standardization by centering and scaling."""
        if self.mean is None or self.std is None:
            raise ValueError("StandardScaler must be fitted before transform.")
        return (X - self.mean) / (self.std + self.eps)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(X).transform(X)

    def to_dict(self) -> dict:
        """Serialize parameters for model persistence."""
        return {
            "mean": self.mean.tolist() if self.mean is not None else None,
            "std": self.std.tolist() if self.std is not None else None,
            "eps": self.eps,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "StandardScaler":
        """Load scaler from serialized parameters."""
        scaler = cls(eps=data.get("eps", 1e-8))
        if data.get("mean") is not None:
            scaler.mean = np.array(data["mean"], dtype=np.float64)
        if data.get("std") is not None:
            scaler.std = np.array(data["std"], dtype=np.float64)
        return scaler


def load_dataset(filepath: str) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw Wisconsin breast cancer dataset from CSV.

    Expected format per row:
        col 0: ID (int)
        col 1: Diagnosis ('M' or 'B')
        cols 2-31: 30 numerical features

    Returns:
        X: np.ndarray of shape (N, 30), dtype float64
        y: np.ndarray of shape (N, 1), dtype int (1 for 'M', 0 for 'B')
    """
    features = []
    labels = []

    with open(filepath, mode="r", encoding="utf-8") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or len(row) < 32:
                continue
            diagnosis = row[1].strip().upper()
            if diagnosis not in ("M", "B"):
                # Header row or invalid entry
                continue

            label = 1 if diagnosis == "M" else 0
            feature_row = [float(val) for val in row[2:32]]

            features.append(feature_row)
            labels.append([label])

    X = np.array(features, dtype=np.float64)
    y = np.array(labels, dtype=np.int64)
    return X, y


def to_one_hot(y: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Convert integer labels (N, 1) or (N,) to one-hot encoded matrix (N, num_classes).

    For binary classification:
        0 (Benign)    -> [1.0, 0.0]
        1 (Malignant) -> [0.0, 1.0]
    """
    y_flat = y.reshape(-1)
    one_hot = np.zeros((y_flat.shape[0], num_classes), dtype=np.float64)
    one_hot[np.arange(y_flat.shape[0]), y_flat] = 1.0
    return one_hot


def train_val_split(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float = 0.8,
    seed: Optional[int] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Randomly split X and y into training and validation sets."""
    if seed is not None:
        np.random.seed(seed)

    num_samples = X.shape[0]
    indices = np.random.permutation(num_samples)

    train_size = int(num_samples * train_ratio)
    train_idx = indices[:train_size]
    val_idx = indices[train_size:]

    return X[train_idx], X[val_idx], y[train_idx], y[val_idx]
