"""
src/network.py - NeuralNetwork container managing forward, backward, fit, predict, save, and load.
"""

from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from src.layers import DenseLayer
from src.dataset import StandardScaler


class NeuralNetwork:
    """Multilayer Perceptron container."""

    def __init__(self, layers: Optional[List[DenseLayer]] = None):
        pass

    def add(self, layer: DenseLayer) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grad_output: np.ndarray) -> None:
        pass

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: Optional[np.ndarray] = None,
        y_val: Optional[np.ndarray] = None,
        epochs: int = 70,
        batch_size: int = 8,
        learning_rate: float = 0.0314,
        optimizer: str = "sgd",
        early_stopping: bool = False,
        patience: int = 10,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        pass

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        pass

    def save(
        self,
        filepath: str,
        scaler: Optional[StandardScaler] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        pass

    @classmethod
    def load(cls, filepath: str) -> Tuple["NeuralNetwork", Optional[StandardScaler], Dict[str, Any]]:
        pass
