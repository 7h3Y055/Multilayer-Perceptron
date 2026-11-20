"""
src/optimizers.py - Optimization algorithms (SGD, Adam).
"""

from typing import List, Dict
import numpy as np
from src.layers import DenseLayer


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, layers: List[DenseLayer]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic / Mini-Batch Gradient Descent (Mandatory).
    
    Formula:
        W = W - lr * dW
        b = b - lr * db
    """

    def update(self, layers: List[DenseLayer]) -> None:
        for layer in layers:
            if layer.weights is None or layer.grad_weights is None:
                continue
            layer.weights -= self.learning_rate * layer.grad_weights
            layer.biases -= self.learning_rate * layer.grad_biases


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer (Bonus).
    
    Combines momentum (first moment m) and RMSprop (second moment v) with bias correction.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m_w: Dict[int, np.ndarray] = {}
        self.m_b: Dict[int, np.ndarray] = {}
        self.v_w: Dict[int, np.ndarray] = {}
        self.v_b: Dict[int, np.ndarray] = {}

    def update(self, layers: List[DenseLayer]) -> None:
        self.t += 1
        for idx, layer in enumerate(layers):
            if layer.weights is None or layer.grad_weights is None:
                continue

            if idx not in self.m_w:
                self.m_w[idx] = np.zeros_like(layer.weights)
                self.m_b[idx] = np.zeros_like(layer.biases)
                self.v_w[idx] = np.zeros_like(layer.weights)
                self.v_b[idx] = np.zeros_like(layer.biases)

            # Update biased 1st and 2nd moment estimates
            self.m_w[idx] = self.beta1 * self.m_w[idx] + (1.0 - self.beta1) * layer.grad_weights
            self.m_b[idx] = self.beta1 * self.m_b[idx] + (1.0 - self.beta1) * layer.grad_biases
            self.v_w[idx] = self.beta2 * self.v_w[idx] + (1.0 - self.beta2) * np.square(layer.grad_weights)
            self.v_b[idx] = self.beta2 * self.v_b[idx] + (1.0 - self.beta2) * np.square(layer.grad_biases)

            # Compute bias-corrected estimates
            m_w_hat = self.m_w[idx] / (1.0 - self.beta1 ** self.t)
            m_b_hat = self.m_b[idx] / (1.0 - self.beta1 ** self.t)
            v_w_hat = self.v_w[idx] / (1.0 - self.beta2 ** self.t)
            v_b_hat = self.v_b[idx] / (1.0 - self.beta2 ** self.t)

            # Apply parameter updates
            layer.weights -= (self.learning_rate / (np.sqrt(v_w_hat) + self.eps)) * m_w_hat
            layer.biases -= (self.learning_rate / (np.sqrt(v_b_hat) + self.eps)) * m_b_hat


def get_optimizer(name: str, learning_rate: float = 0.01) -> Optimizer:
    """Factory function returning the optimizer instance by name."""
    opt = name.lower()
    if opt in ("sgd", "gradient_descent"):
        return SGD(learning_rate=learning_rate)
    elif opt in ("adam",):
        return Adam(learning_rate=learning_rate)
    else:
        raise ValueError(f"Unknown optimizer: '{name}'. Choose from: 'sgd' (mandatory) or 'adam' (bonus).")
