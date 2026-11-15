"""
src/optimizers.py - Optimization algorithms (SGD, Momentum, Nesterov, RMSprop, Adam).
"""

from typing import List
from src.layers import DenseLayer


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, learning_rate: float = 0.01):
        self.learning_rate = learning_rate

    def update(self, layers: List[DenseLayer]) -> None:
        raise NotImplementedError


class SGD(Optimizer):
    """Stochastic / Mini-Batch Gradient Descent."""

    def update(self, layers: List[DenseLayer]) -> None:
        pass


class Momentum(Optimizer):
    """Momentum optimizer with optional Nesterov Accelerated Gradient."""

    def __init__(
        self,
        learning_rate: float = 0.01,
        beta: float = 0.9,
        nesterov: bool = False,
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.nesterov = nesterov

    def update(self, layers: List[DenseLayer]) -> None:
        pass


class RMSprop(Optimizer):
    """RMSprop optimizer."""

    def __init__(
        self,
        learning_rate: float = 0.001,
        beta: float = 0.9,
        eps: float = 1e-8,
    ):
        super().__init__(learning_rate)
        self.beta = beta
        self.eps = eps

    def update(self, layers: List[DenseLayer]) -> None:
        pass


class Adam(Optimizer):
    """Adam (Adaptive Moment Estimation) optimizer."""

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

    def update(self, layers: List[DenseLayer]) -> None:
        pass


def get_optimizer(name: str, learning_rate: float = 0.01) -> Optimizer:
    """Factory function returning the optimizer instance by name."""
    pass
