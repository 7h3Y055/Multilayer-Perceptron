"""
src/layers.py - Neural network layer abstractions, activation functions, and initializers.
"""

from typing import Optional
import numpy as np


class Layer:
    """Base class for all neural network layers."""

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        raise NotImplementedError


# --- Activation Functions & Derivatives ---

def sigmoid(z: np.ndarray) -> np.ndarray:
    pass


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    pass


def relu(z: np.ndarray) -> np.ndarray:
    pass


def relu_derivative(z: np.ndarray) -> np.ndarray:
    pass


def tanh(z: np.ndarray) -> np.ndarray:
    pass


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    pass


def softmax(z: np.ndarray) -> np.ndarray:
    pass


# --- Dense Layer ---

class DenseLayer(Layer):
    """Fully connected layer with weights, biases, and optional activation."""

    def __init__(
        self,
        output_dim: int,
        input_dim: Optional[int] = None,
        activation: Optional[str] = None,
        weights_initializer: str = "heUniform",
    ):
        pass

    def build(self, input_dim: int) -> None:
        pass

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        pass

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        pass
