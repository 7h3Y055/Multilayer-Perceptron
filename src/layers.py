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
    """Sigmoid: squashes numbers between 0 and 1."""
    z_clipped = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z_clipped))


def sigmoid_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid where 'a' is already sigmoid(z)."""
    return a * (1.0 - a)


def relu(z: np.ndarray) -> np.ndarray:
    """ReLU (Rectified Linear Unit): returns max(0, z)."""
    return np.maximum(0.0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    """Derivative of ReLU: 1 if z > 0, otherwise 0."""
    return (z > 0.0).astype(np.float64)


def tanh(z: np.ndarray) -> np.ndarray:
    """Hyperbolic tangent: squashes numbers between -1 and +1."""
    return np.tanh(z)


def tanh_derivative(a: np.ndarray) -> np.ndarray:
    """Derivative of tanh where 'a' is already tanh(z)."""
    return 1.0 - np.square(a)


def softmax(z: np.ndarray) -> np.ndarray:
    """Softmax: converts raw scores into probabilities that sum to 1.
    Subtracting np.max(z) prevents numerical overflow (e^(huge number) = inf).
    """
    shift_z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(shift_z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


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
