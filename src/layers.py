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
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation_name = activation.lower() if activation else None
        self.weights_initializer = weights_initializer

        self.weights: Optional[np.ndarray] = None
        self.biases: Optional[np.ndarray] = None

        self.grad_weights: Optional[np.ndarray] = None
        self.grad_biases: Optional[np.ndarray] = None

        # Caches saved during forward pass for backpropagation
        self.inputs_cache: Optional[np.ndarray] = None
        self.z_cache: Optional[np.ndarray] = None
        self.a_cache: Optional[np.ndarray] = None

        if self.input_dim is not None:
            self.build(self.input_dim)

    def build(self, input_dim: int) -> None:
        """Initialize weights and biases based on the input dimension."""
        self.input_dim = input_dim
        init_mode = self.weights_initializer.lower()

        # 1. He (Kaiming) Uniform: limit = sqrt(6 / n_in)
        if init_mode in ("heuniform", "he_uniform"):
            limit = np.sqrt(6.0 / input_dim)
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.output_dim))
        # 2. He Normal: std = sqrt(2 / n_in)
        elif init_mode in ("henormal", "he_normal"):
            std = np.sqrt(2.0 / input_dim)
            self.weights = np.random.normal(0.0, std, (input_dim, self.output_dim))
        # 3. Xavier (Glorot) Uniform: limit = sqrt(6 / (n_in + n_out))
        elif init_mode in ("xavieruniform", "xavier_uniform", "glorot_uniform"):
            limit = np.sqrt(6.0 / (input_dim + self.output_dim))
            self.weights = np.random.uniform(-limit, limit, (input_dim, self.output_dim))
        # 4. Xavier Normal: std = sqrt(2 / (n_in + n_out))
        elif init_mode in ("xaviernormal", "xavier_normal", "glorot_normal"):
            std = np.sqrt(2.0 / (input_dim + self.output_dim))
            self.weights = np.random.normal(0.0, std, (input_dim, self.output_dim))
        else:
            self.weights = np.random.randn(input_dim, self.output_dim) * 0.01

        # Biases are initialized to zeros: shape (1, output_dim)
        self.biases = np.zeros((1, self.output_dim), dtype=np.float64)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Forward pass: computes Z = X * W + b and applies activation."""
        if self.weights is None or self.biases is None:
            self.build(inputs.shape[1])

        # Cache inputs for backpropagation
        self.inputs_cache = inputs
        self.z_cache = np.dot(inputs, self.weights) + self.biases

        # Apply activation function
        if self.activation_name == "sigmoid":
            self.a_cache = sigmoid(self.z_cache)
        elif self.activation_name == "relu":
            self.a_cache = relu(self.z_cache)
        elif self.activation_name == "tanh":
            self.a_cache = tanh(self.z_cache)
        elif self.activation_name == "softmax":
            self.a_cache = softmax(self.z_cache)
        elif self.activation_name is None:
            self.a_cache = self.z_cache
        else:
            raise ValueError(f"Unsupported activation: {self.activation_name}")

        return self.a_cache

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """Backward pass: computes gradients dW, db and returns dX for the previous layer."""
        m = self.inputs_cache.shape[0]

        # 1. Backpropagate through the activation function to find delta = dL/dZ
        if self.activation_name == "sigmoid":
            delta = grad_output * sigmoid_derivative(self.a_cache)
        elif self.activation_name == "relu":
            delta = grad_output * relu_derivative(self.z_cache)
        elif self.activation_name == "tanh":
            delta = grad_output * tanh_derivative(self.a_cache)
        elif self.activation_name in ("softmax", None):
            # For Softmax with Cross-Entropy, grad_output is already (A - Y)
            delta = grad_output
        else:
            delta = grad_output

        # 2. Compute gradients for weights and biases
        self.grad_weights = np.dot(self.inputs_cache.T, delta) / m
        self.grad_biases = np.sum(delta, axis=0, keepdims=True) / m

        # 3. Compute gradient w.r.t input to pass down to previous layer
        grad_input = np.dot(delta, self.weights.T)
        return grad_input
