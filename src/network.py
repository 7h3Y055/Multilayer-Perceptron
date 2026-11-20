"""
src/network.py - NeuralNetwork container managing forward, backward, fit, predict, save, and load.
"""

import json
import os
from typing import List, Optional, Dict, Tuple, Any
import numpy as np

from src.layers import DenseLayer
from src.optimizers import Optimizer, get_optimizer
from src.losses import categorical_crossentropy, compute_accuracy
from src.dataset import StandardScaler, to_one_hot


class NeuralNetwork:
    """Multilayer Perceptron container."""

    def __init__(self, layers: Optional[List[DenseLayer]] = None):
        self.layers: List[DenseLayer] = layers if layers is not None else []
        self.history: Dict[str, List[float]] = {
            "loss": [],
            "val_loss": [],
            "accuracy": [],
            "val_accuracy": [],
        }

    def add(self, layer: DenseLayer) -> None:
        """Append a dense layer to the network."""
        self.layers.append(layer)

    def forward(self, inputs: np.ndarray) -> np.ndarray:
        """Sequential forward pass across all layers."""
        out = inputs
        for layer in self.layers:
            out = layer.forward(out)
        return out

    def backward(self, grad_output: np.ndarray) -> None:
        """Sequential backward pass across all layers in reverse order."""
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

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
        """Train the network using mini-batch gradient descent and backpropagation."""
        num_classes = self.layers[-1].output_dim

        # Ensure one-hot targets for Categorical Cross-Entropy with Softmax
        if y_train.ndim == 1 or y_train.shape[1] == 1:
            y_train_oh = to_one_hot(y_train, num_classes=num_classes)
        else:
            y_train_oh = y_train

        if X_val is not None and y_val is not None:
            if y_val.ndim == 1 or y_val.shape[1] == 1:
                y_val_oh = to_one_hot(y_val, num_classes=num_classes)
            else:
                y_val_oh = y_val
        else:
            y_val_oh = None

        opt: Optimizer = get_optimizer(optimizer, learning_rate=learning_rate)
        num_samples = X_train.shape[0]
        self.history = {"loss": [], "val_loss": [], "accuracy": [], "val_accuracy": []}

        # Early stopping tracker (Bonus Chapter V)
        best_val_loss = float("inf")
        patience_counter = 0
        best_weights: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None

        for epoch in range(1, epochs + 1):
            # Shuffle training set at the beginning of each epoch
            perm = np.random.permutation(num_samples)
            X_shuffled = X_train[perm]
            y_shuffled = y_train_oh[perm]

            # Mini-batch gradient descent loop
            for i in range(0, num_samples, batch_size):
                X_batch = X_shuffled[i : i + batch_size]
                y_batch = y_shuffled[i : i + batch_size]

                # 1. Forward pass
                y_pred = self.forward(X_batch)

                # 2. Output error delta = (A - Y) for Softmax + Cross-Entropy
                grad_output = y_pred - y_batch

                # 3. Backward pass (computes dW and db)
                self.backward(grad_output)

                # 4. Update parameters using optimizer
                opt.update(self.layers)

            # Evaluate epoch metrics
            train_preds = self.forward(X_train)
            train_loss = categorical_crossentropy(train_preds, y_train_oh)
            train_acc = compute_accuracy(train_preds, y_train_oh)

            self.history["loss"].append(train_loss)
            self.history["accuracy"].append(train_acc)

            val_log = ""
            if X_val is not None and y_val_oh is not None:
                val_preds = self.forward(X_val)
                val_loss = categorical_crossentropy(val_preds, y_val_oh)
                val_acc = compute_accuracy(val_preds, y_val_oh)

                self.history["val_loss"].append(val_loss)
                self.history["val_accuracy"].append(val_acc)
                val_log = f" - val_loss: {val_loss:.4f}"

                # Early stopping check
                if early_stopping:
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        patience_counter = 0
                        best_weights = [(np.copy(l.weights), np.copy(l.biases)) for l in self.layers]
                    else:
                        patience_counter += 1
                        if patience_counter >= patience:
                            if verbose:
                                print(f"Early stopping triggered at epoch {epoch:02d}/{epochs:02d} (patience={patience}).")
                            if best_weights is not None:
                                for l_idx, (w, b) in enumerate(best_weights):
                                    self.layers[l_idx].weights = w
                                    self.layers[l_idx].biases = b
                            break

            if verbose:
                # Matches exact subject format: epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
                print(f"epoch {epoch:02d}/{epochs:02d} - loss: {train_loss:.4f}{val_log}")

        return self.history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Returns probabilistic output from Softmax (N, num_classes)."""
        return self.forward(X)

    def predict_classes(self, X: np.ndarray) -> np.ndarray:
        """Returns predicted class index (0 or 1)."""
        probs = self.forward(X)
        return np.argmax(probs, axis=-1)

    def save(
        self,
        filepath: str,
        scaler: Optional[StandardScaler] = None,
        extra_meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Saves network topology, weights, biases, and scaler to disk."""
        topology = []
        weights_dict = {}

        for idx, layer in enumerate(self.layers):
            topology.append({
                "layer_idx": idx,
                "input_dim": layer.input_dim,
                "output_dim": layer.output_dim,
                "activation": layer.activation_name,
                "weights_initializer": layer.weights_initializer,
            })
            weights_dict[f"w_{idx}"] = layer.weights
            weights_dict[f"b_{idx}"] = layer.biases

        meta = {
            "topology": topology,
            "scaler": scaler.to_dict() if scaler else None,
            "history": self.history,
            "extra": extra_meta or {},
        }

        # Using an open file handle prevents np.savez from forcing a .npz extension
        with open(filepath, "wb") as f:
            np.savez(f, meta=np.array(json.dumps(meta)), **weights_dict)
        print(f"> saving model '{filepath}' to disk...")

    @classmethod
    def load(cls, filepath: str) -> Tuple["NeuralNetwork", Optional[StandardScaler], Dict[str, Any]]:
        """Loads saved model, reconstructing network architecture, weights, and scaler."""
        actual_path = filepath
        if not os.path.exists(actual_path) and os.path.exists(actual_path + ".npz"):
            actual_path = actual_path + ".npz"

        data = np.load(actual_path, allow_pickle=True)
        meta = json.loads(str(data["meta"]))

        layers = []
        for l_meta in meta["topology"]:
            layer = DenseLayer(
                output_dim=l_meta["output_dim"],
                input_dim=l_meta["input_dim"],
                activation=l_meta["activation"],
                weights_initializer=l_meta["weights_initializer"],
            )
            idx = l_meta["layer_idx"]
            layer.weights = data[f"w_{idx}"]
            layer.biases = data[f"b_{idx}"]
            layers.append(layer)

        model = cls(layers=layers)
        model.history = meta.get("history", {})

        scaler = None
        if meta.get("scaler") is not None:
            scaler = StandardScaler.from_dict(meta["scaler"])

        return model, scaler, meta
