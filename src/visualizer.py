"""
src/visualizer.py - Plotting utilities for learning curves and model comparisons.
"""

from typing import Dict, List, Optional
import matplotlib.pyplot as plt


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = "learning_curves.png",
    show: bool = False,
) -> None:
    """Plot Loss and Accuracy curves for training and validation across epochs.
    Matches Figure IV.1 and Figure IV.2 in Subject Chapter IV.3.
    """
    epochs = range(1, len(history["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Figure IV.1: Loss Curve ---
    axes[0].plot(epochs, history["loss"], label="training loss", color="#1f77b4")
    if "val_loss" in history and history["val_loss"]:
        axes[0].plot(epochs, history["val_loss"], label="validation loss", color="#ff7f0e", linestyle="--")
    axes[0].set_title("Loss", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Epochs", fontsize=11)
    axes[0].set_ylabel("Loss", fontsize=11)
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend()

    # --- Figure IV.2: Accuracy Curve ---
    if "accuracy" in history and history["accuracy"]:
        axes[1].plot(epochs, history["accuracy"], label="training acc", color="#1f77b4")
    if "val_accuracy" in history and history["val_accuracy"]:
        axes[1].plot(epochs, history["val_accuracy"], label="validation acc", color="#ff7f0e")
    axes[1].set_title("Accuracy", fontsize=13, fontweight="bold")
    axes[1].set_xlabel("Epochs", fontsize=11)
    axes[1].set_ylabel("Accuracy", fontsize=11)
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"> Learning curves saved to '{save_path}'")

    if show:
        plt.show()

    plt.close()


def plot_multiple_models_comparison(
    experiments: Dict[str, Dict[str, List[float]]],
    metric: str = "val_loss",
    save_path: Optional[str] = "models_comparison.png",
    show: bool = False,
) -> None:
    """Plot multiple models on the same graph for comparison (Bonus Chapter V)."""
    plt.figure(figsize=(10, 6))

    for model_name, hist in experiments.items():
        if metric in hist:
            epochs = range(1, len(hist[metric]) + 1)
            plt.plot(epochs, hist[metric], label=model_name)

    plt.title(f"Model Comparison: {metric.replace('_', ' ').title()}", fontsize=13, fontweight="bold")
    plt.xlabel("Epochs", fontsize=11)
    plt.ylabel(metric.replace("_", " ").title(), fontsize=11)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"> Model comparison plot saved to '{save_path}'")

    if show:
        plt.show()

    plt.close()
