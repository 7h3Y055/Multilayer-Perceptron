"""
src/visualizer.py - Plotting utilities for learning curves and model comparisons.
"""

from typing import Dict, List, Optional


def plot_learning_curves(
    history: Dict[str, List[float]],
    save_path: Optional[str] = "learning_curves.png",
    show: bool = False,
) -> None:
    """Plot Loss and Accuracy curves for training and validation across epochs."""
    pass


def plot_multiple_models_comparison(
    experiments: Dict[str, Dict[str, List[float]]],
    metric: str = "val_loss",
    save_path: Optional[str] = "models_comparison.png",
    show: bool = False,
) -> None:
    """Plot multiple models on the same graph for comparison (Bonus)."""
    pass
