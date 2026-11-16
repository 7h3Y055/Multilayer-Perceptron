#!/usr/bin/env python3
"""
eda.py - Exploratory Data Analysis for the Wisconsin Diagnostic Breast Cancer dataset.
Subject IV.2: "A good practice would be to begin by exploring the dataset,
displaying it with graphs, visualizing, and manipulating its different features."
"""

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt

from src.dataset import load_dataset

FEATURE_NAMES = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis on dataset.")
    parser.add_argument("--dataset", type=str, default="data.csv", help="Path to raw dataset.")
    parser.add_argument("--plot-dist", type=str, default="eda_distributions.png", help="Path for feature distributions plot.")
    parser.add_argument("--plot-corr", type=str, default="eda_correlations.png", help="Path for correlation heatmap plot.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def plot_distributions(X: np.ndarray, y: np.ndarray, save_path: str = None):
    """Plot histograms of key features separated by class (M vs B)."""
    y_flat = y.reshape(-1)
    indices = [0, 1, 2, 3, 4, 7]  # radius, texture, perimeter, area, smoothness, concave_points

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for plot_idx, feat_idx in enumerate(indices):
        name = FEATURE_NAMES[feat_idx]
        ax = axes[plot_idx]
        feat_b = X[y_flat == 0, feat_idx]
        feat_m = X[y_flat == 1, feat_idx]

        ax.hist(feat_b, bins=25, alpha=0.65, label="Benign (B)", color="#2ca02c", density=True, edgecolor="black", linewidth=0.5)
        ax.hist(feat_m, bins=25, alpha=0.65, label="Malignant (M)", color="#d62728", density=True, edgecolor="black", linewidth=0.5)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Density")
        ax.grid(True, linestyle=":", alpha=0.5)
        ax.legend()

    plt.suptitle("Breast Cancer Feature Distributions: Malignant (M) vs Benign (B)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"> Feature distribution plot saved to '{save_path}'")


def plot_correlation_matrix(X: np.ndarray, save_path: str = None):
    """Plot correlation heatmap for the 10 'mean' features."""
    mean_features = X[:, :10]
    mean_names = [name.replace("_mean", "") for name in FEATURE_NAMES[:10]]

    # Compute Pearson correlation matrix
    corr = np.corrcoef(mean_features, rowvar=False)

    fig, ax = plt.subplots(figsize=(10, 8))
    cax = ax.imshow(corr, cmap="coolwarm", vmin=-1.0, vmax=1.0)
    fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(mean_names, rotation=45, ha="right", fontsize=10)
    ax.set_yticklabels(mean_names, fontsize=10)

    # Annotate correlation values in cells
    for i in range(10):
        for j in range(10):
            val = corr[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8)

    ax.set_title("Correlation Heatmap (Core 'Mean' Features)", fontsize=14, fontweight="bold", pad=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"> Correlation matrix plot saved to '{save_path}'")


def main():
    args = parse_args()

    try:
        X, y = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    y_flat = y.reshape(-1)
    num_samples, num_features = X.shape
    num_malignant = int(np.sum(y_flat == 1))
    num_benign = int(np.sum(y_flat == 0))

    print("=" * 65)
    print("           EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 65)
    print(f"Dataset path       : {args.dataset}")
    print(f"Total samples      : {num_samples}")
    print(f"Total features     : {num_features}")
    print(f"Malignant (M / 1)  : {num_malignant} ({num_malignant / num_samples * 100:.2f}%)")
    print(f"Benign    (B / 0)  : {num_benign} ({num_benign / num_samples * 100:.2f}%)")
    print("-" * 65)

    print("Feature Scale Discrepancies (Why Normalization is Vital):")
    print(f"  {'Feature Name':<25} {'Min':>10} {'Max':>10} {'Mean':>10} {'Std':>10}")
    print(f"  {'-'*25} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
    indices_to_show = [0, 1, 2, 3, 4, 7]
    for idx in indices_to_show:
        name = FEATURE_NAMES[idx]
        f_min = np.min(X[:, idx])
        f_max = np.max(X[:, idx])
        f_mean = np.mean(X[:, idx])
        f_std = np.std(X[:, idx])
        print(f"  {name:<25} {f_min:10.2f} {f_max:10.2f} {f_mean:10.2f} {f_std:10.2f}")
    print("=" * 65)

    plot_distributions(X, y, args.plot_dist)
    plot_correlation_matrix(X, args.plot_corr)

    if args.show:
        print("> Displaying plots interactively... (close the plot windows to exit)")
        plt.show()

    plt.close("all")


if __name__ == "__main__":
    main()
