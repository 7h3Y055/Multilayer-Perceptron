#!/usr/bin/env python3
"""
compare.py - Train and compare multiple models on the same graph.
Subject Chapter V Bonus: "A display of multiple learning curves on the same graph (really useful to compare different models)"
"""

import argparse
import sys
import numpy as np

from src.dataset import load_dataset, StandardScaler
from src.layers import DenseLayer
from src.network import NeuralNetwork
from src.visualizer import plot_multiple_models_comparison


def parse_args():
    parser = argparse.ArgumentParser(description="Compare multiple optimizers/models on one learning curve graph.")
    parser.add_argument("--train-dataset", type=str, default="dataset/data_train.csv", help="Training dataset path.")
    parser.add_argument("--val-dataset", type=str, default="dataset/data_valid.csv", help="Validation dataset path.")
    parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs (default: 50).")
    parser.add_argument("--batch_size", type=int, default=8, help="Batch size (default: 8).")
    parser.add_argument("--plot", type=str, default="models_comparison.png", help="Path to save comparison plot.")
    return parser.parse_args()


def build_fresh_network():
    net = NeuralNetwork()
    net.add(DenseLayer(output_dim=24, input_dim=30, activation="sigmoid", weights_initializer="heUniform"))
    net.add(DenseLayer(output_dim=24, activation="sigmoid", weights_initializer="heUniform"))
    net.add(DenseLayer(output_dim=2, activation="softmax", weights_initializer="heUniform"))
    return net


def main():
    args = parse_args()

    # Load and scale data
    X_train_raw, y_train = load_dataset(args.train_dataset)
    X_val_raw, y_val = load_dataset(args.val_dataset)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print("=" * 55)
    print("      COMPARING OPTIMIZERS (SGD vs Adam)")
    print("=" * 55)

    experiments = {}

    # 1. Train SGD
    print("\n--- Training Model 1: SGD (lr=0.0314) ---")
    np.random.seed(42)
    net_sgd = build_fresh_network()
    hist_sgd = net_sgd.fit(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.0314,
        optimizer="sgd", verbose=False
    )
    experiments["SGD (lr=0.0314)"] = hist_sgd
    print(f"Final Val Loss: {hist_sgd['val_loss'][-1]:.4f} | Final Val Acc: {hist_sgd['val_accuracy'][-1]*100:.2f}%")

    # 2. Train Adam
    print("\n--- Training Model 2: Adam (lr=0.005) ---")
    np.random.seed(42)
    net_adam = build_fresh_network()
    hist_adam = net_adam.fit(
        X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, learning_rate=0.005,
        optimizer="adam", verbose=False
    )
    experiments["Adam (lr=0.005)"] = hist_adam
    print(f"Final Val Loss: {hist_adam['val_loss'][-1]:.4f} | Final Val Acc: {hist_adam['val_accuracy'][-1]*100:.2f}%")

    print("-" * 55)
    plot_multiple_models_comparison(experiments, metric="val_loss", save_path=args.plot)
    print("=" * 55)


if __name__ == "__main__":
    main()
