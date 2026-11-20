#!/usr/bin/env python3
"""
train.py - Multilayer Perceptron training program.
Subject IV.4: "A training program"
"""

import argparse
import sys
import numpy as np

from src.dataset import load_dataset, train_val_split, StandardScaler
from src.layers import DenseLayer
from src.network import NeuralNetwork
from src.visualizer import plot_learning_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multilayer Perceptron model.")
    parser.add_argument("--dataset", type=str, default="dataset/data_train.csv", help="Path to training dataset.")
    parser.add_argument("--val-dataset", type=str, default="dataset/data_valid.csv", help="Path to validation dataset.")
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[24, 24],
        help="Hidden layer dimensions (must have at least 2 hidden layers by default, e.g. --layer 24 24)",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs (default: 70).")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size (default: 8).")
    parser.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate (default: 0.0314).")
    parser.add_argument(
        "--loss",
        type=str,
        default="categoricalCrossentropy",
        help="Loss function (default: categoricalCrossentropy).",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "adam"],
        help="Optimizer: 'sgd' (mandatory) or 'adam' (bonus).",
    )
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping based on val_loss.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping (default: 10).")
    parser.add_argument("--save", type=str, default="./saved_model.npy", help="Path to save trained model.")
    parser.add_argument("--plot", type=str, default="learning_curves.png", help="Path to save learning curves.")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Subject constraint: at least two hidden layers by default
    if len(args.layer) < 2:
        print("Error: The neural network must contain at least two hidden layers by default.", file=sys.stderr)
        sys.exit(1)

    # 1. Load training and validation datasets
    try:
        X_train_raw, y_train = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading training dataset from '{args.dataset}': {e}", file=sys.stderr)
        sys.exit(1)

    if args.val_dataset:
        try:
            X_val_raw, y_val = load_dataset(args.val_dataset)
        except Exception as e:
            print(f"Error loading validation dataset from '{args.val_dataset}': {e}", file=sys.stderr)
            sys.exit(1)
    else:
        # Fallback in-memory split if only one dataset was provided
        X_train_raw, X_val_raw, y_train, y_val = train_val_split(X_train_raw, y_train, train_ratio=0.8, seed=42)

    # 2. Preprocessing: Fit scaler strictly on training data, then transform both
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_val = scaler.transform(X_val_raw)

    print(f"x_train shape : {X_train.shape}")
    print(f"x_valid shape : {X_val.shape}")

    # 3. Build Neural Network architecture
    input_dim = X_train.shape[1]  # 30 features
    num_classes = 2               # Binary classification with 2 output neurons (Softmax)

    network = NeuralNetwork()

    # First hidden layer
    network.add(
        DenseLayer(
            output_dim=args.layer[0],
            input_dim=input_dim,
            activation="sigmoid",
            weights_initializer="heUniform",
        )
    )

    # Middle hidden layer(s)
    for hidden_dim in args.layer[1:]:
        network.add(
            DenseLayer(
                output_dim=hidden_dim,
                activation="sigmoid",
                weights_initializer="heUniform",
            )
        )

    # Output layer (Softmax as required by Subject IV.3)
    network.add(
        DenseLayer(
            output_dim=num_classes,
            activation="softmax",
            weights_initializer="heUniform",
        )
    )

    # 4. Train the network
    network.fit(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        optimizer=args.optimizer,
        early_stopping=args.early_stopping,
        patience=args.patience,
        verbose=True,
    )

    # 5. Save model parameters and scaler
    network.save(
        args.save,
        scaler=scaler,
        extra_meta={
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "optimizer": args.optimizer,
            "loss": args.loss,
            "layer": args.layer,
        },
    )

    # 6. Render learning curves
    plot_learning_curves(network.history, save_path=args.plot, show=args.show_plot)


if __name__ == "__main__":
    main()
