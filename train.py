#!/usr/bin/env python3
"""
train.py - Multilayer Perceptron training program.
Subject IV.4: "A training program"
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Train Multilayer Perceptron model.")
    parser.add_argument("--dataset", type=str, default="dataset/data_train.csv", help="Path to training dataset.")
    parser.add_argument("--val-dataset", type=str, default="dataset/data_valid.csv", help="Path to validation dataset.")
    parser.add_argument(
        "--layer",
        type=int,
        nargs="+",
        default=[24, 24],
        help="Hidden layer dimensions (e.g. --layer 24 24)",
    )
    parser.add_argument("--epochs", type=int, default=70, help="Number of training epochs.")
    parser.add_argument("--batch_size", type=int, default=8, help="Mini-batch size.")
    parser.add_argument("--learning_rate", type=float, default=0.0314, help="Learning rate.")
    parser.add_argument(
        "--loss",
        type=str,
        default="categoricalCrossentropy",
        help="Loss function.",
    )
    parser.add_argument(
        "--optimizer",
        type=str,
        default="sgd",
        choices=["sgd", "momentum", "nesterov", "rmsprop", "adam"],
        help="Optimizer.",
    )
    parser.add_argument("--early_stopping", action="store_true", help="Enable early stopping.")
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping.")
    parser.add_argument("--save", type=str, default="./saved_model.npy", help="Path to save trained model.")
    parser.add_argument("--plot", type=str, default="learning_curves.png", help="Path to save learning curves.")
    parser.add_argument("--show-plot", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
