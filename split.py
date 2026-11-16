#!/usr/bin/env python3
"""
split.py - Dataset splitting program.
Subject IV.4: "A program to separate the dataset into two parts, one for training and the other for validation"
"""

import argparse
import sys
from src.dataset import load_dataset, train_val_split, save_split_csv


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("--dataset", type=str, default="data.csv", help="Path to raw dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-out", type=str, default="data_train.csv", help="Path for output training CSV.")
    parser.add_argument("--val-out", type=str, default="data_valid.csv", help="Path for output validation CSV.")
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading dataset from '{args.dataset}'...")
    try:
        X, y = load_dataset(args.dataset)
    except Exception as e:
        print(f"Error loading dataset: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Total samples loaded: {X.shape[0]}, Features: {X.shape[1]}")

    X_train, X_val, y_train, y_val = train_val_split(
        X, y, train_ratio=args.train_ratio, seed=args.seed
    )

    save_split_csv(args.train_out, X_train, y_train)
    save_split_csv(args.val_out, X_val, y_val)

    print(f"Training set:   {X_train.shape[0]} samples -> '{args.train_out}'")
    print(f"Validation set: {X_val.shape[0]} samples -> '{args.val_out}'")
    print("Dataset split completed successfully.")


if __name__ == "__main__":
    main()
