#!/usr/bin/env python3
"""
split.py - Dataset splitting program.
Subject IV.4: "A program to separate the dataset into two parts, one for training and the other for validation"
"""

import argparse
import os
import random
import sys


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("pos_dataset", nargs="?", default=None, help="Path to raw dataset (positional).")
    parser.add_argument("--dataset", type=str, default="data.csv", help="Path to raw dataset (flag, default: data.csv).")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set (default: 0.8).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-out", type=str, default="dataset/data_train.csv", help="Path for output training CSV.")
    parser.add_argument("--val-out", type=str, default="dataset/data_valid.csv", help="Path for output validation CSV.")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset_file = args.pos_dataset if args.pos_dataset is not None else args.dataset

    try:
        with open(dataset_file, "r", encoding="utf-8") as f:
            lines = [line for line in f if line.strip()]
    except Exception as e:
        print(f"Error reading '{dataset_file}': {e}", file=sys.stderr)
        sys.exit(1)

    # Shuffle lines with reproducible seed
    if args.seed is not None:
        random.seed(args.seed)
    random.shuffle(lines)

    # Split lines
    split_idx = int(len(lines) * args.train_ratio)
    train_lines = lines[:split_idx]
    val_lines = lines[split_idx:]

    # Ensure output directories exist
    for out_path in (args.train_out, args.val_out):
        out_dir = os.path.dirname(out_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)

    with open(args.train_out, "w", encoding="utf-8") as f:
        f.writelines(train_lines)

    with open(args.val_out, "w", encoding="utf-8") as f:
        f.writelines(val_lines)

    print(f"Total samples:  {len(lines)}")
    print(f"Training set:   {len(train_lines)} samples -> '{args.train_out}'")
    print(f"Validation set: {len(val_lines)} samples -> '{args.val_out}'")
    print("Dataset split completed successfully.")


if __name__ == "__main__":
    main()
