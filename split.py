#!/usr/bin/env python3
"""
split.py - Dataset splitting program.
Subject IV.4: "A program to separate the dataset into two parts, one for training and the other for validation"
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Split dataset into training and validation sets.")
    parser.add_argument("--dataset", type=str, default="data.csv", help="Path to raw dataset.")
    parser.add_argument("--train-ratio", type=float, default=0.8, help="Ratio for training set.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility.")
    parser.add_argument("--train-out", type=str, default="data_train.csv", help="Path for output training CSV.")
    parser.add_argument("--val-out", type=str, default="data_valid.csv", help="Path for output validation CSV.")
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
