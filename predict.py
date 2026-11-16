#!/usr/bin/env python3
"""
predict.py - Multilayer Perceptron prediction and evaluation program.
Subject IV.4: "A prediction program"
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions and evaluate with Binary Cross-Entropy.")
    parser.add_argument("--dataset", type=str, default="dataset/data_valid.csv", help="Path to evaluation dataset.")
    parser.add_argument("--model", type=str, default="./saved_model.npy", help="Path to saved model file.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save predictions.")
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
