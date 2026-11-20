#!/usr/bin/env python3
"""
mlp.py - Unified entry point for Multilayer Perceptron.
Subject IV.4: "(or you can submit a single program with an option to switch between the three phases)"
"""

import sys

import split
import train
import predict
import eda


def print_help():
    print("""
Multilayer Perceptron (42 School Machine Learning Project)

Usage:
  1. Direct training (matches subject example):
     python3 mlp.py --dataset dataset/data_train.csv --val-dataset dataset/data_valid.csv

  2. Or switch between phases:
     python3 mlp.py split   [options]
     python3 mlp.py train   [options]
     python3 mlp.py predict [options]
     python3 mlp.py eda     [options]

For help on a specific phase:
  python3 mlp.py <phase> --help
""")


def main():
    if len(sys.argv) > 1 and sys.argv[1] in ("-h", "--help") and len(sys.argv) == 2:
        print_help()
        sys.exit(0)

    if len(sys.argv) > 1 and sys.argv[1] == "split":
        sys.argv.pop(1)
        split.main()
    elif len(sys.argv) > 1 and sys.argv[1] == "predict":
        sys.argv.pop(1)
        predict.main()
    elif len(sys.argv) > 1 and sys.argv[1] == "eda":
        sys.argv.pop(1)
        eda.main()
    elif len(sys.argv) > 1 and sys.argv[1] == "train":
        sys.argv.pop(1)
        train.main()
    else:
        # Default behavior matches Subject page 10 example:
        # python mlp.py --dataset data_training.csv
        train.main()


if __name__ == "__main__":
    main()
