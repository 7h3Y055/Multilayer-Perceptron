#!/usr/bin/env python3
"""
eda.py - Exploratory Data Analysis.
Subject IV.2: "begin by exploring the dataset, displaying it with graphs, visualizing, and manipulating its different features."
"""

import argparse


def parse_args():
    parser = argparse.ArgumentParser(description="Perform Exploratory Data Analysis on dataset.")
    parser.add_argument("--dataset", type=str, default="data.csv", help="Path to raw dataset.")
    parser.add_argument("--plot", type=str, default="eda_summary.png", help="Path to save EDA graph.")
    parser.add_argument("--show", action="store_true", help="Display plot interactively.")
    return parser.parse_args()


def main():
    pass


if __name__ == "__main__":
    main()
