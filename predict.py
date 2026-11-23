#!/usr/bin/env python3
"""
predict.py - Multilayer Perceptron prediction and evaluation program.
Subject IV.4: "A prediction program"
"""

import argparse
import os
import sys
import numpy as np

from src.dataset import load_dataset
from src.network import NeuralNetwork
from src.losses import binary_crossentropy, compute_classification_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions and evaluate with Binary Cross-Entropy.")
    parser.add_argument("pos_dataset", nargs="?", default=None, help="Path to evaluation dataset (positional).")
    parser.add_argument("--dataset", type=str, default=None, help="Path to evaluation dataset (flag).")
    parser.add_argument("--model", type=str, default="./saved_model.npy", help="Path to saved model file.")
    parser.add_argument("--output", type=str, default=None, help="Optional path to save predictions.")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Load trained model, architecture, and scaler
    try:
        network, scaler, meta = NeuralNetwork.load(args.model)
        print(f"> Loaded model from '{args.model}'")
    except Exception as e:
        print(f"Error loading model from '{args.model}': {e}", file=sys.stderr)
        sys.exit(1)

    # Resolve evaluation dataset path
    dataset_path = args.pos_dataset or args.dataset
    if dataset_path is None:
        if os.path.exists("data_test.csv"):
            dataset_path = "data_test.csv"
        elif os.path.exists("dataset/data_valid.csv"):
            dataset_path = "dataset/data_valid.csv"
        else:
            dataset_path = "data.csv"

    # 2. Load evaluation dataset
    try:
        X_raw, y_true = load_dataset(dataset_path)
    except Exception as e:
        print(f"Error loading dataset from '{dataset_path}': {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {X_raw.shape[0]} samples with {X_raw.shape[1]} features from '{dataset_path}'.")

    # 3. Standardize features using the saved scaler (prevents data leakage)
    if scaler is not None:
        X = scaler.transform(X_raw)
    else:
        print("Warning: No scaler found in model file. Using unnormalized features.", file=sys.stderr)
        X = X_raw

    # 4. Predict probabilities (Softmax output)
    probs = network.predict(X)
    p_malignant = probs[:, 1]
    predicted_classes = network.predict_classes(X)

    # 5. Evaluate using the exact Binary Cross-Entropy formula from Subject IV.4
    bce_loss = binary_crossentropy(p_malignant, y_true)
    metrics = compute_classification_metrics(probs, y_true)

    print("\n" + "=" * 50)
    print("              EVALUATION RESULTS")
    print("=" * 50)
    print(f"Binary Cross-Entropy Error (E) : {bce_loss:.6f}")
    print(f"Accuracy                       : {metrics['accuracy'] * 100:.2f}%")
    print(f"Precision                      : {metrics['precision'] * 100:.2f}%")
    print(f"Recall (Sensitivity)           : {metrics['recall'] * 100:.2f}%")
    print(f"F1-Score                       : {metrics['f1_score'] * 100:.2f}%")
    print(f"Specificity                    : {metrics['specificity'] * 100:.2f}%")
    print("-" * 50)
    print(f"Confusion Matrix: TP={metrics['tp']}, TN={metrics['tn']}, FP={metrics['fp']}, FN={metrics['fn']}")
    print("=" * 50 + "\n")

    # 6. Optional CSV export
    if args.output:
        import csv
        with open(args.output, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Sample_Index", "True_Diagnosis", "Prob_Benign", "Prob_Malignant", "Predicted_Diagnosis"])
            for idx in range(len(y_true)):
                true_label = "M" if y_true[idx][0] == 1 else "B"
                pred_label = "M" if predicted_classes[idx] == 1 else "B"
                writer.writerow([
                    idx,
                    true_label,
                    f"{probs[idx, 0]:.6f}",
                    f"{probs[idx, 1]:.6f}",
                    pred_label,
                ])
        print(f"> Predictions saved to '{args.output}'")


if __name__ == "__main__":
    main()
