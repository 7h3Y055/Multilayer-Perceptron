# Multilayer Perceptron (MLP) — 42 Project Study Plan & Guide

An artificial neural network built completely from scratch in Python (using only `NumPy` for linear algebra and `Matplotlib` for visualization) to predict breast cancer diagnoses (Malignant vs. Benign) from the Wisconsin Diagnostic Breast Cancer dataset.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Subject Requirements & Constraints](#subject-requirements--constraints)
- [Study & Implementation Roadmap](#study--implementation-roadmap)
  - [Phase 1: Mathematical Foundations](#phase-1-mathematical-foundations)
  - [Phase 2: Data Preprocessing & Split](#phase-2-data-preprocessing--split)
  - [Phase 3: Modular Architecture Design](#phase-3-modular-architecture-design)
  - [Phase 4: Backpropagation & Training Loop](#phase-4-backpropagation--training-loop)
  - [Phase 5: Evaluation, Prediction & Curves](#phase-5-evaluation-prediction--curves)
  - [Phase 6: Bonuses & Defense Preparation](#phase-6-bonuses--defense-preparation)
- [Mathematical Cheat Sheet (Whiteboard Defense)](#mathematical-cheat-sheet-whiteboard-defense)
- [Target Project Structure](#target-project-structure)
- [CLI Programs Specification](#cli-programs-specification)
- [Recommended Resources](#recommended-resources)

---

## Project Overview

The objective of this project is to implement a Multilayer Perceptron (feedforward artificial neural network) trained via backpropagation and gradient descent.

- **Dataset:** Wisconsin Breast Cancer dataset (`data.csv`, 32 columns: ID, Diagnosis `M`/`B`, 30 continuous real-valued features).
- **Core Task:** Binary classification predicting whether a tumor sample is **Malignant (M)** or **Benign (B)**.
- **Rules:** No high-level ML frameworks permitted (no PyTorch, TensorFlow, Keras, scikit-learn neural network models). Everything must be implemented from first principles.

---

## Subject Requirements & Constraints

| Component | Requirement |
| :--- | :--- |
| **Allowed Libraries** | Only libraries for linear algebra (`NumPy`) and plotting (`Matplotlib`). |
| **Hidden Layers** | Must contain **at least 2 hidden layers** by default (e.g., input $\to 24 \to 24 \to$ output). |
| **Output Layer** | Must implement **Softmax** activation to output a valid probability distribution. |
| **Modularity** | Architecture must be configurable via CLI flags (e.g., `--layer 24 24 24 --epochs 84 --learning_rate 0.0314 --batch_size 8`) or a configuration file. |
| **Required Programs** | 1. **Data Splitter:** Splits dataset into training and validation sets (with optional seed).<br>2. **Training Program:** Displays metrics per epoch, shows learning curves, and saves model topology & weights.<br>3. **Prediction Program:** Loads model weights, executes inference, and evaluates Binary Cross-Entropy loss. |
| **Peer-Evaluation** | You must be able to explain and write out the mathematics of **feedforward**, **backpropagation**, and **gradient descent** on a whiteboard. |

---

## Study & Implementation Roadmap

```
Phase 1: Math Foundations ──► Phase 2: Data & Split ──► Phase 3: Network Architecture
                                                                  │
Phase 6: Defense & Bonuses ◄── Phase 5: Predict & Curves ◄── Phase 4: Backprop & Training
```

### Phase 1: Mathematical Foundations
*Goal: Understand the mathematical engine before writing network code.*

1. **Matrix Dimensions & Vectorization:**
   - Let $m$ be the batch size, $n^{[l-1]}$ input features, $n^{[l]}$ neurons in current layer.
   - Input batch: $X \in \mathbb{R}^{m \times n^{[0]}}$
   - Layer weights: $W^{[l]} \in \mathbb{R}^{n^{[l-1]} \times n^{[l]}}$
   - Layer bias: $b^{[l]} \in \mathbb{R}^{1 \times n^{[l]}}$
   - Affine transformation: $Z^{[l]} = A^{[l-1]} W^{[l]} + b^{[l]}$
   - Activation output: $A^{[l]} = f(Z^{[l]})$
2. **Activation Functions & Derivatives:**
   - **Sigmoid:** $\sigma(z) = \frac{1}{1 + e^{-z}}$, with $\sigma'(z) = \sigma(z)(1 - \sigma(z))$
   - **ReLU:** $f(z) = \max(0, z)$, with $f'(z) = \begin{cases} 1 & z > 0 \\ 0 & z \le 0 \end{cases}$
   - **Softmax:** $S(z_i) = \frac{e^{z_i - \max(z)}}{\sum_j e^{z_j - \max(z)}}$ (subtract max for numerical stability)
3. **Loss Functions:**
   - **Binary Cross-Entropy (BCE):**
     $$E = -\frac{1}{N} \sum_{n=1}^N [y_n \log p_n + (1 - y_n) \log(1 - p_n)]$$
   - **Categorical Cross-Entropy:**
     $$E = -\frac{1}{N} \sum_{n=1}^N \sum_{k=1}^K y_{nk} \log(p_{nk})$$
4. **Backpropagation Derivation:**
   - For Softmax + Cross-Entropy, the output layer gradient is:
     $$\delta^{[L]} = \frac{\partial L}{\partial Z^{[L]}} = A^{[L]} - Y$$
   - Gradients for any layer $l$:
     $$dW^{[l]} = \frac{1}{m} (A^{[l-1]})^T \delta^{[l]}$$
     $$db^{[l]} = \frac{1}{m} \sum_{i=1}^m \delta_i^{[l]}$$
     $$\delta^{[l-1]} = (\delta^{[l]} (W^{[l]})^T) \odot f'(Z^{[l-1]})$$

---

### Phase 2: Data Exploration, Preprocessing & Split
*Goal: Explore and understand `data.csv`, implement clean scaling, and create `split.py`.*

1. **Dataset Structure (`data.csv`):**
   - Column 0: Patient ID (non-predictive identifier, drop during preprocessing).
   - Column 1: Diagnosis (`M` = Malignant $\to [1, 0]$ or $1$; `B` = Benign $\to [0, 1]$ or $0$).
   - Columns 2–31: 30 continuous features (radius, texture, perimeter, area, smoothness, etc.).
2. **Exploratory Data Analysis (EDA):**
   > The subject states: *"A good practice would be to begin by exploring the dataset, displaying it with graphs, visualizing, and manipulating its different features."*
   - Check for missing values and class balance (count of M vs B).
   - Plot histograms or box plots for each feature to understand value distributions.
   - Use a correlation heatmap to identify highly correlated features.
   - Use pair plots or scatter plots colored by diagnosis to visualize feature separability.
   - Identify which features have vastly different scales (motivates normalization).
2. **Feature Normalization (Z-Score Standardization):**
   $$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
   - **Crucial Rule:** Compute $\mu$ and $\sigma$ strictly on the training set, then apply them to the validation set to prevent data leakage.
3. **Implementation — `split.py`:**
   - Shuffle dataset randomly (with reproducible `--seed`).
   - Partition into training set (e.g., 80%) and validation set (e.g., 20%).
   - Export `data_train.csv` and `data_valid.csv`.

---

### Phase 3: Modular Architecture Design
*Goal: Build an object-oriented, layer-based neural network framework.*

1. **Weight Initializations:**
   - Avoid zero-initialization (breaks symmetry).
   - **He (Kaiming) Uniform/Normal:** Optimal for ReLU ($\text{std} = \sqrt{2 / n_{in}}$).
   - **Xavier (Glorot) Uniform/Normal:** Optimal for Sigmoid/Softmax ($\text{std} = \sqrt{2 / (n_{in} + n_{out})}$).
2. **Layer Abstractions:**
   - `DenseLayer`: Manages $W$, $b$, forward pass $Z = XW + b$, and backward pass computing $dW, db$.
   - `ActivationLayer`: Computes forward non-linear mapping and backward element-wise derivative.

---

### Phase 4: Backpropagation & Training Loop
*Goal: Implement batching, the backward pass, and weight updates.*

1. **Mini-Batch Gradient Descent:**
   - Shuffle training indices each epoch.
   - Iterate through mini-batches of size `batch_size`.
   - Update parameters:
     $$W \leftarrow W - \alpha \cdot dW, \quad b \leftarrow b - \alpha \cdot db$$
2. **Epoch Tracking:**
   - Compute training loss and validation loss at the end of each epoch.
   - Format terminal output matching the subject:
     ```text
     epoch 01/70 - loss: 0.6882 - val_loss: 0.6788
     ...
     epoch 70/70 - loss: 0.0640 - val_loss: 0.0474
     > saving model './saved_model.npy' to disk...
     ```

---

### Phase 5: Evaluation, Prediction & Curves
*Goal: Build `train.py`, `predict.py`, and visual learning curves.*

1. **Model Persistence:**
   - Serialize layer topology, learned weights, biases, and normalization scaling factors into a `.npy`, `.npz`, or `.json` file.
2. **Visual Learning Curves:**
   - Display two graphs upon training completion:
     - **Loss Curve:** Training loss and validation loss vs epochs.
     - **Accuracy Curve:** Training accuracy and validation accuracy vs epochs.
3. **Implementation — `predict.py`:**
   - Load saved model and weights.
   - Standardize input test data using saved $\mu, \sigma$.
   - Perform forward pass to compute class probabilities.
   - Output predictions and evaluate with Binary Cross-Entropy loss.

---

### Phase 6: Bonuses & Defense Preparation
*Goal: Implement all 5 subject bonuses and master evaluation defense questions.*

> The bonus part will only be assessed if the mandatory part is **PERFECT**.
> Make sure all mandatory requirements work flawlessly before starting bonuses.

#### Bonus 1: Advanced Optimizers
*Subject: "A more complex optimization function (for example: Nesterov momentum, RMSprop, Adam, ...)"*

Implement at least one (ideally all three) as selectable options via CLI (e.g., `--optimizer adam`):

- **Nesterov Momentum** (specifically named in the subject):
  - Look-ahead gradient: compute gradient at the anticipated next position.
  - $v_t = \gamma v_{t-1} + \alpha \nabla_\theta L(\theta - \gamma v_{t-1})$
  - $\theta = \theta - v_t$
  - Typical $\gamma = 0.9$.
- **RMSprop:**
  - Adapts learning rate per-parameter using a moving average of squared gradients.
  - $s_t = \beta s_{t-1} + (1 - \beta) (\nabla_\theta L)^2$
  - $\theta = \theta - \frac{\alpha}{\sqrt{s_t + \epsilon}} \nabla_\theta L$
  - Typical $\beta = 0.9$, $\epsilon = 10^{-8}$.
- **Adam** (Adaptive Moment Estimation):
  - Combines momentum (first moment) and RMSprop (second moment) with bias correction.
  - $m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L$ (first moment)
  - $v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L)^2$ (second moment)
  - Bias-corrected: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}$, $\hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
  - $\theta = \theta - \frac{\alpha \hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}$
  - Typical $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$.

#### Bonus 2: Multiple Learning Curves on the Same Graph
*Subject: "A display of multiple learning curves on the same graph (really useful to compare different models)"*

- Train multiple models with different hyperparameters (e.g., different learning rates, optimizers, or architectures).
- Plot all their loss curves (or accuracy curves) overlaid on a single Matplotlib figure with a legend identifying each configuration.
- Example: compare SGD vs Adam vs RMSprop convergence on the same axes.

#### Bonus 3: Metrics History
*Subject: "A history of the metrics obtained during training"*

- At each epoch, record all computed metrics (loss, accuracy, val_loss, val_accuracy, and any bonus metrics) into a structured history object (e.g., a dictionary of lists).
- Save this history to disk alongside the model (e.g., `training_history.json` or `.csv`).
- Allow re-loading and re-plotting the history without re-training.
- Example structure:
  ```json
  {
    "epoch": [1, 2, 3, ...],
    "loss": [0.6882, 0.5431, ...],
    "val_loss": [0.6788, 0.5201, ...],
    "accuracy": [0.55, 0.72, ...],
    "val_accuracy": [0.58, 0.74, ...]
  }
  ```

#### Bonus 4: Early Stopping
*Subject: "The implementation of early stopping"*

- Monitor validation loss at each epoch.
- If validation loss does not improve for $P$ consecutive epochs (patience), stop training early.
- Restore the model weights from the best epoch (lowest validation loss).
- Add CLI flag: `--early_stopping --patience 10`.
- Print a message when triggered:
  ```text
  Early stopping at epoch 45/200 (patience=10). Restoring best weights from epoch 35.
  ```

#### Bonus 5: Multiple Evaluation Metrics
*Subject: "Evaluate the learning phase with multiple metrics"*

- Beyond loss and accuracy, compute and display during training or at prediction time:
  - **Confusion Matrix:** True Positives, True Negatives, False Positives, False Negatives.
  - **Precision:** $\frac{TP}{TP + FP}$
  - **Recall (Sensitivity):** $\frac{TP}{TP + FN}$
  - **F1-Score:** $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
  - **Specificity:** $\frac{TN}{TN + FP}$
- Display these in `predict.py` output and optionally during training epochs.

---

#### 42 Peer Defense Checklist
Prepare to explain and write on a whiteboard:

- [ ] Derive $\frac{\partial L}{\partial W}$ and $\frac{\partial L}{\partial b}$ step-by-step using the chain rule.
- [ ] Explain why feature scaling is indispensable for gradient descent.
- [ ] Explain vanishing and exploding gradients and how proper weight initialization resolves them.
- [ ] Explain how Softmax coupled with Cross-Entropy produces the clean error $\hat{y} - y$.
- [ ] Explain the difference between SGD, Nesterov Momentum, RMSprop, and Adam.
- [ ] Explain what early stopping prevents and how it relates to the bias-variance tradeoff.

---

## Target Project Structure

```text
Multilayer-Perceptron/
├── data.csv                 # Raw Wisconsin Breast Cancer dataset
├── dataset/
│   ├── data_train.csv       # Generated training set
│   └── data_valid.csv       # Generated validation set
├── saved_model.npy          # Saved weights, topology & scaler stats
├── split.py                 # Program 1: Dataset splitting
├── train.py                 # Program 2: Network training & plotting
├── predict.py               # Program 3: Inference & BCE loss evaluation
├── src/
│   ├── __init__.py
│   ├── dataset.py           # Parsing, label encoding, Z-score normalization
│   ├── layers.py            # DenseLayer, Activation layers
│   ├── network.py           # NeuralNetwork container (fit, predict, save/load)
│   ├── losses.py            # Binary & Categorical Cross-Entropy
│   ├── optimizers.py        # SGD, Momentum, Adam
│   └── visualizer.py        # Matplotlib learning curves
└── README.md
```

---

## CLI Programs Specification

### 1. Data Splitting (`split.py`)
```bash
python split.py --dataset data.csv --train-ratio 0.8 --seed 42
```
- Splits `data.csv` into `dataset/data_train.csv` and `dataset/data_valid.csv`.

### 2. Training (`train.py`)
```bash
python train.py --dataset dataset/data_train.csv --val-dataset dataset/data_valid.csv \
                --layer 24 24 24 \
                --epochs 84 \
                --batch_size 8 \
                --learning_rate 0.0314 \
                --save saved_model.npy
```
- Prints epoch losses, renders learning curves, and saves model parameters.

### 3. Prediction (`predict.py`)
```bash
python predict.py --dataset dataset/data_valid.csv --model saved_model.npy
```
- Loads the trained model, performs inference, outputs predictions, and calculates final Binary Cross-Entropy loss.

---

## Recommended Resources

- **3Blue1Brown:** *Neural Networks series* (Visual intuition on forward pass, gradient descent, and backprop)
- **Andrew Ng (DeepLearning.AI):** *Neural Networks and Deep Learning* (Course 1 on Coursera / YouTube)
- **Michael Nielsen:** *Neural Networks and Deep Learning* (Free online text with mathematical derivations)
- **CS231n (Stanford):** *Neural Networks Architecture and Optimization notes*
