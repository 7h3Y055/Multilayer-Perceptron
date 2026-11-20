# Multilayer Perceptron (MLP) — 42 School Machine Learning

A complete, production-grade Artificial Neural Network implemented **from scratch** in Python using only `NumPy` for matrix mathematics and `Matplotlib` for visualization. Built to predict whether a breast cancer tumor is **Malignant (M)** or **Benign (B)** based on the Wisconsin Diagnostic Breast Cancer (WDBC) dataset.

---

## Table of Contents

1. [Project Overview & 42 Constraints](#1-project-overview--42-constraints)
2. [Project File Structure](#2-project-file-structure)
3. [Quickstart & Evaluation Commands](#3-quickstart--evaluation-commands)
4. [Theoretical Deep-Dive (Whiteboard Defense Guide)](#4-theoretical-deep-dive-whiteboard-defense-guide)
   - [A. Feedforward Propagation & Matrix Dimensions](#a-feedforward-propagation--matrix-dimensions)
   - [B. Activation Functions](#b-activation-functions)
   - [C. Backpropagation Derivation (The Chain Rule)](#c-backpropagation-derivation-the-chain-rule)
   - [D. Softmax + Cross-Entropy Derivative Proof ($A - Y$)](#d-softmax--cross-entropy-derivative-proof-a---y)
   - [E. Why Feature Normalization is Mandatory (The 25,000x Gap)](#e-why-feature-normalization-is-mandatory-the-25000x-gap)
   - [F. Weight Initializations (He Uniform vs. Zeros)](#f-weight-initializations-he-uniform-vs-zeros)
   - [G. Optimizers: SGD vs. Adam](#g-optimizers-sgd-vs-adam)
   - [H. Learning Curves: Diagnosing Overfitting & Early Stopping](#h-learning-curves-diagnosing-overfitting--early-stopping)
5. [Bonus Features Implemented](#5-bonus-features-implemented)
6. [Defense FAQ: Rapid-Fire Questions & Answers](#6-defense-faq-rapid-fire-questions--answers)

---

## 1. Project Overview & 42 Constraints

* **Dataset:** 569 samples with 30 continuous cell nucleus features (radius, texture, perimeter, area, concavity, etc.).
* **Target:** Predict diagnosis (`M` = Malignant $\to 1$, `B` = Benign $\to 0$).
* **Subject Constraints:**
  * **No ML/Neural Network frameworks allowed** (no PyTorch, TensorFlow, Keras, scikit-learn).
  * Only linear algebra (`NumPy`) and plotting (`Matplotlib`) are permitted.
  * Must contain **at least two hidden layers** by default (e.g. `30 -> 24 -> 24 -> 2`).
  * Output layer must use **Softmax** to output calibrated probabilities.
  * Must submit 3 programs: data split, training, prediction (or 1 unified program with options).
  * Evaluated on understanding of **feedforward**, **backpropagation**, and **gradient descent**.

---

## 2. Project File Structure

```text
Multilayer-Perceptron/
├── data.csv                 # Raw Wisconsin Breast Cancer dataset (569 rows)
├── requirements.txt         # NumPy and Matplotlib
│
├── eda.py                   # Exploratory Data Analysis & visualizer (Subject IV.2)
├── split.py                 # Program 1: Shuffles and splits dataset with seed
├── train.py                 # Program 2: Trains network, logs epochs, plots curves, saves model
├── predict.py               # Program 3: Loads model, runs inference, evaluates with BCE
├── compare.py               # Bonus: Trains & compares multiple optimizers on one graph
├── mlp.py                   # Unified CLI program switching between all phases
│
├── dataset/                 # Automatically created by split.py
│   ├── data_train.csv       # 455 training samples (80%)
│   └── data_valid.csv       # 114 validation samples (20%)
│
└── src/
    ├── __init__.py          # Package initializer
    ├── dataset.py           # Data loader, one-hot encoder, and Z-score StandardScaler
    ├── layers.py            # DenseLayer (He/Xavier initializers, Sigmoid, ReLU, Softmax)
    ├── losses.py            # Categorical Cross-Entropy, Binary Cross-Entropy, Metrics
    ├── optimizers.py        # SGD (Mandatory) and Adam (Bonus)
    ├── network.py           # NeuralNetwork container (forward, backward, fit, save, load)
    └── visualizer.py        # Matplotlib learning curves & multi-model comparison
```

---

## 3. Quickstart & Evaluation Commands

### Step 0: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 1: Exploratory Data Analysis (Subject IV.2)
Inspects class balance, scale discrepancies, feature distributions, and the correlation heatmap:
```bash
python3 eda.py --dataset data.csv --show
```

### Step 2: Split Dataset (Program 1)
Partitions `data.csv` into `dataset/data_train.csv` (80%) and `dataset/data_valid.csv` (20%) using random seed `42`:
```bash
python3 split.py data.csv
# Or with explicit flags:
python3 split.py --dataset data.csv --train-ratio 0.8 --seed 42
```

### Step 3: Train the Model (Program 2)
Trains an MLP with 2 hidden layers of 24 neurons using **SGD** (mandatory):
```bash
python3 train.py --layer 24 24 --epochs 70 --batch_size 8 --learning_rate 0.0314 --optimizer sgd
```
* Or train with the **Adam optimizer + Early Stopping** (bonus):
```bash
python3 train.py --layer 24 24 --epochs 80 --batch_size 8 --learning_rate 0.005 --optimizer adam --early_stopping --patience 10 --history training_history.json
```
*Generates `./saved_model.npy` and `learning_curves.png`.*

### Step 4: Run Predictions & Evaluate BCE (Program 3)
Loads the saved model and evaluates on the unseen validation set using the **Binary Cross-Entropy Error formula ($E$)**:
```bash
python3 predict.py --dataset dataset/data_valid.csv --model ./saved_model.npy
```

### Step 5: Compare Optimizers on One Graph (Bonus Chapter V)
Trains both SGD and Adam side-by-side and plots their validation loss curves on a single plot:
```bash
python3 compare.py
```
*Generates `models_comparison.png`.*

### Alternative: Unified CLI (`mlp.py`)
Matches the Subject IV.4 single-program option:
```bash
python3 mlp.py                  # Trains with default parameters
python3 mlp.py split            # Runs split phase
python3 mlp.py train [flags]    # Runs training phase
python3 mlp.py predict [flags]  # Runs prediction phase
python3 mlp.py eda              # Runs EDA phase
python3 mlp.py compare          # Runs comparison phase
```

---

## 4. Theoretical Deep-Dive (Whiteboard Defense Guide)

During your defense, the evaluator will ask you to derive equations and justify architectural choices on a whiteboard. Here is everything you need:

### A. Feedforward Propagation & Matrix Dimensions

Let:
* $m$ = batch size (e.g., 8)
* $n^{[l-1]}$ = number of input neurons to layer $l$
* $n^{[l]}$ = number of neurons in layer $l$

For any layer $l$:
$$Z^{[l]} = A^{[l-1]} \cdot W^{[l]} + b^{[l]}$$
$$A^{[l]} = f(Z^{[l]})$$

#### Matrix Dimensions (Show this on the whiteboard):
* **Inputs batch $X = A^{[0]}$:** $(m \times 30)$
* **Layer 1 Weights $W^{[1]}$:** $(30 \times 24)$
* **Layer 1 Biases $b^{[1]}$:** $(1 \times 24)$
* **Layer 1 Output $Z^{[1]}$:** $(m \times 30) \times (30 \times 24) + (1 \times 24) = (m \times 24)$
* **Layer 2 Weights $W^{[2]}$:** $(24 \times 24)$
* **Output Layer Weights $W^{[3]}$:** $(24 \times 2)$
* **Final Output $A^{[3]}$:** $(m \times 2)$ (Softmax probabilities)

**Parameter Count Formula:** $(\text{inputs} \times \text{neurons}) + \text{biases}$
* Layer 1: $(30 \times 24) + 24 = 744$
* Layer 2: $(24 \times 24) + 24 = 600$
* Output: $(24 \times 2) + 2 = 50$
* **Total parameters:** $744 + 600 + 50 = \mathbf{1,394}$ (with 2 hidden layers) or $\mathbf{1,994}$ (with 3 hidden layers).

---

### B. Activation Functions

1. **Sigmoid:**
   $$\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \sigma'(z) = \sigma(z)(1 - \sigma(z)) = A(1 - A)$$
   * Squashes values into $(0, 1)$.
   * Derivative is fast because $A$ is already cached during forward propagation.
   * Numerical safeguard: $z$ is clipped to $[-500, 500]$ to prevent exponential overflow.

2. **Softmax (Output Layer):**
   $$S(z_i) = \frac{e^{z_i - \max(z)}}{\sum_{j=1}^K e^{z_j - \max(z)}}$$
   * Turns raw logits into probabilities that sum strictly to $1.0$.
   * Subtracting $\max(z)$ makes the largest exponent $e^0 = 1$, eliminating floating-point overflow (`inf`).

---

### C. Backpropagation Derivation (The Chain Rule)

Backpropagation calculates how much each weight and bias contributed to the final loss.

1. **Output Layer Error ($\delta^{[L]}$):**
   $$\delta^{[L]} = \frac{\partial L}{\partial Z^{[L]}} = A^{[L]} - Y \quad (\text{shape: } m \times 2)$$

2. **Weight Gradient ($dW^{[l]}$):**
   By the chain rule: $\frac{\partial L}{\partial W^{[l]}} = \frac{\partial L}{\partial Z^{[l]}} \cdot \frac{\partial Z^{[l]}}{\partial W^{[l]}}$
   $$dW^{[l]} = \frac{1}{m} (A^{[l-1]})^T \cdot \delta^{[l]} \quad (\text{shape: } n^{[l-1]} \times n^{[l]})$$

3. **Bias Gradient ($db^{[l]}$):**
   $$db^{[l]} = \frac{1}{m} \sum_{i=1}^m \delta_i^{[l]} \quad (\text{shape: } 1 \times n^{[l]})$$

4. **Passing Error to Previous Layer ($\delta^{[l-1]}$):**
   $$\delta^{[l-1]} = \left( \delta^{[l]} \cdot (W^{[l]})^T \right) \odot f'(Z^{[l-1]}) \quad (\text{shape: } m \times n^{[l-1]})$$
   * Where $\odot$ denotes element-wise (Hadamard) product.

---

### D. Softmax + Cross-Entropy Derivative Proof ($A - Y$)

Evaluators love to ask: *"Why does the gradient at the output layer simplify to $\hat{y} - y$?"*

**Proof:**
1. Categorical Cross-Entropy: $L = -\sum_k y_k \ln(a_k)$
2. Softmax: $a_i = \frac{e^{z_i}}{\sum_j e^{z_j}}$
3. By the chain rule: $\frac{\partial L}{\partial z_i} = \sum_k \frac{\partial L}{\partial a_k} \frac{\partial a_k}{\partial z_i}$
4. Derivative of Softmax:
   * When $k = i$: $\frac{\partial a_i}{\partial z_i} = a_i(1 - a_i)$
   * When $k \ne i$: $\frac{\partial a_k}{\partial z_i} = -a_k a_i$
5. Substitute into the sum:
   $$\frac{\partial L}{\partial z_i} = -\frac{y_i}{a_i} \cdot a_i(1 - a_i) - \sum_{k \ne i} \frac{y_k}{a_k} (-a_k a_i)$$
   $$\frac{\partial L}{\partial z_i} = -y_i(1 - a_i) + \sum_{k \ne i} y_k a_i$$
   $$\frac{\partial L}{\partial z_i} = -y_i + y_i a_i + a_i \sum_{k \ne i} y_k$$
   $$\frac{\partial L}{\partial z_i} = -y_i + a_i \underbrace{\left( y_i + \sum_{k \ne i} y_k \right)}_{= 1 \text{ (one-hot labels sum to 1)}} = a_i - y_i$$
$$\mathbf{\frac{\partial L}{\partial Z} = A - Y} \quad \blacksquare$$

---

### E. Why Feature Normalization is Mandatory (The 25,000x Gap)

Looking at raw `data.csv`:
* `area_mean`: min $143.5$, max **$2501.0$** (mean $\approx 655$)
* `smoothness_mean`: min $0.05$, max **$0.16$** (mean $\approx 0.10$)

**The Problem:**
Since $\frac{\partial L}{\partial W} = \frac{1}{m} X^T \delta$, the gradient is directly proportional to input value $X$.
* The weight gradient for `area` is **$25,000\times$ larger** than for `smoothness`.
* The loss surface becomes an elongated, pathological canyon. Gradient descent oscillates violently in the steep direction and stalls in the flat direction.

**The Solution:**
Z-score standardization:
$$X_{\text{norm}} = \frac{X - \mu}{\sigma}$$
* Centers all features at mean $\mu = 0$ with variance $\sigma = 1$.
* **Data Leakage Rule:** We fit $\mu, \sigma$ **only on the training split**, and apply those exact parameters to validation/test sets.

---

### F. Weight Initializations (He Uniform vs. Zeros)

1. **Why not Zeros?**
   If all weights are initialized to 0, every neuron in a layer computes $Z = 0 \cdot X + 0 = 0$, producing identical activations and identical gradients. All neurons update identically, breaking **symmetry** and preventing the network from learning different features.
2. **He (Kaiming) Uniform (`heUniform`):**
   * Formula: Uniform distribution in $[-\text{limit}, +\text{limit}]$ where $\text{limit} = \sqrt{\frac{6}{n_{\text{in}}}}$.
   * Specifically derived to keep signal variance constant across layers when using non-linear activations.

---

### G. Optimizers: SGD vs. Adam

1. **SGD (Stochastic / Mini-Batch Gradient Descent):**
   $$W \leftarrow W - \alpha \cdot dW, \quad b \leftarrow b - \alpha \cdot db$$
   * Simple and effective, but uses a fixed global learning rate $\alpha$ for all parameters.

2. **Adam (Adaptive Moment Estimation):**
   Combines **Momentum** (first moment $m$) and **RMSprop** (second moment $v$):
   * 1st moment (velocity): $m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t$
   * 2nd moment (scale): $v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2$
   * Bias correction: $\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}$
   * Parameter update: $\theta \leftarrow \theta - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t$
   * Automatically gives small updates to frequently updated weights and large updates to infrequent features.

---

### H. Learning Curves: Diagnosing Overfitting & Early Stopping

Subject IV.3 requires displaying two graphs:

```text
       Figure IV.1: Loss                      Figure IV.2: Accuracy
 Loss                                   Acc
  │ ─── Training Loss                    │ ─── Training Acc
  │ - - Validation Loss                  │ ─── Validation Acc
  │                                      │       /───────
  │  \                                   │      / 
  │   \────────                          │  ───/
  └───────────────── Epochs              └───────────────── Epochs
```

* **Healthy Training:** Both training and validation loss smoothly decrease together and plateau.
* **Overfitting:** Training loss continues to drop toward $0$, but validation loss starts curving upwards. The model is memorizing training noise.
* **Underfitting:** Both training and validation loss remain high and refuse to drop (learning rate too low or model too small).
* **Early Stopping:** Monitors validation loss; if it does not improve for $P$ epochs (patience), training halts and restores the best saved weights.

---

## 5. Bonus Features Implemented

All 5 bonuses suggested in **Subject Chapter V** are fully functional:

1. **Complex Optimization Function:** Adam optimizer with momentum and bias correction (`--optimizer adam`).
2. **Multiple Learning Curves on Same Graph:** [compare.py](file:///Users/youssef/Documents/Multilayer-Perceptron/compare.py) compares SGD vs. Adam on a single figure (`models_comparison.png`).
3. **Training Metrics History:** Saves history inside the model and allows JSON export (`--history training_history.json`).
4. **Early Stopping:** Halts when validation loss plateaus and restores best epoch weights (`--early_stopping --patience 10`).
5. **Multiple Evaluation Metrics:** Outputs Confusion Matrix, Precision, Recall, Specificity, and F1-Score in [predict.py](file:///Users/youssef/Documents/Multilayer-Perceptron/predict.py).

---

## 6. Defense FAQ: Rapid-Fire Questions & Answers

| Question | Short Defense Answer |
| :--- | :--- |
| **"Why is output 2 neurons with Softmax instead of 1 with Sigmoid?"** | Subject IV.3 states: *"You must also implement the softmax function on the output layer in order to obtain the output as a probabilistic distribution."* Softmax outputs $[P(\text{Benign}), P(\text{Malignant})]$ where $\sum P = 1$. |
| **"Why do we evaluate with BCE if we trained with Categorical CE?"** | Training with Softmax naturally couples with Categorical CE ($\hat{y} - y$). At evaluation, Subject IV.4 explicitly requires reporting the Binary Cross-Entropy formula: $E = -\frac{1}{N} \sum [y \ln p + (1-y)\ln(1-p)]$ using the positive class probability $p$. |
| **"Did you balance the 63% / 37% dataset?"** | No, because $1.7:1$ is a mild imbalance. Over-sampling tabular rows duplicates exact points causing overfitting, and under-sampling deletes 25% of medical records. With Z-score normalization, the model naturally achieves $>97\%$ accuracy without artificial distortion. |
| **"How do you prevent data leakage?"** | By fitting `StandardScaler` mean and std **strictly on `data_train.csv`**, and applying those same statistics to transform `data_valid.csv`. |
| **"What happens if learning rate is too large?"** | The parameter updates overshoot the minimum, causing loss to oscillate or diverge to infinity (`NaN`). |
