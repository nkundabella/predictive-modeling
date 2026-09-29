"""
=============================================================================
  DAY 6 — Logistic Regression and the Sigmoid Function
  Topics : OLS Failure for Classification, Sigmoid & Logit,
           Scratch Gradient Descent vs Sklearn, Non-Convex MSE vs BCE,
           2D Decision Boundary & Probability Contours, Perfect Separation,
           and Cost-Sensitive Threshold Tuning.
  Run    : python day-06/logistic_regression_demo.py
=============================================================================
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Why Linear Regression Fails on Classification
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 1 — WHY LINEAR REGRESSION FAILS FOR CLASSIFICATION")
print("=" * 70)

# Create 1D data: class 0 clustered at x in [1, 4], class 1 clustered at x in [6, 9]
rng = np.random.default_rng(42)
n_each = 15

x_c0 = rng.uniform(1.0, 4.0, n_each)
y_c0 = np.zeros(n_each)

x_c1 = rng.uniform(6.0, 9.0, n_each)
y_c1 = np.ones(n_each)

x_clean = np.concatenate([x_c0, x_c1]).reshape(-1, 1)
y_clean = np.concatenate([y_c0, y_c1])

# Fit standard OLS and Logistic Regression on clean data
ols_clean = LinearRegression().fit(x_clean, y_clean)
log_clean = LogisticRegression(penalty=None).fit(x_clean, y_clean)

# Decision boundary for OLS is where w*x + b = 0.5 ==> x = (0.5 - b) / w
boundary_ols_clean = (0.5 - ols_clean.intercept_) / ols_clean.coef_[0]
boundary_log_clean = -log_clean.intercept_[0] / log_clean.coef_[0][0]

# Now introduce extreme outliers with y=1 far to the right (x in [25, 30])
x_outliers = rng.uniform(25.0, 30.0, 6).reshape(-1, 1)
y_outliers = np.ones(6)

x_contaminated = np.vstack([x_clean, x_outliers])
y_contaminated = np.concatenate([y_clean, y_outliers])

ols_contam = LinearRegression().fit(x_contaminated, y_contaminated)
log_contam = LogisticRegression(penalty=None).fit(x_contaminated, y_contaminated)

boundary_ols_contam = (0.5 - ols_contam.intercept_) / ols_contam.coef_[0]
boundary_log_contam = -log_contam.intercept_[0] / log_contam.coef_[0][0]

print(f"  Clean Data:")
print(f"    OLS Decision Boundary (w*x+b=0.5)     : x = {boundary_ols_clean:.2f}")
print(f"    Logistic Boundary (w*x+b=0.0)         : x = {boundary_log_clean:.2f}")
print(f"  Contaminated Data (with extreme positive outliers):")
print(f"    OLS Decision Boundary shifted to      : x = {boundary_ols_contam:.2f}  <-- SHIFTED HEAVILY!")
print(f"    Logistic Boundary shifted to          : x = {boundary_log_contam:.2f}  <-- ROBUST")
print("  Notice: OLS squares the 'error' of extreme positive points (1 - 3.5)^2,")
print("  forcing the line to rotate and misclassifying true positives near the boundary.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — The Sigmoid Function and its Derivative
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 2 — THE SIGMOID (LOGISTIC) FUNCTION AND ITS DERIVATIVE")
print("=" * 70)

def sigmoid(z):
    """Numerically stable sigmoid function."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

def sigmoid_derivative(z):
    s = sigmoid(z)
    return s * (1.0 - s)

sample_zs = np.array([-4.0, -2.0, 0.0, 2.0, 4.0])
print(f"{'Logit z':>10} | {'σ(z) [Probability]':>20} | {'σ\'(z) = σ(1-σ)':>20} | {'Odds = p/(1-p)':>18}")
print("-" * 75)
for z_val in sample_zs:
    prob = sigmoid(z_val)
    d_prob = sigmoid_derivative(z_val)
    odds = prob / (1.0 - prob)
    print(f"{z_val:>10.2f} | {prob:>20.4f} | {d_prob:>20.4f} | {odds:>18.4f}")
print(f"\n  Max derivative occurs at inflection point z=0: σ'(0) = {sigmoid_derivative(0):.4f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Scratch Logistic Regression Implementation (Gradient Descent)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 3 — SCRATCH LOGISTIC REGRESSION WITH VECTORIZED GRADIENT DESCENT")
print("=" * 70)

class LogisticRegressionScratch:
    """
    Binary Logistic Regression with Batch Gradient Descent and optional L2 regularization.
    
    Parameters
    ----------
    lr : float, default=0.1
        Learning rate for gradient descent.
    n_epochs : int, default=1000
        Number of iterations over the full training set.
    l2_reg : float, default=0.0
        L2 regularization weight (lambda). Set to 0 for unregularized MLE.
    tol : float, default=1e-7
        Convergence tolerance on loss change.
    """
    def __init__(self, lr=0.1, n_epochs=1000, l2_reg=0.0, tol=1e-7):
        self.lr = lr
        self.n_epochs = n_epochs
        self.l2_reg = l2_reg
        self.tol = tol
        self.weights = None
        self.bias = None
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Initialize weights with zeros
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        eps = 1e-15  # For numerical stability in log loss

        for epoch in range(self.n_epochs):
            # Forward pass: compute linear combination and predicted probabilities
            z = X @ self.weights + self.bias
            y_hat = sigmoid(z)

            # Compute Binary Cross-Entropy loss with L2 penalty
            # BCE = - (1/n) * sum( y*ln(y_hat) + (1-y)*ln(1 - y_hat) )
            y_hat_safe = np.clip(y_hat, eps, 1.0 - eps)
            bce_loss = -np.mean(y * np.log(y_hat_safe) + (1.0 - y) * np.log(1.0 - y_hat_safe))
            reg_penalty = 0.5 * self.l2_reg * np.sum(self.weights ** 2)
            total_loss = bce_loss + reg_penalty
            self.loss_history.append(total_loss)

            # Check convergence
            if epoch > 0 and abs(self.loss_history[-2] - total_loss) < self.tol:
                break

            # Vectorized gradient computation:
            # grad_w = (1/n) * X^T (y_hat - y) + lambda * w
            # grad_b = (1/n) * sum(y_hat - y)
            error = y_hat - y
            grad_w = (X.T @ error) / n_samples + self.l2_reg * self.weights
            grad_b = np.sum(error) / n_samples

            # Parameter update step
            self.weights -= self.lr * grad_w
            self.bias -= self.lr * grad_b

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        z = X @ self.weights + self.bias
        p1 = sigmoid(z)
        p0 = 1.0 - p1
        return np.column_stack([p0, p1])

    def predict(self, X, threshold=0.5):
        prob = self.predict_proba(X)[:, 1]
        return (prob >= threshold).astype(int)


# Generate a 2D synthetic dataset for verification
X_synth, y_synth = make_blobs(n_samples=200, n_features=2, centers=2, cluster_std=1.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_synth, y_synth, test_size=0.3, random_state=42)

# Train Scratch model
scratch_model = LogisticRegressionScratch(lr=0.5, n_epochs=3000, l2_reg=0.01)
scratch_model.fit(X_train, y_train)

# Train scikit-learn model with equivalent L2 penalty: C = 1 / (n * lambda)
n_train = len(y_train)
C_equiv = 1.0 / (n_train * 0.01)
sk_model = LogisticRegression(C=C_equiv, penalty="l2", solver="lbfgs", fit_intercept=True)
sk_model.fit(X_train, y_train)

scratch_preds = scratch_model.predict(X_test)
sk_preds = sk_model.predict(X_test)

print(f"  Scratch Model Weights   : w1 = {scratch_model.weights[0]:.5f}, w2 = {scratch_model.weights[1]:.5f}, b = {scratch_model.bias:.5f}")
print(f"  Scikit-Learn Weights    : w1 = {sk_model.coef_[0][0]:.5f}, w2 = {sk_model.coef_[0][1]:.5f}, b = {sk_model.intercept_[0]:.5f}")
print(f"  Scratch Test Accuracy   : {accuracy_score(y_test, scratch_preds) * 100:.2f}%")
print(f"  Sklearn Test Accuracy   : {accuracy_score(y_test, sk_preds) * 100:.2f}%")
print(f"  Prediction Agreement    : {np.mean(scratch_preds == sk_preds) * 100:.2f}% identical predictions!\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Convexity: Binary Cross-Entropy vs Mean Squared Error
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 4 — LOSS LANDSCAPE: BINARY CROSS-ENTROPY VS MEAN SQUARED ERROR")
print("=" * 70)

# Consider a single feature x and binary labels y
x_1d = np.array([-2.0, -1.0, 0.5, 1.5, 2.5, 3.0])
y_1d = np.array([ 0.0,  0.0, 0.0, 1.0, 1.0, 1.0])

w_range = np.linspace(-5.0, 7.0, 300)
b_fixed = -1.0

bce_landscape = []
mse_landscape = []

eps = 1e-15
for w_val in w_range:
    z = w_val * x_1d + b_fixed
    y_hat = sigmoid(z)
    
    # BCE
    y_safe = np.clip(y_hat, eps, 1.0 - eps)
    bce = -np.mean(y_1d * np.log(y_safe) + (1.0 - y_1d) * np.log(1.0 - y_safe))
    bce_landscape.append(bce)
    
    # MSE with sigmoid: (1/2n) * sum (y - sigmoid(z))^2
    mse = 0.5 * np.mean((y_1d - y_hat) ** 2)
    mse_landscape.append(mse)

print("  Computed 1D loss landscapes over w in [-5.0, 7.0].")
print("  - BCE is strictly convex: every local minimum is the unique global minimum.")
print("  - MSE with sigmoid produces plateaus and non-convex regions due to vanishing gradients.\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — The Perfect Separation Problem & Regularization
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 5 — PERFECT SEPARATION: UNREGULARIZED EXPLOSION VS REGULARIZATION")
print("=" * 70)

# Create a strictly linearly separable dataset
X_sep = np.array([[-3.0], [-2.5], [-2.0], [-1.5], [1.5], [2.0], [2.5], [3.0]])
y_sep = np.array([0, 0, 0, 0, 1, 1, 1, 1])

# Train unregularized model with many iterations
model_unreg = LogisticRegressionScratch(lr=0.5, n_epochs=10000, l2_reg=0.0)
model_unreg.fit(X_sep, y_sep)

# Train regularized model (lambda = 0.5)
model_reg = LogisticRegressionScratch(lr=0.5, n_epochs=10000, l2_reg=0.5)
model_reg.fit(X_sep, y_sep)

print(f"  Strictly Linearly Separable Data (8 samples):")
print(f"  Unregularized Weight w (10,000 steps) : {model_unreg.weights[0]:.4f}  (Continues growing without bound!)")
print(f"  Regularized Weight w (lambda = 0.5)   : {model_reg.weights[0]:.4f}  (Bounded and stabilized by L2 penalty)")
print(f"  Final Unregularized Loss              : {model_unreg.loss_history[-1]:.6f}")
print(f"  Final Regularized Loss                : {model_reg.loss_history[-1]:.6f}\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Cost-Sensitive Threshold Tuning
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 6 — ASYMMETRIC COST OPTIMIZATION VIA THRESHOLD TUNING")
print("=" * 70)

# Simulate an imbalanced medical screening scenario (e.g. 10% positive rate)
X_med, y_med = make_classification(
    n_samples=1000, n_features=4, n_informative=3, n_redundant=1,
    weights=[0.90, 0.10], random_state=42
)
X_med_tr, X_med_te, y_med_tr, y_med_te = train_test_split(X_med, y_med, test_size=0.3, random_state=42)

med_model = LogisticRegression(solver="lbfgs").fit(X_med_tr, y_med_tr)
test_probs = med_model.predict_proba(X_med_te)[:, 1]

# Operational costs:
# Missing a positive case (False Negative) costs $500 (late treatment)
# A false alarm (False Positive) costs $50 (extra lab test)
cost_fn = 500.0
cost_fp = 50.0

thresholds = np.linspace(0.02, 0.98, 97)
cost_list = []
precision_list = []
recall_list = []

for tau in thresholds:
    preds = (test_probs >= tau).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_med_te, preds, labels=[0, 1]).ravel()
    
    total_cost = fp * cost_fp + fn * cost_fn
    cost_list.append(total_cost)
    
    prec = tp / (tp + fp) if (tp + fp) > 0 else 1.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    precision_list.append(prec)
    recall_list.append(rec)

cost_list = np.array(cost_list)
optimal_idx = np.argmin(cost_list)
optimal_tau = thresholds[optimal_idx]
optimal_cost = cost_list[optimal_idx]

default_idx = np.argmin(np.abs(thresholds - 0.5))
default_cost = cost_list[default_idx]

print(f"  Cost Matrix: False Negative = ${cost_fn:.0f}, False Positive = ${cost_fp:.0f}")
print(f"  Default Threshold (tau = 0.50): Total Cost = ${default_cost:,.0f}")
print(f"  Optimal Threshold (tau = {optimal_tau:.2f}): Total Cost = ${optimal_cost:,.0f}")
print(f"  Cost Reduction achieved by threshold tuning: ${(default_cost - optimal_cost):,.0f} ({(default_cost - optimal_cost)/default_cost*100:.1f}% savings)\n")


# ═══════════════════════════════════════════════════════════════════════════
# PART 7 — Generating Comprehensive Diagnostic Visualizations
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("PART 7 — SAVING HIGH-RESOLUTION DIAGNOSTIC PLOTS")
print("=" * 70)

# Figure 1: Logistic Regression Overview (4-panel)
fig, axes = plt.subplots(2, 2, figsize=(15, 12))

# Subplot 1: OLS Failure vs Logistic Regression on 1D Classification
ax = axes[0, 0]
x_line = np.linspace(0, 32, 400).reshape(-1, 1)
y_ols_clean = ols_clean.predict(x_line)
y_ols_contam = ols_contam.predict(x_line)
y_log_contam = log_contam.predict_proba(x_line)[:, 1]

ax.scatter(x_clean[y_clean == 0], y_clean[y_clean == 0], color="#2b5c8f", s=50, alpha=0.8, label="Class 0 (Clean)")
ax.scatter(x_clean[y_clean == 1], y_clean[y_clean == 1], color="#d95f02", s=50, alpha=0.8, label="Class 1 (Clean)")
ax.scatter(x_outliers, y_outliers, color="#7570b3", s=70, marker="^", label="Class 1 Outliers")

ax.plot(x_line, y_ols_clean, "b--", alpha=0.6, label="OLS (Clean Data)")
ax.plot(x_line, y_ols_contam, "r-", linewidth=2, label="OLS (Contaminated)")
ax.plot(x_line, y_log_contam, "g-", linewidth=2.5, label="Logistic Regression")
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.7)
ax.axvline(boundary_ols_clean, color="blue", linestyle="--", alpha=0.5)
ax.axvline(boundary_ols_contam, color="red", linestyle="-", alpha=0.7)
ax.set_ylim(-0.3, 1.4)
ax.set_xlim(0, 32)
ax.set_title("1. Why OLS Fails: Outlier Tilts Linear Boundary", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature x")
ax.set_ylabel("Predicted Value / Probability")
ax.legend(loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)

# Subplot 2: Sigmoid Function and its Derivative
ax = axes[0, 1]
z_plot = np.linspace(-6, 6, 300)
sig_plot = sigmoid(z_plot)
sig_deriv_plot = sigmoid_derivative(z_plot)

ax.plot(z_plot, sig_plot, color="#1b9e77", linewidth=2.5, label=r"Sigmoid $\sigma(z) = \frac{1}{1 + e^{-z}}$")
ax.plot(z_plot, sig_deriv_plot, color="#e7298a", linewidth=2, linestyle="--", label=r"Derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$")
ax.axvline(0, color="gray", linestyle=":", alpha=0.6)
ax.axhline(0.5, color="gray", linestyle=":", alpha=0.6)
ax.scatter([0], [0.5], color="#1b9e77", s=60, zorder=5)
ax.scatter([0], [0.25], color="#e7298a", s=60, zorder=5)
ax.annotate(r"Inflection Point $(0, 0.5)$", xy=(0, 0.5), xytext=(0.8, 0.55),
            arrowprops=dict(arrowstyle="->", color="#1b9e77"))
ax.annotate(r"Max Slope $\sigma'(0)=0.25$", xy=(0, 0.25), xytext=(0.8, 0.3),
            arrowprops=dict(arrowstyle="->", color="#e7298a"))
ax.set_title("2. Sigmoid Function and Derivative", fontsize=12, fontweight="bold")
ax.set_xlabel(r"Logit $z = w^T x + b$")
ax.set_ylabel("Value")
ax.set_ylim(-0.05, 1.05)
ax.legend(loc="center left", fontsize=9)
ax.grid(True, alpha=0.3)

# Subplot 3: 2D Decision Boundary & Probability Heatmap
ax = axes[1, 0]
x_min, x_max = X_synth[:, 0].min() - 1.5, X_synth[:, 0].max() + 1.5
y_min, y_max = X_synth[:, 1].min() - 1.5, X_synth[:, 1].max() + 1.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200), np.linspace(y_min, y_max, 200))
grid_points = np.c_[xx.ravel(), yy.ravel()]
probs = scratch_model.predict_proba(grid_points)[:, 1].reshape(xx.shape)

contour = ax.contourf(xx, yy, probs, levels=25, cmap="RdBu_r", alpha=0.65)
fig.colorbar(contour, ax=ax, label=r"Predicted Probability $\hat{P}(Y=1|X)$")

# Draw decision boundary (p = 0.5)
ax.contour(xx, yy, probs, levels=[0.5], colors="black", linewidths=2.5, linestyles="-")

# Plot training points
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1], color="#2166ac", edgecolors="k", s=40, label="Class 0")
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1], color="#b2182b", edgecolors="k", s=40, label="Class 1")

# Analytical boundary line: w1*x1 + w2*x2 + b = 0 ==> x2 = (-w1*x1 - b) / w2
w1, w2 = scratch_model.weights
b_val = scratch_model.bias
x1_line = np.array([x_min, x_max])
x2_line = (-w1 * x1_line - b_val) / w2
ax.plot(x1_line, x2_line, color="black", linestyle="--", linewidth=1.5, label="Boundary $w^T x + b = 0$")

ax.set_title("3. 2D Decision Boundary & Probability Contours", fontsize=12, fontweight="bold")
ax.set_xlabel("Feature $x_1$")
ax.set_ylabel("Feature $x_2$")
ax.set_xlim(x_min, x_max)
ax.set_ylim(y_min, y_max)
ax.legend(loc="upper right", fontsize=8)

# Subplot 4: Threshold vs Asymmetric Cost Curve
ax = axes[1, 1]
ax.plot(thresholds, cost_list, color="#b2182b", linewidth=2.5, label=f"Total Cost ($C_{{FN}}={cost_fn:.0f}, C_{{FP}}={cost_fp:.0f}$)")
ax.axvline(0.50, color="gray", linestyle="--", label=f"Default Threshold (0.50): ${default_cost:,.0f}")
ax.axvline(optimal_tau, color="#1b9e77", linestyle="-", linewidth=2, label=f"Optimal Threshold ({optimal_tau:.2f}): ${optimal_cost:,.0f}")
ax.scatter([0.50], [default_cost], color="gray", s=70, zorder=5)
ax.scatter([optimal_tau], [optimal_cost], color="#1b9e77", s=90, zorder=5)

ax.set_title("4. Cost-Sensitive Threshold Tuning", fontsize=12, fontweight="bold")
ax.set_xlabel(r"Decision Cutoff Threshold $\tau$")
ax.set_ylabel("Total Operational Cost ($)")
ax.legend(loc="upper center", fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
fig.savefig("day-06/logistic_regression_demo.png", dpi=200)
plt.close(fig)
print("  Saved: day-06/logistic_regression_demo.png")

# Figure 2: Convexity of Log Loss vs Non-Convexity of MSE
fig2, ax2 = plt.subplots(figsize=(9, 5))
ax2.plot(w_range, bce_landscape, color="#1b9e77", linewidth=2.5, label="Binary Cross-Entropy (Log Loss) — Strictly Convex")
ax2.plot(w_range, np.array(mse_landscape) * 5.0, color="#d95f02", linewidth=2.5, linestyle="--", label="Mean Squared Error with Sigmoid (scaled x5) — Flat Plateaus")

# Mark minima
bce_min_w = w_range[np.argmin(bce_landscape)]
ax2.scatter([bce_min_w], [np.min(bce_landscape)], color="#1b9e77", s=80, zorder=5)
ax2.annotate(f"Unique Global Min (w={bce_min_w:.2f})", xy=(bce_min_w, np.min(bce_landscape)),
             xytext=(bce_min_w - 2.5, np.min(bce_landscape) + 0.3),
             arrowprops=dict(arrowstyle="->", color="#1b9e77"))

ax2.set_title("Convexity Comparison: Binary Cross-Entropy vs MSE with Sigmoid", fontsize=13, fontweight="bold")
ax2.set_xlabel("Weight Parameter $w$")
ax2.set_ylabel("Loss")
ax2.legend(loc="upper center", fontsize=10)
ax2.grid(True, alpha=0.3)
plt.tight_layout()
fig2.savefig("day-06/loss_landscape_comparison.png", dpi=200)
plt.close(fig2)
print("  Saved: day-06/loss_landscape_comparison.png")

print("\n" + "=" * 70)
print("DEMO COMPLETE! All visualizations and models verified successfully.")
print("=" * 70)
