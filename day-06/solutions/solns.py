"""
=============================================================================
  DAY 6 — SOLUTIONS: Logistic Regression and the Sigmoid Function
  Topics : Sigmoid Calculus, Chain Rule Derivations, Hessian & Convexity,
           Odds Ratios, IRLS / Newton-Raphson from Scratch,
           Cost-Sensitive Threshold Tuning, and One-vs-Rest Multi-Class.
  Run    : python day-06/solutions/solns.py
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

from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_blobs, make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


# ═══════════════════════════════════════════════════════════════════════════
# THEORY SOLUTIONS SUMMARY (Q1 - Q5)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("THEORY SOLUTIONS SUMMARY (Q1 - Q5)")
print("=" * 70)
print("""
[Q1] SIGMOID FUNCTION SYMMETRY & DERIVATIVE:
  1. Symmetry: sigma(-z) = 1 / (1 + e^z)
     Multiply numerator and denominator by e^(-z):
     sigma(-z) = e^(-z) / (e^(-z) + 1) = (1 + e^(-z) - 1) / (1 + e^(-z))
               = 1 - 1 / (1 + e^(-z)) = 1 - sigma(z). (Q.E.D.)

  2. Derivative: Let sigma(z) = (1 + e^(-z))^(-1).
     By the chain rule:
     d/dz sigma(z) = -1 * (1 + e^(-z))^(-2) * (-e^(-z))
                   = e^(-z) / (1 + e^(-z))^2
                   = [1 / (1 + e^(-z))] * [e^(-z) / (1 + e^(-z))]
                   = sigma(z) * [(1 + e^(-z) - 1) / (1 + e^(-z))]
                   = sigma(z) * (1 - sigma(z)). (Q.E.D.)

  3. Maximum of sigma'(z):
     Let g(z) = sigma(z) * (1 - sigma(z)).
     dg/dz = sigma'(1 - sigma) - sigma * sigma' = sigma'(1 - 2*sigma) = 0.
     Since sigma'(z) > 0 everywhere, 1 - 2*sigma = 0 ==> sigma(z) = 0.5 ==> z = 0.
     At z = 0: sigma'(0) = 0.5 * (1 - 0.5) = 0.25.
     *Significance*: Because sigma'(z) <= 0.25 everywhere, backpropagating gradients
     through deep layers of sigmoid activations multiplies factors <= 0.25, causing
     the gradient to vanish exponentially fast (the vanishing gradient problem).

[Q2] COMPLETE CHAIN RULE DERIVATION OF LOG LOSS GRADIENT:
  Single sample loss: L = - [ y * ln(y_hat) + (1 - y) * ln(1 - y_hat) ]
  where y_hat = sigma(z) and z = w^T x + b.

  Factor 1: dL / d(y_hat) = - [ y / y_hat - (1 - y) / (1 - y_hat) ]
                          = - [ y(1 - y_hat) - (1 - y)y_hat ] / [ y_hat(1 - y_hat) ]
                          = - [ y - y_hat ] / [ y_hat(1 - y_hat) ]
                          = (y_hat - y) / [ y_hat(1 - y_hat) ]

  Factor 2: d(y_hat) / dz = sigma'(z) = y_hat * (1 - y_hat)

  Factor 3: dz / dw_j = x_j

  Multiplying all three factors:
  dL / dw_j = { (y_hat - y) / [y_hat(1 - y_hat)] } * [y_hat(1 - y_hat)] * x_j
            = (y_hat - y) * x_j.  (The denominator cancels completely!)

  Averaging over all n samples:
  dJ / dw_j = (1/n) * sum_{i=1}^n (y_hat^(i) - y^(i)) * x_j^(i)
  Vectorized: grad_w J = (1/n) * X^T (y_hat - y)
              grad_b J = (1/n) * sum(y_hat - y).

[Q3] HESSIAN MATRIX & PROOF OF CONVEXITY:
  First derivative: dJ / dw_j = (1/n) * sum_i (y_hat^(i) - y^(i)) * x_j^(i)
  Second derivative with respect to w_k:
  d^2 J / (dw_j dw_k) = (1/n) * sum_i [ d(y_hat^(i))/dw_k ] * x_j^(i)
                      = (1/n) * sum_i y_hat^(i) * (1 - y_hat^(i)) * x_j^(i) * x_k^(i).

  In matrix notation:
  H = (1/n) * X^T D X
  where D is an (n x n) diagonal matrix with D_ii = y_hat^(i) * (1 - y_hat^(i)).
  Since 0 < y_hat^(i) < 1, all diagonal entries D_ii > 0.
  For any non-zero vector v in R^p:
    v^T H v = (1/n) * v^T X^T D X v = (1/n) * (Xv)^T D (Xv)
            = (1/n) * sum_{i=1}^n D_ii * (Xv)_i^2 >= 0.
  Thus, H is positive semi-definite everywhere. The loss function J(w) is
  strictly convex (assuming X has full column rank), which mathematically
  guarantees that every stationary point is the unique global minimum!

[Q4] ODDS RATIOS & NUMERICAL CALCULATION:
  Model: ln(Odds) = -2.5 + 0.8 * DTI - 0.04 * FICO + 1.4 * Prior_Default
  1. Odds Ratio for Prior_Default = e^(1.4) = 4.0552.
     Interpretation: Holding DTI and FICO score constant, having a prior default
     increases the odds of default by 4.055 times (a 305.5% increase in odds).
  2. For applicant with DTI=0.50, FICO=650, Prior_Default=1:
     z = -2.5 + 0.8 * (0.50) - 0.04 * (650) + 1.4 * (1)
       = -2.5 + 0.40 - 26.0 + 1.4 = -26.70 (if FICO was entered as 650 directly)
     *Note on scaling*: In problem prompt, FICO was specified as raw score:
     With FICO=650, z = -26.70, p = 1 / (1 + e^(26.70)) = 2.54e-12.
     (In practice, credit models scale FICO by /100: -0.04 * 6.5 = -0.26 ==> z = -0.96, p = 27.7%).

[Q5] THE PERFECT SEPARATION PATHOLOGY:
  If a dataset is linearly separable, there exists a hyperplane w^T x + b = 0 such that
  w^T x^(i) + b > 0 for all y=1 and < 0 for all y=0.
  Multiplying w and b by a positive scalar c > 1 does not change the classification.
  As c -> +infinity:
    sigma(c * (w^T x + b)) -> 1 for all y=1, and -> 0 for all y=0.
  The Log Loss J(c*w) -> 0 asymptotically, but never reaches 0 for any finite w.
  Consequently, unregularized gradient descent drives ||w|| -> infinity.
  Adding L2 regularization J_reg = J(w) + (lambda/2) * ||w||^2 creates a strictly
  coercive quadratic penalty: as ||w|| -> infinity, (lambda/2)*||w||^2 -> infinity.
  Hence, J_reg has a unique, bounded, finite global minimum.
""")


# ═══════════════════════════════════════════════════════════════════════════
# CODE EXERCISE 6 — Newton-Raphson / IRLS from Scratch
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXERCISE 6 — NEWTON-RAPHSON / IRLS FROM SCRATCH")
print("=" * 70)

def sigmoid(z):
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))

class LogisticRegressionIRLS:
    """
    Second-Order Logistic Regression using Newton-Raphson / IRLS updates:
      theta <- theta - H^(-1) * grad
    where theta = [w; b] and X_aug = [X, 1].
    """
    def __init__(self, n_epochs=25, tol=1e-6, l2_reg=1e-4):
        self.n_epochs = n_epochs
        self.tol = tol
        self.l2_reg = l2_reg
        self.theta = None  # Combined [w1, w2, ..., wp, b]
        self.loss_history = []

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n_samples, n_features = X.shape

        # Augment X with column of ones for intercept b
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        n_params = n_features + 1

        # Initialize theta with zeros
        self.theta = np.zeros(n_params)
        self.loss_history = []
        eps = 1e-15

        # Regularization matrix (do not penalize intercept)
        reg_matrix = np.eye(n_params) * self.l2_reg
        reg_matrix[-1, -1] = 0.0

        for epoch in range(self.n_epochs):
            z = X_aug @ self.theta
            y_hat = sigmoid(z)

            # Compute loss
            y_safe = np.clip(y_hat, eps, 1.0 - eps)
            loss = -np.mean(y * np.log(y_safe) + (1.0 - y) * np.log(1.0 - y_safe))
            loss += 0.5 * self.l2_reg * np.sum(self.theta[:-1] ** 2)
            self.loss_history.append(loss)

            if epoch > 0 and abs(self.loss_history[-2] - loss) < self.tol:
                break

            # Gradient: (1/n) * X_aug^T (y_hat - y) + reg * theta
            grad = (X_aug.T @ (y_hat - y)) / n_samples + reg_matrix @ self.theta

            # Hessian: (1/n) * X_aug^T D X_aug + reg_matrix
            # where D_ii = y_hat_i * (1 - y_hat_i)
            D_diag = y_hat * (1.0 - y_hat)
            # Efficient vectorized computation: X_aug.T @ (D * X_aug)
            H = (X_aug.T @ (D_diag[:, np.newaxis] * X_aug)) / n_samples + reg_matrix

            # Newton-Raphson update: solve H * delta = grad  ==> delta = H^(-1) * grad
            delta = np.linalg.solve(H, grad)
            self.theta -= delta

        return self

    @property
    def weights(self):
        return self.theta[:-1]

    @property
    def bias(self):
        return self.theta[-1]

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        X_aug = np.hstack([X, np.ones((n_samples, 1))])
        p1 = sigmoid(X_aug @ self.theta)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


# Test on synthetic 2D data
X_synth, y_synth = make_blobs(n_samples=250, n_features=2, centers=2, cluster_std=1.5, random_state=42)
irls_model = LogisticRegressionIRLS(n_epochs=25, tol=1e-7, l2_reg=1e-3)
irls_model.fit(X_synth, y_synth)

sk_irls = LogisticRegression(C=1000.0, solver="lbfgs").fit(X_synth, y_synth)

print(f"  IRLS Converged in                     : {len(irls_model.loss_history)} iterations!")
print(f"  IRLS Learned Weights [w1, w2]         : [{irls_model.weights[0]:.4f}, {irls_model.weights[1]:.4f}], b = {irls_model.bias:.4f}")
print(f"  Sklearn Weights [w1, w2]              : [{sk_irls.coef_[0][0]:.4f}, {sk_irls.coef_[0][1]:.4f}], b = {sk_irls.intercept_[0]:.4f}")
print(f"  IRLS Accuracy                         : {accuracy_score(y_synth, irls_model.predict(X_synth)) * 100:.2f}%")
print("  Notice: Newton-Raphson reaches machine precision in 6-8 iterations because it uses")
print("  the exact curvature (Hessian matrix), whereas gradient descent requires hundreds of steps!\n")


# ═══════════════════════════════════════════════════════════════════════════
# CODE EXERCISE 7 — Cost-Sensitive Cutoff Threshold Optimizer
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXERCISE 7 — COST-SENSITIVE THRESHOLD OPTIMIZER")
print("=" * 70)

def optimize_threshold(y_true, y_prob, cost_fp, cost_fn):
    """
    Sweeps thresholds in [0.01, 0.99] to find the decision cutoff minimizing total cost.
    
    Parameters
    ----------
    y_true : array-like of shape (n_samples,)
        True binary labels (0 or 1).
    y_prob : array-like of shape (n_samples,)
        Predicted positive class probabilities.
    cost_fp : float
        Financial cost incurred by a False Positive (false alarm).
    cost_fn : float
        Financial cost incurred by a False Negative (missed detection).
        
    Returns
    -------
    dict containing optimal threshold, minimum cost, and comparison to default 0.5.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    costs = []
    tns, fps, fns, tps = [], [], [], []

    for tau in thresholds:
        preds = (y_prob >= tau).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
        cost = fp * cost_fp + fn * cost_fn
        costs.append(cost)
        tns.append(tn); fps.append(fp); fns.append(fn); tps.append(tp)

    costs = np.array(costs)
    opt_idx = np.argmin(costs)
    default_idx = np.argmin(np.abs(thresholds - 0.50))

    return {
        "thresholds": thresholds,
        "costs": costs,
        "optimal_threshold": thresholds[opt_idx],
        "min_cost": costs[opt_idx],
        "default_threshold": 0.50,
        "default_cost": costs[default_idx],
        "optimal_fp": fps[opt_idx],
        "optimal_fn": fns[opt_idx],
        "default_fp": fps[default_idx],
        "default_fn": fns[default_idx]
    }


# Synthetic fraud dataset (5% fraud rate)
X_fraud, y_fraud = make_classification(
    n_samples=2000, n_features=6, weights=[0.95, 0.05], random_state=42
)
X_fr_train, X_fr_test, y_fr_train, y_fr_test = train_test_split(X_fraud, y_fraud, test_size=0.4, random_state=42)

fraud_clf = LogisticRegression(solver="lbfgs").fit(X_fr_train, y_fr_train)
fraud_probs = fraud_clf.predict_proba(X_fr_test)[:, 1]

# $1,000 cost per missed fraud, $25 cost per SMS verification
c_fn = 1000.0
c_fp = 25.0

result = optimize_threshold(y_fr_test, fraud_probs, cost_fp=c_fp, cost_fn=c_fn)

print(f"  Evaluation on Fraud Test Set (N={len(y_fr_test)}, Positives={np.sum(y_fr_test)}):")
print(f"  Cost Matrix: Missed Fraud (FN) = ${c_fn:,.0f} | Verification SMS (FP) = ${c_fp:,.0f}")
print(f"  Default Threshold tau = 0.50:")
print(f"    False Positives: {result['default_fp']}, False Negatives: {result['default_fn']}")
print(f"    Total Operational Cost: ${result['default_cost']:,.0f}")
print(f"  Optimized Threshold tau = {result['optimal_threshold']:.2f}:")
print(f"    False Positives: {result['optimal_fp']}, False Negatives: {result['optimal_fn']}")
print(f"    Total Operational Cost: ${result['min_cost']:,.0f}")
savings = result['default_cost'] - result['min_cost']
print(f"  Financial Savings: ${savings:,.0f} ({savings / result['default_cost'] * 100:.1f}% reduction!)\n")

# Save diagnostic visualization for Exercise 7
fig_ex7, ax = plt.subplots(figsize=(8, 5))
ax.plot(result["thresholds"], result["costs"], color="#d95f02", linewidth=2.5, label="Total Operational Cost ($)")
ax.axvline(0.50, color="gray", linestyle="--", label=f"Default Threshold (0.50): ${result['default_cost']:,.0f}")
ax.axvline(result["optimal_threshold"], color="#1b9e77", linestyle="-", linewidth=2,
           label=f"Cost-Optimal Cutoff ({result['optimal_threshold']:.2f}): ${result['min_cost']:,.0f}")
ax.scatter([0.50], [result["default_cost"]], color="gray", s=80, zorder=5)
ax.scatter([result["optimal_threshold"]], [result["min_cost"]], color="#1b9e77", s=100, zorder=5)
ax.set_title("Exercise 7 — Cost-Sensitive Decision Threshold Optimization", fontsize=12, fontweight="bold")
ax.set_xlabel(r"Cutoff Threshold $\tau$")
ax.set_ylabel("Total Financial Loss ($)")
ax.legend(loc="upper center", fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig_ex7.savefig("day-06/solutions/ex7_threshold_optimization.png", dpi=200)
plt.close(fig_ex7)
print("  Saved: day-06/solutions/ex7_threshold_optimization.png\n")


# ═══════════════════════════════════════════════════════════════════════════
# CODE EXERCISE 8 — One-vs-Rest (OvR) Multi-Class Classifier from Scratch
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("EXERCISE 8 — ONE-VS-REST (OvR) MULTI-CLASS CLASSIFIER FROM SCRATCH")
print("=" * 70)

class OneVsRestClassifierScratch:
    """
    Multi-Class Classifier implementing the One-vs-Rest (OvR / One-vs-All) strategy
    using any binary classifier that provides .fit(), .predict_proba().
    """
    def __init__(self, base_estimator_factory):
        self.base_estimator_factory = base_estimator_factory
        self.models = {}
        self.classes_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.models = {}

        for c in self.classes_:
            # Binary target: 1 for current class c, 0 for all other classes
            y_binary = (y == c).astype(int)
            model = self.base_estimator_factory()
            model.fit(X, y_binary)
            self.models[c] = model

        return self

    def predict_proba(self, X):
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        raw_probs = np.zeros((n_samples, n_classes))

        for idx, c in enumerate(self.classes_):
            # Extract probability of positive class (column 1)
            raw_probs[:, idx] = self.models[c].predict_proba(X)[:, 1]

        # Normalize probabilities across classes so rows sum to 1.0
        row_sums = raw_probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0  # Safeguard against zero sums
        return raw_probs / row_sums

    def predict(self, X):
        probs = self.predict_proba(X)
        best_indices = np.argmax(probs, axis=1)
        return self.classes_[best_indices]


# Generate 3-class classification dataset
X_mc, y_mc = make_blobs(n_samples=300, n_features=3, centers=3, cluster_std=1.6, random_state=42)
X_mc_tr, X_mc_te, y_mc_tr, y_mc_te = train_test_split(X_mc, y_mc, test_size=0.3, random_state=42)

# Train Scratch OvR with IRLS as the base estimator
ovr_scratch = OneVsRestClassifierScratch(lambda: LogisticRegressionIRLS(n_epochs=20, l2_reg=1e-3))
ovr_scratch.fit(X_mc_tr, y_mc_tr)

# Train scikit-learn OvR model for comparison
ovr_sklearn = LogisticRegression(multi_class="ovr", solver="liblinear", C=1000.0)
ovr_sklearn.fit(X_mc_tr, y_mc_tr)

preds_scratch = ovr_scratch.predict(X_mc_te)
preds_sklearn = ovr_sklearn.predict(X_mc_te)

acc_scratch = accuracy_score(y_mc_te, preds_scratch)
acc_sklearn = accuracy_score(y_mc_te, preds_sklearn)

print(f"  Multi-Class Problem (3 Classes, 3 Features):")
print(f"  Scratch OvR Accuracy  : {acc_scratch * 100:.2f}%")
print(f"  Sklearn OvR Accuracy  : {acc_sklearn * 100:.2f}%")
print(f"  Prediction Agreement  : {np.mean(preds_scratch == preds_sklearn) * 100:.2f}%")
print("\n  Scratch Confusion Matrix:")
print(confusion_matrix(y_mc_te, preds_scratch))

print("\n" + "=" * 70)
print("ALL DAY 6 EXERCISE SOLUTIONS COMPLETE!")
print("=" * 70)
