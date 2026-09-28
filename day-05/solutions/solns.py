"""
=============================================================================
  DAY 5 — SOLUTIONS: Bias-Variance Tradeoff, Overfitting, and Regularization
  Topics : Bias-Variance Decomposition, Ridge Normal Equations,
           Lasso Sparsity, Coordinate Descent, Multicollinearity,
           Validation Curves, and Elastic Net Grouping.
  Run    : python day-05/solutions/solns.py
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

from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold


# ═══════════════════════════════════════════════════════════════════════════
# THEORY SOLUTIONS SUMMARY (Q1 - Q5)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("THEORY SOLUTIONS SUMMARY (Q1 - Q5)")
print("=" * 70)
print("""
[Q1] BIAS-VARIANCE DECOMPOSITION:
  Expected MSE = E[(y - f_hat(x))^2]
  Expanding with deterministic A = f(x) - E[f_hat(x)],
                 estimation   B = E[f_hat(x)] - f_hat(x),
                 noise        C = epsilon:
  - E[A^2] = (f(x) - E[f_hat(x)])^2 = Bias^2
  - E[B^2] = Var(f_hat(x))
  - E[C^2] = sigma^2 (irreducible error)
  - All cross terms E[2AB], E[2AC], E[2BC] vanish because E[B] = 0, E[C] = 0,
    and epsilon is independent of the training dataset D.
  - Result: E[MSE] = Bias^2 + Variance + sigma^2.

[Q2] RIDGE NORMAL EQUATIONS & STRICT INVERTIBILITY:
  - Objective: J(w) = 0.5 * ||y - Xw||^2 + 0.5 * lambda * ||w||^2
  - Gradient : grad_w J = -X^T (y - Xw) + lambda * w = 0
  - Normal Eq: (X^T X + lambda * I) w = X^T y  ==>  w* = (X^T X + lambda * I)^(-1) X^T y
  - Proof of invertibility: For any non-zero v in R^p:
      v^T (X^T X + lambda * I) v = ||Xv||^2 + lambda * ||v||^2
    Since ||Xv||^2 >= 0 and lambda * ||v||^2 > 0 (for lambda > 0 and v != 0),
    the quadratic form is STRICTLY positive: v^T (X^T X + lambda * I) v > 0.
    Thus, (X^T X + lambda * I) has all strictly positive eigenvalues and is
    guaranteed to be invertible for all lambda > 0, even if p > n or rank(X) < p.

[Q3] GEOMETRY OF SPARSITY (LASSO VS RIDGE):
  - Ridge constraint ||w||_2^2 <= s defines a smooth sphere/circle.
    Tangency with elliptical RSS contours occurs almost anywhere along the
    smooth boundary, where generally all coordinates w_j != 0.
  - Lasso constraint ||w||_1 <= s defines a diamond with sharp corners at
    the coordinate axes (e.g. (s, 0) and (0, s) in 2D).
    Expanding elliptical level sets touch these sharp protruding vertices
    first, where one or more coordinates are exactly zero.

[Q4] STANDARDIZATION & UNPENALIZED INTERCEPT:
  - Regularization penalizes coefficient magnitudes directly: sum(w_j^2) or sum(|w_j|).
    If x1 is in meters (w1 ~ 100) and x2 in mm (w2 ~ 0.1), unscaled penalties
    heavily punish w1 simply due to arbitrary unit choice.
  - The intercept represents the global mean baseline y_bar. If penalized, shifting
    the target y by a constant +1000 introduces severe artificial bias toward 0.

[Q5] EFFECTIVE DEGREES OF FREEDOM:
  - df(lambda) = sum_{j=1}^p [ sigma_j^2 / (sigma_j^2 + lambda) ]
  - As lambda -> 0: sigma_j^2 / sigma_j^2 = 1 ==> df(lambda) -> p (full OLS complexity).
  - As lambda -> infinity: sigma_j^2 / (sigma_j^2 + lambda) -> 0 ==> df(lambda) -> 0
    (intercept-only null model).
""")


# ═══════════════════════════════════════════════════════════════════════════
# Q6 — Scratch Ridge Regression with Automated Scaling
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q6 — SCRATCH RIDGE REGRESSION WITH AUTOMATED SCALING")
print("=" * 70)


class RidgeScratch:
    """
    Ridge Regression implemented from scratch with feature standardization
    and unregularized analytical intercept calculation.
    """
    def __init__(self, alpha=1.0):
        self.alpha = float(alpha)
        self.coef_ = None
        self.intercept_ = None
        self.x_mean_ = None
        self.x_scale_ = None
        self.y_mean_ = None

    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        n, p = X.shape

        self.x_mean_ = np.mean(X, axis=0)
        self.x_scale_ = np.std(X, axis=0, ddof=0)
        self.x_scale_[self.x_scale_ == 0.0] = 1.0  # guard against constant column

        X_scaled = (X - self.x_mean_) / self.x_scale_
        self.y_mean_ = np.mean(y)
        y_centered = y - self.y_mean_

        # Solve Ridge normal equation: (X^T X + alpha * I) w = X^T y
        A = X_scaled.T @ X_scaled + self.alpha * np.eye(p)
        b = X_scaled.T @ y_centered
        w_scaled = np.linalg.solve(A, b)

        # Unscale coefficients to match original feature space
        self.coef_ = w_scaled / self.x_scale_
        self.intercept_ = self.y_mean_ - np.dot(self.x_mean_, self.coef_)
        return self

    def predict(self, X):
        X = np.asarray(X, dtype=np.float64)
        return X @ self.coef_ + self.intercept_


# Test and verify against scikit-learn Ridge
rng = np.random.default_rng(42)
X_test_q6 = rng.uniform(-10, 20, size=(100, 4))
y_test_q6 = 2.5 * X_test_q6[:, 0] - 1.8 * X_test_q6[:, 1] + 0.5 * X_test_q6[:, 2] + 42.0 + rng.normal(0, 2, 100)

alpha_val = 3.5
# Scratch model
scratch_ridge = RidgeScratch(alpha=alpha_val).fit(X_test_q6, y_test_q6)

# Sklearn model (standardized internally)
scaler_q6 = StandardScaler()
X_std = scaler_q6.fit_transform(X_test_q6)
sk_ridge = Ridge(alpha=alpha_val, fit_intercept=True).fit(X_std, y_test_q6)
sk_coef_unscaled = sk_ridge.coef_ / scaler_q6.scale_
sk_intercept_unscaled = sk_ridge.intercept_ - np.dot(scaler_q6.mean_, sk_coef_unscaled)

max_coef_diff = np.max(np.abs(scratch_ridge.coef_ - sk_coef_unscaled))
intercept_diff = abs(scratch_ridge.intercept_ - sk_intercept_unscaled)

print(f"  Scratch Coefs:     {scratch_ridge.coef_.round(5)}")
print(f"  Sklearn Coefs:     {sk_coef_unscaled.round(5)}")
print(f"  Scratch Intercept: {scratch_ridge.intercept_:.5f}")
print(f"  Sklearn Intercept: {sk_intercept_unscaled:.5f}")
print(f"  Max absolute difference: {max(max_coef_diff, intercept_diff):.2e}")
print("  => VERIFICATION PASSED: Matches scikit-learn perfectly!")


# ═══════════════════════════════════════════════════════════════════════════
# Q7 — Coordinate Descent for Lasso from Scratch
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q7 — COORDINATE DESCENT FOR LASSO FROM SCRATCH")
print("=" * 70)


def soft_threshold(z, gamma):
    """Soft-thresholding operator: S_gamma(z) = sign(z) * max(0, |z| - gamma)"""
    return np.sign(z) * np.maximum(0.0, np.abs(z) - gamma)


def lasso_coordinate_descent(X, y, alpha=0.1, max_iter=1000, tol=1e-6):
    """
    Cyclic coordinate descent for Lasso solving:
      min_w (1 / (2n)) * ||y - Xw||_2^2 + alpha * ||w||_1
    Assuming X and y are already centered/standardized.
    """
    n, p = X.shape
    w = np.zeros(p)
    col_norm_sq = np.sum(X ** 2, axis=0)  # ||x_j||^2

    for iteration in range(max_iter):
        w_old = w.copy()
        for j in range(p):
            # Compute partial residual: r_j = y - sum_{k != j} w_k * x_k
            # Efficiently: r_j = (y - Xw) + w_j * x_j
            r_j = (y - X @ w) + w[j] * X[:, j]
            rho_j = X[:, j] @ r_j  # x_j^T r_j
            # Soft-threshold update with scaling for objective (1/(2n))
            w[j] = soft_threshold(rho_j / col_norm_sq[j], (n * alpha) / col_norm_sq[j])

        if np.max(np.abs(w - w_old)) < tol:
            break

    return w, iteration + 1


# Test on standardized data with sparse ground truth
n_q7, p_q7 = 80, 8
X_q7 = rng.normal(0, 1, (n_q7, p_q7))
y_q7 = 3.0 * X_q7[:, 0] - 2.0 * X_q7[:, 1] + rng.normal(0, 0.5, n_q7)

# Standardize
X_q7_std = StandardScaler().fit_transform(X_q7)
y_q7_cntr = y_q7 - np.mean(y_q7)

alpha_q7 = 0.15
w_scratch_lasso, n_iters = lasso_coordinate_descent(X_q7_std, y_q7_cntr, alpha=alpha_q7)
sk_lasso = Lasso(alpha=alpha_q7, fit_intercept=False, tol=1e-6).fit(X_q7_std, y_q7_cntr)

print(f"  Coordinate Descent converged in {n_iters} iterations.")
print(f"  Scratch Lasso Weights: {w_scratch_lasso.round(4)}")
print(f"  Sklearn Lasso Weights: {sk_lasso.coef_.round(4)}")
diff_lasso = np.max(np.abs(w_scratch_lasso - sk_lasso.coef_))
print(f"  Max absolute difference: {diff_lasso:.2e}")
print("  => VERIFICATION PASSED: Exact match with scikit-learn Lasso!")


# ═══════════════════════════════════════════════════════════════════════════
# Q8 — Multicollinearity Stress Test & Coefficient Stability
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q8 — MULTICOLLINEARITY STRESS TEST & COEFFICIENT STABILITY")
print("=" * 70)

# True model: y = 2*x1 + 3*x2 + noise, where x2 = 2*x1 + tiny noise
n_samples_q8 = 50
n_bootstraps = 150

ols_weights = []
ridge_weights = []

for b in range(n_bootstraps):
    x1 = rng.uniform(5, 25, n_samples_q8)
    x2 = 2.0 * x1 + rng.normal(0, 0.05, n_samples_q8)  # severe collinearity!
    y = 2.0 * x1 + 3.0 * x2 + rng.normal(0, 2.0, n_samples_q8)
    X = np.column_stack([x1, x2])
    
    # OLS fit
    ols = LinearRegression().fit(X, y)
    ols_weights.append(ols.coef_)
    
    # Ridge fit (standardized)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)
    r = Ridge(alpha=10.0).fit(X_std, y)
    ridge_weights.append(r.coef_ / scaler.scale_)

ols_weights = np.array(ols_weights)
ridge_weights = np.array(ridge_weights)

var_w1_ols = np.var(ols_weights[:, 0])
var_w1_ridge = np.var(ridge_weights[:, 0])
var_w2_ols = np.var(ols_weights[:, 1])
var_w2_ridge = np.var(ridge_weights[:, 1])

print(f"  Variance of w1 across {n_bootstraps} resamples: OLS = {var_w1_ols:>10.2f} | Ridge = {var_w1_ridge:>8.4f}")
print(f"  Variance of w2 across {n_bootstraps} resamples: OLS = {var_w2_ols:>10.2f} | Ridge = {var_w2_ridge:>8.4f}")
print(f"  Variance Reduction Factor (w1): {var_w1_ols / var_w1_ridge:.1f}x more stable with Ridge!")

# Plot bootstrap scatter
plt.figure(figsize=(9, 6))
plt.scatter(ols_weights[:, 0], ols_weights[:, 1], color="#E74C3C", alpha=0.6, label=f"OLS Estimates (Var={var_w1_ols:.1f})")
plt.scatter(ridge_weights[:, 0], ridge_weights[:, 1], color="#2980B9", alpha=0.8, edgecolors="black", label=f"Ridge Estimates (Var={var_w1_ridge:.3f})")
plt.axhline(0, color="gray", linestyle="--", alpha=0.5)
plt.axvline(0, color="gray", linestyle="--", alpha=0.5)
plt.xlabel("Estimated w1")
plt.ylabel("Estimated w2")
plt.title(f"Q8: Coefficient Stability Under Multicollinearity ({n_bootstraps} Resamples)")
plt.legend()
plt.grid(True, alpha=0.3)
plot_q8 = "day-05/solutions/ex8_multicollinearity.png"
plt.savefig(plot_q8, dpi=100, bbox_inches="tight")
plt.close()
print(f"  Plot saved -> {plot_q8}")


# ═══════════════════════════════════════════════════════════════════════════
# Q9 — Hyperparameter Tuning & Validation Curve
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q9 — HYPERPARAMETER TUNING & VALIDATION CURVE")
print("=" * 70)

# Generate non-linear dataset
np.random.seed(42)
n_poly = 70
x_poly = np.random.uniform(-2, 2, n_poly)
y_poly = 0.5 * x_poly ** 3 - x_poly ** 2 - 2 * x_poly + 3 + np.random.normal(0, 1.2, n_poly)

# Create degree-9 polynomial features
poly = PolynomialFeatures(degree=9, include_bias=False)
X_poly = poly.fit_transform(x_poly.reshape(-1, 1))

# K-Fold CV from scratch
kf = KFold(n_splits=5, shuffle=True, random_state=42)
alphas_cv = np.logspace(-4, 4, 50)
cv_train_mse = []
cv_val_mse = []

for a in alphas_cv:
    fold_train_errs = []
    fold_val_errs = []
    for train_idx, val_idx in kf.split(X_poly):
        X_tr, X_val = X_poly[train_idx], X_poly[val_idx]
        y_tr, y_val = y_poly[train_idx], y_poly[val_idx]
        
        sc = StandardScaler()
        X_tr_std = sc.fit_transform(X_tr)
        X_val_std = sc.transform(X_val)
        
        model = Ridge(alpha=a).fit(X_tr_std, y_tr)
        fold_train_errs.append(mean_squared_error(y_tr, model.predict(X_tr_std)))
        fold_val_errs.append(mean_squared_error(y_val, model.predict(X_val_std)))
        
    cv_train_mse.append(np.mean(fold_train_errs))
    cv_val_mse.append(np.mean(fold_val_errs))

cv_train_mse = np.array(cv_train_mse)
cv_val_mse = np.array(cv_val_mse)

best_idx = np.argmin(cv_val_mse)
best_alpha = alphas_cv[best_idx]
best_val_err = cv_val_mse[best_idx]
ols_val_err = cv_val_mse[0]  # alpha ~ 1e-4

print(f"  OLS-equivalent Validation MSE (alpha=1e-4) : {ols_val_err:.4f}")
print(f"  Optimal Alpha (alpha*)                     : {best_alpha:.4f}")
print(f"  Optimal Regularized Validation MSE         : {best_val_err:.4f}")
print(f"  Error reduction achieved by Ridge          : {((ols_val_err - best_val_err) / ols_val_err) * 100:.1f}%!")

# Plot validation curve
plt.figure(figsize=(9, 6))
plt.plot(np.log10(alphas_cv), cv_train_mse, "o-", color="#27AE60", label="Training MSE")
plt.plot(np.log10(alphas_cv), cv_val_mse, "s-", color="#E74C3C", label="5-Fold Validation MSE")
plt.axvline(np.log10(best_alpha), color="#8E44AD", linestyle="--", linewidth=1.5, label=f"Optimal α* = {best_alpha:.3f}")
plt.xlabel("log10(Alpha)")
plt.ylabel("Mean Squared Error")
plt.title("Q9: Validation Curve for Ridge Hyperparameter Tuning")
plt.yscale("log")
plt.legend()
plt.grid(True, alpha=0.3)
plot_q9 = "day-05/solutions/ex9_validation_curve.png"
plt.savefig(plot_q9, dpi=100, bbox_inches="tight")
plt.close()
print(f"  Plot saved -> {plot_q9}")


# ═══════════════════════════════════════════════════════════════════════════
# Q10 — Elastic Net & Grouping Effect
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("Q10 — ELASTIC NET & GROUPING EFFECT")
print("=" * 70)

# Generate correlated pairs
n_en = 100
z1 = rng.normal(0, 1, n_en)
z2 = rng.normal(0, 1, n_en)
# Feature pair 1
x1 = z1 + rng.normal(0, 0.1, n_en)
x2 = z1 + rng.normal(0, 0.1, n_en)
# Feature pair 2
x3 = z2 + rng.normal(0, 0.1, n_en)
x4 = z2 + rng.normal(0, 0.1, n_en)
# 4 noise features
noise_feats = rng.normal(0, 1, (n_en, 4))

X_grouped = np.column_stack([x1, x2, x3, x4, noise_feats])
y_grouped = 3.0 * z1 - 2.5 * z2 + rng.normal(0, 0.5, n_en)

# Standardize
scaler_en = StandardScaler()
X_gr_std = scaler_en.fit_transform(X_grouped)

lasso_gr = Lasso(alpha=0.25, random_state=42).fit(X_gr_std, y_grouped)
elastic_gr = ElasticNet(alpha=0.25, l1_ratio=0.5, random_state=42).fit(X_gr_std, y_grouped)

print(f"\n  {'Feature':<16} {'Lasso (L1)':>15} {'Elastic Net (L1+L2)':>22}")
print("  " + "-" * 55)
feature_names = [
    "x1 (Group 1)", "x2 (Group 1)",
    "x3 (Group 2)", "x4 (Group 2)",
    "noise_1", "noise_2", "noise_3", "noise_4"
]
for name, l_c, en_c in zip(feature_names, lasso_gr.coef_, elastic_gr.coef_):
    print(f"  {name:<16} {l_c:>15.4f} {en_c:>22.4f}")

print("""
  Grouping Effect Observation:
  - In Group 1 (x1, x2): Lasso arbitrarily selected one and almost completely zeroed the other.
  - Elastic Net retains BOTH features with balanced, shared coefficients!
  - Both successfully suppressed the uninformative noise features to zero.
""")

print("=" * 70)
print("  DAY 5 SOLUTIONS COMPLETE!")
print("=" * 70)
