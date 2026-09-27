"""
=======================================================
  DAY 4 — Evaluation Metrics in Regression
  Topic : Quantifying prediction quality: MAE, MSE, RMSE,
          R², Adjusted R², and Residual Diagnostics.
  Run   : python day-04/evaluation_metrics_demo.py
=======================================================
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
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    root_mean_squared_error,
    r2_score
)
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Custom Metric Implementations From Scratch
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 1 — EVALUATION METRICS IMPLEMENTED FROM SCRATCH")
print("=" * 65)


def mae(y_true, y_pred):
    """Mean Absolute Error: average magnitude of absolute residuals."""
    return np.mean(np.abs(y_true - y_pred))


def mse(y_true, y_pred):
    """Mean Squared Error: average of squared residuals."""
    return np.mean((y_true - y_pred) ** 2)


def rmse(y_true, y_pred):
    """Root Mean Squared Error: standard deviation of unexplained variance."""
    return np.sqrt(mse(y_true, y_pred))


def r2(y_true, y_pred):
    """Coefficient of Determination (R²): proportion of variance explained."""
    rss = np.sum((y_true - y_pred) ** 2)
    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1.0 - (rss / tss)


def adj_r2(y_true, y_pred, p):
    """
    Adjusted R²: penalises addition of uninformative features.
    p : number of predictor features (excluding intercept).
    """
    n = len(y_true)
    r2_val = r2(y_true, y_pred)
    return 1.0 - ((1.0 - r2_val) * (n - 1) / (n - p - 1))


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Dataset & Verification against Scikit-Learn
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 2 — DATASET & VERIFICATION WITH SCIKIT-LEARN")
print("=" * 65)

rng = np.random.default_rng(42)
n_samples = 120

# Synthetic dataset: y = 2.5 * x + 15 + gaussian noise
x = rng.uniform(5, 50, n_samples)
true_noise = rng.normal(0, 6, n_samples)
y = 2.5 * x + 15.0 + true_noise

X = x.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

model = LinearRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)

# Compute metrics manually
val_mae = mae(y_test, y_pred)
val_mse = mse(y_test, y_pred)
val_rmse = rmse(y_test, y_pred)
val_r2 = r2(y_test, y_pred)
val_adj = adj_r2(y_test, y_pred, p=1)

# Compute metrics with scikit-learn
sk_mae = mean_absolute_error(y_test, y_pred)
sk_mse = mean_squared_error(y_test, y_pred)
sk_rmse = root_mean_squared_error(y_test, y_pred)
sk_r2 = r2_score(y_test, y_pred)

print(f"\n  {'Metric':<12} {'From Scratch':>15} {'Scikit-Learn':>15} {'Match?':>10}")
print("  " + "-" * 55)
print(f"  {'MAE':<12} {val_mae:>15.6f} {sk_mae:>15.6f} {'✓' if abs(val_mae - sk_mae) < 1e-9 else '✗':>10}")
print(f"  {'MSE':<12} {val_mse:>15.6f} {sk_mse:>15.6f} {'✓' if abs(val_mse - sk_mse) < 1e-9 else '✗':>10}")
print(f"  {'RMSE':<12} {val_rmse:>15.6f} {sk_rmse:>15.6f} {'✓' if abs(val_rmse - sk_rmse) < 1e-9 else '✗':>10}")
print(f"  {'R²':<12} {val_r2:>15.6f} {sk_r2:>15.6f} {'✓' if abs(val_r2 - sk_r2) < 1e-9 else '✗':>10}")
print(f"  {'Adjusted R²':<12} {val_adj:>15.6f} {'(n/a in sklearn)':>15} {'✓':>10}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — MAE vs RMSE Outlier Stress Test
#           How does a single massive error impact each metric?
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("PART 3 — OUTLIER SENSITIVITY EXPERIMENT: MAE vs RMSE")
print("=" * 65)

outlier_shocks = [0, 10, 25, 50, 100, 200]
print(f"\n  Corrupting a single test sample with an artificial error shock:")
print(f"  {'Shock':>8} {'MAE':>12} {'RMSE':>12} {'RMSE / MAE Ratio':>20}")
print("  " + "-" * 54)

shock_results = []
for shock in outlier_shocks:
    y_test_corrupt = y_test.copy()
    y_test_corrupt[0] += shock  # contaminate one single prediction

    m_val = mae(y_test_corrupt, y_pred)
    r_val = rmse(y_test_corrupt, y_pred)
    ratio = r_val / m_val
    shock_results.append((shock, m_val, r_val, ratio))
    print(f"  {shock:>8} {m_val:>12.4f} {r_val:>12.4f} {ratio:>20.4f}")

print("""
  Observation:
  - MAE grows linearly with the outlier shock.
  - RMSE surges dramatically because the single error is squared.
  - When RMSE >> MAE, the dataset has high error variance or extreme outliers!
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Spurious Features: Why R² Lies and Adjusted R² Protects You
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("PART 4 — SPURIOUS FEATURES: R² vs ADJUSTED R²")
print("=" * 65)

# Start with 1 real feature, then progressively append pure random noise
n_spurious = 20
X_base = X_train.copy()
X_test_base = X_test.copy()

# Pre-generate noise columns so the feature subspace is strictly nested
noise_all_tr = rng.normal(0, 1, (len(X_train), n_spurious - 1))
noise_all_te = rng.normal(0, 1, (len(X_test), n_spurious - 1))

r2_train_history = []
adj_r2_train_history = []
r2_test_history = []

feature_counts = range(1, n_spurious + 1)

for p in feature_counts:
    if p == 1:
        X_cur_tr = X_base
        X_cur_te = X_test_base
    else:
        # Progressively append noise columns
        X_cur_tr = np.hstack([X_base, noise_all_tr[:, :p - 1]])
        X_cur_te = np.hstack([X_test_base, noise_all_te[:, :p - 1]])

    m = LinearRegression().fit(X_cur_tr, y_train)
    pred_tr = m.predict(X_cur_tr)
    pred_te = m.predict(X_cur_te)

    r2_tr = r2(y_train, pred_tr)
    adj_tr = adj_r2(y_train, pred_tr, p=p)
    r2_te = r2(y_test, pred_te)

    r2_train_history.append(r2_tr)
    adj_r2_train_history.append(adj_tr)
    r2_test_history.append(r2_te)

print(f"\n  {'Features (p)':>14} {'Train R²':>12} {'Train Adj R²':>14} {'Test R²':>12}")
print("  " + "-" * 56)
for p in [1, 5, 10, 15, 20]:
    idx = p - 1
    print(f"  {p:>14} {r2_train_history[idx]:>12.4f} {adj_r2_train_history[idx]:>14.4f} {r2_test_history[idx]:>12.4f}")

print("""
  Observation:
  - Training R² strictly rises as meaningless noise features are added.
  - Adjusted R² drops because it penalises complexity that doesn't earn its keep.
  - Test R² plummets as the model overfits to the random noise columns.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Diagnostic Plots
# ═══════════════════════════════════════════════════════════════════════════
residuals = y_test - y_pred

fig, axes = plt.subplots(2, 2, figsize=(15, 12))
fig.suptitle("Day 4 — Evaluation Metrics & Residual Diagnostics", fontsize=15, fontweight="bold")

# Plot 1: Actual vs Predicted with Error Spikes
ax = axes[0, 0]
ax.scatter(y_test, y_pred, color="#4A90D9", edgecolors="black", s=55, alpha=0.85, label="Test samples")
min_val = min(y_test.min(), y_pred.min()) - 5
max_val = max(y_test.max(), y_pred.max()) + 5
ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=1.5, label="Perfect Fit (y = ŷ)")
ax.set_xlabel("Actual y")
ax.set_ylabel("Predicted ŷ")
ax.set_title(f"Actual vs Predicted  (R²={val_r2:.3f}, RMSE={val_rmse:.2f})")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 2: Residuals vs Predicted (Homoscedasticity Check)
ax = axes[0, 1]
ax.scatter(y_pred, residuals, color="#E67E22", edgecolors="black", s=55, alpha=0.85)
ax.axhline(0, color="black", linestyle="--", linewidth=1.2)
ax.axhline(np.std(residuals) * 2, color="grey", linestyle=":", label="±2σ Error Band")
ax.axhline(-np.std(residuals) * 2, color="grey", linestyle=":")
ax.set_xlabel("Predicted ŷ")
ax.set_ylabel("Residual (y - ŷ)")
ax.set_title("Residuals vs Fitted Values (Checking Variance)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 3: Outlier Sensitivity (MAE vs RMSE)
ax = axes[1, 0]
shocks = [r[0] for r in shock_results]
maes = [r[1] for r in shock_results]
rmses = [r[2] for r in shock_results]
ax.plot(shocks, maes, "o-", color="#27AE60", linewidth=2, label="MAE (linear growth)")
ax.plot(shocks, rmses, "s-", color="#E74C3C", linewidth=2, label="RMSE (quadratic penalty)")
ax.set_xlabel("Outlier Shock Magnitude Added to One Sample")
ax.set_ylabel("Metric Value")
ax.set_title("Outlier Impact: MAE vs RMSE")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Plot 4: Spurious Features R² vs Adjusted R²
ax = axes[1, 1]
ax.plot(feature_counts, r2_train_history, "o-", color="#3498DB", linewidth=2, label="Train R² (naive)")
ax.plot(feature_counts, adj_r2_train_history, "^-", color="#9B59B6", linewidth=2, label="Train Adjusted R² (penalised)")
ax.plot(feature_counts, r2_test_history, "x--", color="#E74C3C", linewidth=1.8, label="Test R² (true performance)")
ax.set_xlabel("Number of Features (1 signal + (p-1) pure noise)")
ax.set_ylabel("R² Score")
ax.set_title("R² vs Adjusted R² with Noise Features")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_plot = "day-04/plot.png"
plt.savefig(out_plot, dpi=120, bbox_inches="tight")
print(f"\n  Plot saved -> {out_plot}")

print("\n" + "=" * 65)
print("  DAY 4 DEMO COMPLETE! Read exercises.md for your challenge.")
print("=" * 65)
