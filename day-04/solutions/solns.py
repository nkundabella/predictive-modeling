"""
=======================================================
  DAY 4 — SOLUTIONS: Evaluation Metrics in Regression
  Topics : MAE, MSE, RMSE, R², Adjusted R²,
           Outlier sensitivity, Negative R²,
           Heteroscedasticity, and Huber Loss.
  Run    : python day-04/solutions/solns.py
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
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════
# Q6 — Standalone Regression Report Function
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 65)
print("Q6 — STANDALONE REGRESSION EVALUATION REPORT")
print("=" * 65)


def regression_report(y_true, y_pred, p=1, verbose=True):
    """
    Computes comprehensive regression evaluation metrics.
    y_true  : ground truth target values
    y_pred  : model predictions
    p       : number of predictors (features) excluding intercept
    verbose : whether to print formatted table
    """
    y_true = np.asarray(y_true).flatten()
    y_pred = np.asarray(y_pred).flatten()
    n = len(y_true)

    residuals = y_true - y_pred
    mae_val = np.mean(np.abs(residuals))
    mse_val = np.mean(residuals ** 2)
    rmse_val = np.sqrt(mse_val)

    tss = np.sum((y_true - np.mean(y_true)) ** 2)
    rss = np.sum(residuals ** 2)
    r2_val = 1.0 - (rss / tss) if tss > 0 else 0.0

    if n > p + 1:
        adj_r2_val = 1.0 - ((1.0 - r2_val) * (n - 1) / (n - p - 1))
    else:
        adj_r2_val = np.nan

    rmse_mae_ratio = rmse_val / mae_val if mae_val > 0 else 1.0

    metrics = {
        "MAE": mae_val,
        "MSE": mse_val,
        "RMSE": rmse_val,
        "R2": r2_val,
        "Adj_R2": adj_r2_val,
        "RMSE_MAE_Ratio": rmse_mae_ratio
    }

    if verbose:
        print(f"\n  {'Metric':<20} {'Value':>15}")
        print("  " + "-" * 37)
        print(f"  {'MAE':<20} {mae_val:>15.4f}")
        print(f"  {'MSE':<20} {mse_val:>15.4f}")
        print(f"  {'RMSE':<20} {rmse_val:>15.4f}")
        print(f"  {'R²':<20} {r2_val:>15.4f}")
        print(f"  {'Adjusted R² (p=' + str(p) + ')':<20} {adj_r2_val:>15.4f}")
        print(f"  {'RMSE / MAE Ratio':<20} {rmse_mae_ratio:>15.4f}")

    return metrics


# Quick verification
rng = np.random.default_rng(42)
y_synth_true = rng.uniform(20, 100, 50)
y_synth_pred = y_synth_true + rng.normal(0, 5, 50)
rep = regression_report(y_synth_true, y_synth_pred, p=2, verbose=True)

# Verification assertions against sklearn
assert abs(rep["MAE"] - mean_absolute_error(y_synth_true, y_synth_pred)) < 1e-9
assert abs(rep["MSE"] - mean_squared_error(y_synth_true, y_synth_pred)) < 1e-9
assert abs(rep["RMSE"] - root_mean_squared_error(y_synth_true, y_synth_pred)) < 1e-9
assert abs(rep["R2"] - r2_score(y_synth_true, y_synth_pred)) < 1e-9
print("\n  ✓ Verified: Custom metrics match sklearn exactly.")


# ═══════════════════════════════════════════════════════════════════════════
# Q7 — Outlier Stress Test: MAE vs RMSE vs R²
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Q7 — OUTLIER STRESS TEST (k OUTLIERS)")
print("=" * 65)

n_q7 = 100
x_q7 = rng.uniform(0, 20, n_q7)
y_q7 = 4.0 * x_q7 + 10.0 + rng.normal(0, 4, n_q7)

X_q7 = x_q7.reshape(-1, 1)
Xtr_7, Xte_7, ytr_7, yte_7 = train_test_split(X_q7, y_q7, test_size=0.3, random_state=42)
m7 = LinearRegression().fit(Xtr_7, ytr_7)
yp_clean = m7.predict(Xte_7)

k_values = [0, 1, 2, 5, 10]
q7_results = []

print(f"\n  {'k outliers':>12} {'MAE':>12} {'RMSE':>12} {'R²':>12} {'RMSE/MAE':>12}")
print("  " + "-" * 54)

for k in k_values:
    yte_corrupt = yte_7.copy()
    if k > 0:
        # Corrupt first k test points by adding an extreme error shock of +100
        yte_corrupt[:k] += 100.0

    k_mae = mean_absolute_error(yte_corrupt, yp_clean)
    k_rmse = root_mean_squared_error(yte_corrupt, yp_clean)
    k_r2 = r2_score(yte_corrupt, yp_clean)
    q7_results.append((k, k_mae, k_rmse, k_r2, k_rmse / k_mae))
    print(f"  {k:>12} {k_mae:>12.3f} {k_rmse:>12.3f} {k_r2:>12.4f} {k_rmse/k_mae:>12.3f}")

# Plot Q7 results
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(k_values, [r[1] for r in q7_results], "o-", color="#27AE60", linewidth=2, label="MAE")
ax.plot(k_values, [r[2] for r in q7_results], "s-", color="#E74C3C", linewidth=2, label="RMSE")
ax.set_xlabel("Number of Corrupted Test Points (k)")
ax.set_ylabel("Metric Value")
ax.set_title("Q7 — Outlier Sensitivity: MAE vs RMSE")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day-04/solutions/ex7_outliers.png", dpi=100)
print("  Plot saved -> day-04/solutions/ex7_outliers.png")


# ═══════════════════════════════════════════════════════════════════════════
# Q8 — Negative R² on Test Data
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Q8 — DEMONSTRATING NEGATIVE R² ON TEST DATA")
print("=" * 65)

# Train on a restricted domain with non-linear true relation
# y = x^2, but we fit a linear model on x in [0, 4]
x_train_8 = np.linspace(0, 4, 30).reshape(-1, 1)
y_train_8 = (x_train_8.flatten() ** 2) + rng.normal(0, 0.5, 30)
m8 = LinearRegression().fit(x_train_8, y_train_8)

# Evaluate on extrapolated out-of-distribution domain x in [6, 10]
x_test_8 = np.linspace(6, 10, 20).reshape(-1, 1)
y_test_8 = (x_test_8.flatten() ** 2) + rng.normal(0, 0.5, 20)
yp_8 = m8.predict(x_test_8)

r2_neg = r2_score(y_test_8, yp_8)
rss_neg = np.sum((y_test_8 - yp_8) ** 2)
tss_neg = np.sum((y_test_8 - np.mean(y_test_8)) ** 2)

print(f"  Test RSS (model errors)         : {rss_neg:.2f}")
print(f"  Test TSS (naive mean errors)    : {tss_neg:.2f}")
print(f"  Test R² = 1 - (RSS / TSS)       : {r2_neg:.4f}")
print(f"  Negative R² confirmed: {r2_neg < 0}")
print("""
  Why is R² negative?
  RSS measures the model's total squared error. TSS measures the error if you
  simply guessed the test set's mean (y_bar_test) for every sample.
  When RSS > TSS, the model's predictions are worse than guessing a flat horizontal
  line at y_bar_test! Thus, R² = 1 - (RSS/TSS) becomes strictly negative.
""")

fig, ax = plt.subplots(figsize=(8, 5))
ax.scatter(x_test_8, y_test_8, color="#2980B9", s=45, label="Test Points (y_test)")
ax.plot(x_test_8, yp_8, "r--", linewidth=2, label=f"Model Predictions (R²={r2_neg:.2f})")
ax.axhline(np.mean(y_test_8), color="#27AE60", linestyle=":", linewidth=2, label="Naive Mean (ȳ_test, R²=0)")
ax.set_xlabel("x (extrapolated test domain)")
ax.set_ylabel("y")
ax.set_title("Q8 — When Models Perform Worse than Naive Mean (Negative R²)")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day-04/solutions/ex8_negative_r2.png", dpi=100)
print("  Plot saved -> day-04/solutions/ex8_negative_r2.png")


# ═══════════════════════════════════════════════════════════════════════════
# Q9 — Heteroscedasticity Diagnostic Tool
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Q9 — HETEROSCEDASTICITY DIAGNOSTIC TOOL")
print("=" * 65)


def check_heteroscedasticity(y_pred, residuals, alpha=0.05):
    """
    Tests whether residual variance correlates with fitted predictions.
    Uses Spearman rank correlation between |residuals| and y_pred.
    """
    abs_res = np.abs(residuals)
    corr, p_value = spearmanr(y_pred, abs_res)
    is_hetero = p_value < alpha
    return corr, p_value, is_hetero


# Generate Homoscedastic dataset (constant noise variance)
x_homo = rng.uniform(10, 100, 100)
y_homo = 2.5 * x_homo + 20.0 + rng.normal(0, 15, 100)
m_homo = LinearRegression().fit(x_homo.reshape(-1, 1), y_homo)
pred_homo = m_homo.predict(x_homo.reshape(-1, 1))
res_homo = y_homo - pred_homo

# Generate Heteroscedastic dataset (noise scales proportionally with x)
x_hetero = rng.uniform(10, 100, 100)
y_hetero = 2.5 * x_hetero + 20.0 + rng.normal(0, 0.4 * x_hetero, 100)
m_hetero = LinearRegression().fit(x_hetero.reshape(-1, 1), y_hetero)
pred_hetero = m_hetero.predict(x_hetero.reshape(-1, 1))
res_hetero = y_hetero - pred_hetero

corr_h, p_h, flag_h = check_heteroscedasticity(pred_homo, res_homo)
corr_het, p_het, flag_het = check_heteroscedasticity(pred_hetero, res_hetero)

print(f"\n  Homoscedastic data   : Spearman r={corr_h:+.3f}, p={p_h:.4f} -> Heteroscedastic? {flag_h}")
print(f"  Heteroscedastic data : Spearman r={corr_het:+.3f}, p={p_het:.4e} -> Heteroscedastic? {flag_het}")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
axes[0].scatter(pred_homo, res_homo, color="#3498DB", alpha=0.8, edgecolors="black")
axes[0].axhline(0, color="red", linestyle="--")
axes[0].set_title(f"Homoscedastic: Constant Spread (p={p_h:.3f})")
axes[0].set_xlabel("Fitted Values (ŷ)")
axes[0].set_ylabel("Residuals")
axes[0].grid(True, alpha=0.3)

axes[1].scatter(pred_hetero, res_hetero, color="#E67E22", alpha=0.8, edgecolors="black")
axes[1].axhline(0, color="red", linestyle="--")
axes[1].set_title(f"Heteroscedastic: Fan Spread (p={p_het:.2e})")
axes[1].set_xlabel("Fitted Values (ŷ)")
axes[1].set_ylabel("Residuals")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("day-04/solutions/ex9_heteroscedasticity.png", dpi=100)
print("  Plot saved -> day-04/solutions/ex9_heteroscedasticity.png")


# ═══════════════════════════════════════════════════════════════════════════
# Q10 — Huber Loss Implementation & Comparison
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("Q10 — HUBER LOSS IMPLEMENTATION")
print("=" * 65)


def huber_loss(y_true, y_pred, delta=1.35):
    """
    Huber Loss: quadratic for |error| <= delta, linear for |error| > delta.
    Smooth and differentiable everywhere, yet robust to large outliers.
    """
    error = np.asarray(y_true) - np.asarray(y_pred)
    abs_error = np.abs(error)
    quadratic = np.minimum(abs_error, delta)
    linear = abs_error - quadratic
    loss = 0.5 * (quadratic ** 2) + delta * linear
    return np.mean(loss)


# Clean dataset evaluation
y_true_clean = yte_7.copy()
yp_clean_eval = yp_clean.copy()

# Outlier dataset evaluation (add +150 to 3 samples)
y_true_outliers = yte_7.copy()
y_true_outliers[:3] += 150.0

print(f"\n  {'Dataset':<18} {'MAE':>12} {'RMSE':>12} {'Huber (δ=1.35)':>18}")
print("  " + "-" * 52)
for name, yt in [("Clean", y_true_clean), ("With Outliers", y_true_outliers)]:
    mae_v = mean_absolute_error(yt, yp_clean_eval)
    rmse_v = root_mean_squared_error(yt, yp_clean_eval)
    huber_v = huber_loss(yt, yp_clean_eval, delta=1.35)
    print(f"  {name:<18} {mae_v:>12.3f} {rmse_v:>12.3f} {huber_v:>18.3f}")

print("""
  Huber Loss Advantage:
  - For small errors (|e| <= delta), Huber is quadratic (smooth gradient, easy optimization).
  - For large errors (|e| > delta), Huber transitions to linear (penalty grows linearly,
    preventing outliers from dominating parameter learning).
  - Delta=1.35 achieves 95% statistical efficiency of normal distribution while guarding
    against contamination.
""")

print("\n" + "=" * 65)
print("  DAY 4 SOLUTIONS COMPLETE!")
print("=" * 65)
