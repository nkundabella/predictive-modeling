"""
=======================================================
  DAY 3 — Gradient Descent
  Topic : Iterative parameter learning via gradient descent.
          Implementing batch GD from scratch on MSE, comparing
          to OLS, and exploring how learning rate affects convergence.
  Run   : python day-03/gradient_descent_demo.py
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
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Dataset: synthetic single-feature regression
#           We keep one feature so we can visualise the 2-D loss surface.
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1 — DATASET")
print("=" * 60)

rng = np.random.default_rng(42)
n = 80

# True relationship: y = 3.5 * x + 10  + noise
x_raw = rng.uniform(0, 10, n)
y = 3.5 * x_raw + 10.0 + rng.normal(0, 4, n)

# Standardise x (important for gradient descent — see notes)
x_mean, x_std = x_raw.mean(), x_raw.std()
x = (x_raw - x_mean) / x_std          # zero-mean, unit-variance

print(f"\n  n = {n} observations")
print(f"  x: raw range [{x_raw.min():.1f}, {x_raw.max():.1f}]  →  scaled mean={x.mean():.4f}, std={x.std():.4f}")
print(f"  y: range [{y.min():.1f}, {y.max():.1f}], mean={y.mean():.2f}")

# Train/test split (80/20) — using Day 2 best practice
X_2d = x.reshape(-1, 1)
X_tr, X_te, y_tr, y_te = train_test_split(X_2d, y, test_size=0.2, random_state=0)
x_tr = X_tr.flatten()
x_te = X_te.flatten()
print(f"\n  Training : {len(x_tr)} rows  |  Test : {len(x_te)} rows")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — Batch Gradient Descent from scratch
#
#  Model   : y_hat = w * x + b
#  Loss    : MSE = (1/n) * sum( (y_i - y_hat_i)^2 )
#  Gradient:
#    ∂MSE/∂w = (-2/n) * sum( (y_i - y_hat_i) * x_i )
#    ∂MSE/∂b = (-2/n) * sum( (y_i - y_hat_i) )
#  Update  : w ← w − α * ∂MSE/∂w
#            b ← b − α * ∂MSE/∂b
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2 — BATCH GRADIENT DESCENT (from scratch)")
print("=" * 60)


def mse(w, b, x, y):
    """Mean Squared Error for model y_hat = w*x + b."""
    residuals = y - (w * x + b)
    return np.mean(residuals ** 2)


def gradients(w, b, x, y):
    """Return (∂MSE/∂w, ∂MSE/∂b)."""
    n = len(y)
    residuals = y - (w * x + b)           # shape (n,)
    dw = (-2 / n) * np.dot(residuals, x)  # scalar
    db = (-2 / n) * np.sum(residuals)     # scalar
    return dw, db


def gradient_descent(x, y, alpha=0.05, epochs=200, w_init=0.0, b_init=0.0):
    """
    Run batch gradient descent.

    Returns
    -------
    w, b         : final learned parameters
    history      : list of (epoch, loss, w, b)
    """
    w, b = w_init, b_init
    history = []

    for epoch in range(epochs):
        loss = mse(w, b, x, y)
        history.append((epoch, loss, w, b))

        dw, db = gradients(w, b, x, y)
        w -= alpha * dw
        b -= alpha * db

    # record the final state
    history.append((epochs, mse(w, b, x, y), w, b))
    return w, b, history


ALPHA  = 0.05
EPOCHS = 300

w_gd, b_gd, history = gradient_descent(x_tr, y_tr, alpha=ALPHA, epochs=EPOCHS)

losses = [h[1] for h in history]
epochs_list = [h[0] for h in history]

print(f"\n  Learning rate : {ALPHA}")
print(f"  Epochs        : {EPOCHS}")
print(f"\n  Initial loss  : {losses[0]:.4f}")
print(f"  Final loss    : {losses[-1]:.4f}")
print(f"\n  Learned parameters:")
print(f"    w (slope)    : {w_gd:.6f}")
print(f"    b (intercept): {b_gd:.6f}")

# Print a sample of the convergence
print(f"\n  {'Epoch':>6} | {'MSE Loss':>12}")
print("  " + "-" * 22)
for epoch_idx in [0, 10, 25, 50, 100, 200, EPOCHS]:
    row = history[min(epoch_idx, len(history) - 1)]
    print(f"  {row[0]:>6} | {row[1]:>12.6f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — Comparison: OLS (sklearn) vs Gradient Descent
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3 — OLS (sklearn) vs GRADIENT DESCENT comparison")
print("=" * 60)

sk_model = LinearRegression()
sk_model.fit(X_tr, y_tr)
w_ols = sk_model.coef_[0]
b_ols = sk_model.intercept_

# Evaluate both on test set
y_pred_gd  = w_gd  * x_te + b_gd
y_pred_ols = sk_model.predict(X_te)

r2_gd   = r2_score(y_te, y_pred_gd)
r2_ols  = r2_score(y_te, y_pred_ols)
mse_gd  = mean_squared_error(y_te, y_pred_gd)
mse_ols = mean_squared_error(y_te, y_pred_ols)

print(f"\n  {'Method':<22} {'w':>10} {'b':>10} {'Test R²':>10} {'Test MSE':>12}")
print("  " + "-" * 66)
print(f"  {'OLS (exact)':<22} {w_ols:>10.6f} {b_ols:>10.6f} {r2_ols:>10.4f} {mse_ols:>12.4f}")
print(f"  {'Gradient Descent':<22} {w_gd:>10.6f} {b_gd:>10.6f} {r2_gd:>10.4f} {mse_gd:>12.4f}")

param_diff_w = abs(w_gd - w_ols)
param_diff_b = abs(b_gd - b_ols)
print(f"\n  Parameter difference:  |Δw| = {param_diff_w:.6f},  |Δb| = {param_diff_b:.6f}")
if param_diff_w < 0.01 and param_diff_b < 0.01:
    print("  ✓ GD converged to nearly the same solution as OLS.")
else:
    print("  ⚠ GD has not fully converged — try more epochs or a smaller learning rate.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Effect of learning rate on convergence
#           Three α values: too small, good, too large
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4 — LEARNING RATE SENSITIVITY")
print("=" * 60)

alphas = {"Too small (α=0.001)": 0.001, "Good (α=0.05)": 0.05, "Too large (α=1.05)": 1.05}
lr_histories = {}

for label, alpha in alphas.items():
    try:
        _, _, hist = gradient_descent(x_tr, y_tr, alpha=alpha, epochs=300)
        final_loss = hist[-1][1]
        # cap diverged runs
        if not np.isfinite(final_loss):
            final_loss = float("nan")
        lr_histories[label] = ([h[0] for h in hist], [h[1] for h in hist])
        print(f"\n  {label}")
        print(f"    Final MSE : {final_loss:.4f}" if np.isfinite(final_loss) else "    ⚠ Loss diverged (NaN/Inf)")
    except Exception as e:
        print(f"\n  {label}: ERROR — {e}")
        lr_histories[label] = ([], [])


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Loss surface visualisation (contour plot)
#           Plot MSE as a function of (w, b) and overlay the GD trajectory
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 5 — LOSS SURFACE AND GD TRAJECTORY")
print("=" * 60)

# Build a grid of (w, b) values around the true minimum
w_grid = np.linspace(w_ols - 5, w_ols + 5, 300)
b_grid = np.linspace(b_ols - 8, b_ols + 8, 300)
W, B   = np.meshgrid(w_grid, b_grid)

# MSE on the training set for each (w, b) pair
# Vectorised: broadcast over the grid
# residuals: shape (300, 300, n_tr)
residuals_grid = y_tr[np.newaxis, np.newaxis, :] - (
    W[:, :, np.newaxis] * x_tr[np.newaxis, np.newaxis, :]
    + B[:, :, np.newaxis]
)
Z = np.mean(residuals_grid ** 2, axis=2)

# Extract trajectory from history (sub-sampled)
traj_w = np.array([h[2] for h in history])
traj_b = np.array([h[3] for h in history])

print(f"\n  Loss surface computed over {W.shape[0]}×{W.shape[1]} grid.")
print(f"  OLS minimum at   w={w_ols:.4f}, b={b_ols:.4f}, MSE={mse(w_ols, b_ols, x_tr, y_tr):.4f}")
print(f"  GD trajectory: {len(traj_w)} recorded points.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Plots
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle("Day 3 — Gradient Descent", fontsize=14, fontweight="bold")

# ── Plot 1: Data + fitted lines ──────────────────────────────────────────
ax = axes[0, 0]
x_line = np.linspace(x.min() - 0.2, x.max() + 0.2, 200)
ax.scatter(x_tr, y_tr, color="#4A90D9", s=40, alpha=0.8, label="Train", zorder=4)
ax.scatter(x_te, y_te, color="#E74C3C", s=50, marker="D", alpha=0.9, label="Test", zorder=5)
ax.plot(x_line, w_ols * x_line + b_ols, color="#27AE60", linewidth=2,
        linestyle="--", label=f"OLS  (w={w_ols:.2f}, b={b_ols:.2f})", zorder=6)
ax.plot(x_line, w_gd * x_line + b_gd,  color="#E67E22", linewidth=2,
        label=f"GD   (w={w_gd:.2f}, b={b_gd:.2f})", zorder=6)
ax.set_xlabel("x (standardised)")
ax.set_ylabel("y")
ax.set_title("Data + Fitted Lines: OLS vs GD")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 2: Loss curve for the main GD run ───────────────────────────────
ax = axes[0, 1]
ax.plot(epochs_list, losses, color="#4A90D9", linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (training)")
ax.set_title(f"Loss Curve  (α={ALPHA})")
ax.set_yscale("log")
ax.grid(True, alpha=0.3)
ax.annotate(f"Final: {losses[-1]:.4f}",
            xy=(epochs_list[-1], losses[-1]),
            xytext=(-80, 15), textcoords="offset points",
            fontsize=9, arrowprops=dict(arrowstyle="->", color="grey"))

# ── Plot 3: Learning rate comparison ─────────────────────────────────────
ax = axes[1, 0]
colors_lr = {"Too small (α=0.001)": "#E74C3C", "Good (α=0.05)": "#27AE60", "Too large (α=1.05)": "#9B59B6"}
for label, (ep, ls) in lr_histories.items():
    ls_clean = [v if np.isfinite(v) else None for v in ls]
    ep_plot  = [e for e, v in zip(ep, ls_clean) if v is not None]
    ls_plot  = [v for v in ls_clean if v is not None]
    if ep_plot:
        ax.plot(ep_plot[:150], ls_plot[:150], label=label,
                color=colors_lr[label], linewidth=1.8)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE (training, log scale)")
ax.set_title("Learning Rate Sensitivity")
ax.set_yscale("log")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ── Plot 4: Loss surface contour + GD trajectory ─────────────────────────
ax = axes[1, 1]
levels = np.percentile(Z[np.isfinite(Z)], np.linspace(0, 85, 30))
contour = ax.contourf(W, B, Z, levels=levels, cmap="Blues_r", alpha=0.85)
ax.contour(W, B, Z, levels=levels, colors="white", linewidths=0.3, alpha=0.5)
plt.colorbar(contour, ax=ax, label="MSE", shrink=0.85)

# Sub-sample trajectory for clarity
step = max(1, len(traj_w) // 40)
ax.plot(traj_w[::step], traj_b[::step], "o-",
        color="#E74C3C", markersize=4, linewidth=1.5,
        label="GD trajectory", zorder=5)
ax.plot(traj_w[0], traj_b[0], "s", color="#E67E22", markersize=9,
        zorder=6, label="Start")
ax.plot(w_ols, b_ols, "*", color="#27AE60", markersize=14,
        zorder=7, label="OLS minimum")
ax.set_xlabel("w (slope)")
ax.set_ylabel("b (intercept)")
ax.set_title("Loss Surface + GD Trajectory")
ax.legend(fontsize=8)

plt.tight_layout()
out_path = "day-03/plot.png"
plt.savefig(out_path, dpi=120, bbox_inches="tight")
print(f"\n  Plot saved -> {out_path}")

print("\n" + "=" * 60)
print("  DAY 3 COMPLETE! Read exercises.md for your challenge.")
print("=" * 60)
