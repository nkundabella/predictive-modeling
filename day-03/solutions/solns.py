"""
=======================================================
  DAY 3 — SOLUTIONS
  Gradient Descent
=======================================================
"""

import sys
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Shared dataset (same as demo script)
# ---------------------------------------------------------------------------
rng = np.random.default_rng(42)
n = 80
x_raw = rng.uniform(0, 10, n)
y = 3.5 * x_raw + 10.0 + rng.normal(0, 4, n)
x = (x_raw - x_raw.mean()) / x_raw.std()

X_2d = x.reshape(-1, 1)
X_tr_1d, X_te_1d, y_tr, y_te = train_test_split(X_2d, y, test_size=0.2, random_state=0)
x_tr = X_tr_1d.flatten()
x_te = X_te_1d.flatten()

# House-price dataset (for Q7)
rng2 = np.random.default_rng(0)
n_h = 50
size     = rng2.uniform(40, 200, n_h)
bedrooms = rng2.integers(1, 6, n_h).astype(float)
age      = rng2.uniform(0, 80, n_h)
price    = 1.8 * size + 12.0 * bedrooms - 0.5 * age + rng2.normal(0, 8, n_h)
X_house  = np.column_stack([size, bedrooms, age])
# Standardise features AFTER splitting (Day 2 best practice)
Xh_tr, Xh_te, yh_tr, yh_te = train_test_split(X_house, price, test_size=0.2, random_state=42)
mu  = Xh_tr.mean(axis=0)
sig = Xh_tr.std(axis=0)
Xh_tr_sc = (Xh_tr - mu) / sig
Xh_te_sc = (Xh_te - mu) / sig


# ---------------------------------------------------------------------------
# Q6 — Gradient descent with early stopping
# ---------------------------------------------------------------------------
print("=" * 60)
print("Q6 — Gradient Descent with Early Stopping")
print("=" * 60)


def gradient_descent_early_stop(x, y, alpha=0.05, max_epochs=5000, tol=1e-6, patience=5, w_init=0.0, b_init=0.0):
    w, b = w_init, b_init
    history = []
    no_improve = 0
    prev_loss = float("inf")

    for epoch in range(max_epochs):
        residuals = y - (w * x + b)
        loss = np.mean(residuals ** 2)
        history.append((epoch, loss))

        # Early stopping check
        if prev_loss - loss < tol:
            no_improve += 1
        else:
            no_improve = 0

        if no_improve >= patience:
            print(f"  Early stopping triggered at epoch {epoch}  (loss={loss:.8f})")
            break

        prev_loss = loss
        dw = (-2 / len(y)) * np.dot(residuals, x)
        db = (-2 / len(y)) * np.sum(residuals)
        w -= alpha * dw
        b -= alpha * db

    return w, b, history


w6, b6, hist6 = gradient_descent_early_stop(x_tr, y_tr)
print(f"  Epochs to converge : {len(hist6)}")
print(f"  w={w6:.6f},  b={b6:.6f}")


# ---------------------------------------------------------------------------
# Q7 — Multi-feature gradient descent (vectorised)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q7 — Multi-Feature Gradient Descent")
print("=" * 60)


def gd_multi(X, y, alpha=0.05, epochs=500):
    n, p = X.shape
    w = np.zeros(p)
    b = 0.0
    for _ in range(epochs):
        residuals = y - (X @ w + b)
        dw = (-2 / n) * (X.T @ residuals)
        db = (-2 / n) * np.sum(residuals)
        w -= alpha * dw
        b -= alpha * db
    return w, b


w7, b7 = gd_multi(Xh_tr_sc, yh_tr, alpha=0.05, epochs=500)
sk7 = LinearRegression().fit(Xh_tr_sc, yh_tr)

print(f"\n  {'Feature':<14} {'OLS coef':>12} {'GD coef':>12} {'|Δ|':>10}")
print("  " + "-" * 52)
feature_names = ["Size (m²)", "Bedrooms", "Age (years)"]
for name, ols_c, gd_c in zip(feature_names, sk7.coef_, w7):
    print(f"  {name:<14} {ols_c:>12.4f} {gd_c:>12.4f} {abs(ols_c - gd_c):>10.4f}")
print(f"  {'Intercept':<14} {sk7.intercept_:>12.4f} {b7:>12.4f} {abs(sk7.intercept_ - b7):>10.4f}")

r2_gd7  = r2_score(yh_te, Xh_te_sc @ w7 + b7)
r2_ols7 = r2_score(yh_te, sk7.predict(Xh_te_sc))
print(f"\n  Test R²  — OLS: {r2_ols7:.4f}  |  GD: {r2_gd7:.4f}")


# ---------------------------------------------------------------------------
# Q8 — Different random initialisations
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q8 — Random Initialisations")
print("=" * 60)

rng_init = np.random.default_rng(7)
print(f"\n  {'Init w':>10} {'Init b':>10} {'Final w':>10} {'Final b':>10} {'Final MSE':>12}")
print("  " + "-" * 56)

for _ in range(5):
    w_init = rng_init.normal(0, 10)
    b_init = rng_init.normal(0, 10)
    w_f, b_f, hist_f = gradient_descent_early_stop(x_tr, y_tr, alpha=0.05, tol=1e-8, w_init=w_init, b_init=b_init)
    final_mse = hist_f[-1][1]
    print(f"  {w_init:>10.3f} {b_init:>10.3f} {w_f:>10.6f} {b_f:>10.6f} {final_mse:>12.6f}")

print("""
  Observation: all runs converge to the same (w, b) and loss.
  MSE is a convex quadratic — there is exactly one global minimum.
  Gradient descent always finds it, regardless of starting point.
""")


# ---------------------------------------------------------------------------
# Q9 — Stochastic Gradient Descent vs Batch GD
# ---------------------------------------------------------------------------
print("=" * 60)
print("Q9 — SGD vs Batch GD")
print("=" * 60)


def sgd(x, y, alpha=0.05, n_steps=300):
    """SGD: one random sample per update step."""
    rng_sgd = np.random.default_rng(99)
    w, b = 0.0, 0.0
    history = []
    for step in range(n_steps):
        # Compute full-set MSE for fair comparison of loss values
        loss = np.mean((y - (w * x + b)) ** 2)
        history.append((step, loss))
        # Pick one random sample
        idx = rng_sgd.integers(len(y))
        xi, yi = x[idx], y[idx]
        res = yi - (w * xi + b)
        w += alpha * 2 * res * xi
        b += alpha * 2 * res
    return w, b, history


def batch_gd_300(x, y, alpha=0.05, epochs=300):
    w, b = 0.0, 0.0
    history = []
    for epoch in range(epochs):
        residuals = y - (w * x + b)
        loss = np.mean(residuals ** 2)
        history.append((epoch, loss))
        dw = (-2 / len(y)) * np.dot(residuals, x)
        db = (-2 / len(y)) * np.sum(residuals)
        w -= alpha * dw
        b -= alpha * db
    return w, b, history


w_sgd, b_sgd, hist_sgd   = sgd(x_tr, y_tr, alpha=0.05, n_steps=300)
w_bgd, b_bgd, hist_bgd   = batch_gd_300(x_tr, y_tr, alpha=0.05, epochs=300)

print(f"  Batch GD final MSE : {hist_bgd[-1][1]:.6f}  (w={w_bgd:.4f}, b={b_bgd:.4f})")
print(f"  SGD     final MSE  : {hist_sgd[-1][1]:.6f}  (w={w_sgd:.4f}, b={b_sgd:.4f})")
print("""
  SGD loss curve is noisy (jumps up and down) because each update
  uses only one data point. Batch GD decreases smoothly.
  After 300 "data-touches", batch GD has seen each sample 300 times
  (300 full passes), while SGD has seen 300 individual samples
  (~4 passes). Batch GD converges closer to the minimum here.
""")

# Plot Q9 comparison
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot([h[0] for h in hist_bgd], [h[1] for h in hist_bgd],
        color="#4A90D9", linewidth=2, label="Batch GD")
ax.plot([h[0] for h in hist_sgd], [h[1] for h in hist_sgd],
        color="#E74C3C", linewidth=1, alpha=0.8, label="SGD (1 sample/step)")
ax.set_xlabel("Step / Epoch")
ax.set_ylabel("MSE")
ax.set_title("Q9 — SGD vs Batch GD Loss Curves")
ax.set_yscale("log")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("day-03/solutions/sgd_vs_bgd.png", dpi=100, bbox_inches="tight")
print("  Plot saved -> day-03/solutions/sgd_vs_bgd.png")


# ---------------------------------------------------------------------------
# Q10 — Vectorised update (θ = [w, b])
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("Q10 — Vectorised Parameter Update")
print("=" * 60)


def gd_vectorised(x, y, alpha=0.05, epochs=300):
    """Single numpy line update: θ ← θ - α * ∇L(θ)"""
    n = len(y)
    # Augment x with a column of ones for the bias term
    X_aug = np.column_stack([x, np.ones(n)])  # shape (n, 2)
    theta = np.zeros(2)                        # [w, b]
    history = []
    for epoch in range(epochs):
        residuals = y - X_aug @ theta
        loss = np.mean(residuals ** 2)
        history.append(loss)
        grad = (-2 / n) * (X_aug.T @ residuals)   # shape (2,)
        theta = theta - alpha * grad               # single update line
    return theta[0], theta[1], history


w_v, b_v, hist_v = gd_vectorised(x_tr, y_tr)
_, _, hist_ref   = batch_gd_300(x_tr, y_tr)

print(f"  Vectorised:    w={w_v:.6f}, b={b_v:.6f}, final MSE={hist_v[-1]:.6f}")
print(f"  Non-vectorised: w={w_bgd:.6f}, b={b_bgd:.6f}, final MSE={hist_ref[-1][1]:.6f}")
max_diff = max(abs(hist_v[i] - hist_ref[i][1]) for i in range(len(hist_v)))
print(f"  Max loss difference across all epochs: {max_diff:.2e}")
if max_diff < 1e-10:
    print("  ✓ Both implementations produce identical results.")

print("\n" + "=" * 60)
print("  DAY 3 SOLUTIONS COMPLETE.")
print("=" * 60)
