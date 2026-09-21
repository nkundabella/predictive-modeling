"""
=======================================================
  DAY 2 — Features, Labels, Train/Test Split, Data Leakage
  Topic : Why you can't evaluate on training data, and how
          preprocessing before splitting causes leakage.
  Run   : python day-02/train_test_split_demo.py
=======================================================
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Dataset: 50 synthetic house price observations
#           Features: size (m²), bedrooms, age (years)
#           Label   : price (£k)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1 — DATASET: HOUSE PRICES (synthetic)")
print("=" * 60)

rng = np.random.default_rng(0)
n = 50

size     = rng.uniform(40, 200, n)          # floor area in m²
bedrooms = rng.integers(1, 6, n).astype(float)
age      = rng.uniform(0, 80, n)            # years old

# True relationship (with some noise)
price = 1.8 * size + 12.0 * bedrooms - 0.5 * age + rng.normal(0, 8, n)

# Stack into feature matrix X and label vector y
X = np.column_stack([size, bedrooms, age])
y = price

feature_names = ["Size (m²)", "Bedrooms", "Age (years)"]

print(f"\n  Dataset shape: X={X.shape}, y={y.shape}")
print(f"  Features      : {feature_names}")
print(f"  Label         : Price (£k)")
print(f"\n  First 5 rows:")
print(f"  {'Size':>8} {'Beds':>6} {'Age':>6} {'Price £k':>10}")
print("  " + "-" * 36)
for i in range(5):
    print(f"  {X[i,0]:>8.1f} {X[i,1]:>6.0f} {X[i,2]:>6.1f} {y[i]:>10.1f}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 2 — The wrong way: train and evaluate on the same data
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 2 — WRONG WAY: fit and score on the same data")
print("=" * 60)

model_wrong = LinearRegression()
model_wrong.fit(X, y)
y_pred_all = model_wrong.predict(X)

r2_wrong = r2_score(y, y_pred_all)
rmse_wrong = np.sqrt(mean_squared_error(y, y_pred_all))

print(f"\n  R² (train=test) : {r2_wrong:.4f}")
print(f"  RMSE            : {rmse_wrong:.2f} £k")
print("\n  This score looks great — but it is meaningless.")
print("  The model already saw every answer before being tested.")


# ═══════════════════════════════════════════════════════════════════════════
# PART 3 — The correct way: train/test split
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 3 — CORRECT WAY: 80/20 train/test split")
print("=" * 60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n  Training rows : {len(X_train)}")
print(f"  Test rows     : {len(X_test)}")

model = LinearRegression()
model.fit(X_train, y_train)

train_r2   = r2_score(y_train, model.predict(X_train))
test_r2    = r2_score(y_test,  model.predict(X_test))
train_rmse = np.sqrt(mean_squared_error(y_train, model.predict(X_train)))
test_rmse  = np.sqrt(mean_squared_error(y_test,  model.predict(X_test)))

print(f"\n  {'Metric':<12} {'Train':>10} {'Test':>10}")
print("  " + "-" * 34)
print(f"  {'R²':<12} {train_r2:>10.4f} {test_r2:>10.4f}")
print(f"  {'RMSE (£k)':<12} {train_rmse:>10.2f} {test_rmse:>10.2f}")

print("\n  Learned coefficients:")
for name, coef in zip(feature_names, model.coef_):
    print(f"    {name:<14}: {coef:+.4f}")
print(f"    {'Intercept':<14}: {model.intercept_:+.4f}")

gap = train_r2 - test_r2
if gap < 0.05:
    diagnosis = "Good fit — train and test R² are close."
elif gap < 0.15:
    diagnosis = "Mild overfitting — slight gap between train and test."
else:
    diagnosis = "Overfitting — model has memorised the training data."
print(f"\n  Diagnosis: {diagnosis}")


# ═══════════════════════════════════════════════════════════════════════════
# PART 4 — Data leakage: standardising before splitting
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART 4 — DATA LEAKAGE: scaling before vs after split")
print("=" * 60)

# --- LEAKY: fit scaler on ALL data, then split ---
scaler_leaky = StandardScaler()
X_scaled_all = scaler_leaky.fit_transform(X)            # uses test rows!
X_tr_leak, X_te_leak, y_tr_leak, y_te_leak = train_test_split(
    X_scaled_all, y, test_size=0.2, random_state=42
)
model_leaky = LinearRegression().fit(X_tr_leak, y_tr_leak)
r2_leaky = r2_score(y_te_leak, model_leaky.predict(X_te_leak))

# --- CORRECT: split first, then fit scaler on train only ---
scaler_correct = StandardScaler()
scaler_correct.fit(X_train)                             # train rows only
X_tr_sc = scaler_correct.transform(X_train)
X_te_sc = scaler_correct.transform(X_test)             # same stats!
model_correct = LinearRegression().fit(X_tr_sc, y_train)
r2_correct = r2_score(y_test, model_correct.predict(X_te_sc))

print(f"\n  Leaky pipeline R² (test)   : {r2_leaky:.6f}")
print(f"  Correct pipeline R² (test) : {r2_correct:.6f}")
print(f"  Difference                 : {abs(r2_leaky - r2_correct):.6f}")
print("""
  On this clean dataset the difference is small, but on real-world
  data with heavy preprocessing (imputation, encoding, scaling) leakage
  can inflate R² by 0.05–0.30 — making a mediocre model look excellent.

  The fix is always: SPLIT FIRST, then fit all preprocessing on X_train.
  Use sklearn Pipelines (Day 15) to enforce this automatically.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 5 — Effect of random_state on test score
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 5 — How much does the split matter?")
print("         Running 20 different random splits")
print("=" * 60)

seeds = range(20)
test_scores = []
for seed in seeds:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    m = LinearRegression().fit(Xtr, ytr)
    test_scores.append(r2_score(yte, m.predict(Xte)))

print(f"\n  Test R² across 20 splits:")
print(f"    min  : {min(test_scores):.4f}")
print(f"    max  : {max(test_scores):.4f}")
print(f"    mean : {np.mean(test_scores):.4f}")
print(f"    std  : {np.std(test_scores):.4f}")
print("""
  The spread shows that a single train/test split can be unlucky.
  Day 14 covers k-fold cross-validation, which averages over many
  splits to give a more stable estimate of generalisation performance.
""")


# ═══════════════════════════════════════════════════════════════════════════
# PART 6 — Plots
# ═══════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Day 2 — Train/Test Split & Leakage Demo", fontsize=13, fontweight="bold")

# --- Plot 1: train/test actual vs predicted ---
ax = axes[0]
y_pred_train = model.predict(X_train)
y_pred_test  = model.predict(X_test)
ax.scatter(y_train, y_pred_train, color="#4A90D9", s=50, alpha=0.8,
           label=f"Train (R²={train_r2:.2f})", zorder=4)
ax.scatter(y_test, y_pred_test, color="#E74C3C", s=60, marker="D", alpha=0.9,
           label=f"Test (R²={test_r2:.2f})", zorder=5)
lims = [y.min() - 5, y.max() + 5]
ax.plot(lims, lims, "k--", linewidth=1, label="Perfect prediction")
ax.set_xlabel("Actual Price (£k)")
ax.set_ylabel("Predicted Price (£k)")
ax.set_title("Actual vs Predicted")
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# --- Plot 2: test R² across different random seeds ---
ax = axes[1]
ax.bar(range(20), test_scores, color="#27AE60", alpha=0.75, edgecolor="black", linewidth=0.5)
ax.axhline(np.mean(test_scores), color="#E74C3C", linewidth=1.5,
           linestyle="--", label=f"Mean={np.mean(test_scores):.3f}")
ax.set_xlabel("random_state")
ax.set_ylabel("Test R²")
ax.set_title("Test R² vs Random Seed\n(20 different splits)")
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3, axis="y")

# --- Plot 3: leaky vs correct pipeline ---
ax = axes[2]
labels = ["Leaky\n(scale before split)", "Correct\n(scale after split)"]
scores = [r2_leaky, r2_correct]
colors = ["#E74C3C", "#27AE60"]
bars = ax.bar(labels, scores, color=colors, alpha=0.82, edgecolor="black", linewidth=0.7)
for bar, score in zip(bars, scores):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{score:.4f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
ax.set_ylabel("Test R²")
ax.set_title("Leaky vs Correct Preprocessing Pipeline")
ax.set_ylim(0, 1.1)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("day-02/plot.png", dpi=120, bbox_inches="tight")
print("  Plot saved -> day-02/plot.png")

print("\n" + "=" * 60)
print("  DAY 2 COMPLETE! Read exercises.md for your challenge.")
print("=" * 60)
