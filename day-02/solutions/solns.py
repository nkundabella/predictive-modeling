"""
=======================================================
  DAY 2 — Solutions to exercises
  Run   : python day-02/solutions/solns.py
=======================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler

# ── Rebuild the same dataset used in train_test_split_demo.py ───────────────
rng = np.random.default_rng(0)
n = 50
size     = rng.uniform(40, 200, n)
bedrooms = rng.integers(1, 6, n).astype(float)
age      = rng.uniform(0, 80, n)
price    = 1.8 * size + 12.0 * bedrooms - 0.5 * age + rng.normal(0, 8, n)
X = np.column_stack([size, bedrooms, age])
y = price


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 6 — Effect of test_size
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXERCISE 6 — test_size effect on test R²")
print("=" * 60)

for ts in [0.1, 0.2, 0.4]:
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=ts, random_state=42)
    m = LinearRegression().fit(Xtr, ytr)
    print(f"  test_size={ts:.1f}  train={len(Xtr):>3}  test={len(Xte):>3}  "
          f"test R²={r2_score(yte, m.predict(Xte)):.4f}")

print("""
  As test_size grows the training set shrinks, so the model is fitted on
  fewer examples. With a very small training set (ts=0.9) performance can
  degrade substantially. The test score becomes more reliable (less variance)
  as test size grows, but at the cost of a weaker model.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 7 — Leaky mean-centering
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXERCISE 7 — Leaky vs correct mean-centering")
print("=" * 60)

# Leaky: centre using full-dataset means before splitting
X_centered_leaky = X - X.mean(axis=0)
Xtr_l, Xte_l, ytr_l, yte_l = train_test_split(
    X_centered_leaky, y, test_size=0.2, random_state=42)
r2_leaky = r2_score(yte_l, LinearRegression().fit(Xtr_l, ytr_l).predict(Xte_l))

# Correct: split first, compute means on train only
Xtr_c, Xte_c, ytr_c, yte_c = train_test_split(X, y, test_size=0.2, random_state=42)
train_means = Xtr_c.mean(axis=0)
r2_correct  = r2_score(yte_c,
    LinearRegression().fit(Xtr_c - train_means, ytr_c)
                       .predict(Xte_c - train_means))

print(f"  Leaky R²   : {r2_leaky:.6f}")
print(f"  Correct R² : {r2_correct:.6f}")
print(f"  Difference : {abs(r2_leaky - r2_correct):.6f}")
print("""
  Mean-centering leaks the test-set means into the training pipeline.
  For centering alone the effect is small on clean data. The impact
  grows with operations that "learn" more from the data — e.g. PCA,
  clipping outliers, or imputing missing values.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 8 — Leaky feature (price + noise)
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXERCISE 8 — Leaky feature: price_plus_noise")
print("=" * 60)

noise        = rng.normal(0, 1, n)
price_noisy  = price + noise
X_leaky      = np.column_stack([size, bedrooms, age, price_noisy])

Xtr_lk, Xte_lk, ytr_lk, yte_lk = train_test_split(
    X_leaky, y, test_size=0.2, random_state=42)
m_lk = LinearRegression().fit(Xtr_lk, ytr_lk)
print(f"  WITH leaky feature — train R²: {r2_score(ytr_lk, m_lk.predict(Xtr_lk)):.4f}  "
      f"test R²: {r2_score(yte_lk, m_lk.predict(Xte_lk)):.4f}")

Xtr_cl, Xte_cl, ytr_cl, yte_cl = train_test_split(
    X, y, test_size=0.2, random_state=42)
m_cl = LinearRegression().fit(Xtr_cl, ytr_cl)
print(f"  WITHOUT leaky feature — train R²: {r2_score(ytr_cl, m_cl.predict(Xtr_cl)):.4f}  "
      f"test R²: {r2_score(yte_cl, m_cl.predict(Xte_cl)):.4f}")

print("""
  With the leaky feature the model achieves near-perfect train and test R²
  because it can reconstruct y almost exactly from (y + epsilon).
  This is target leakage — the feature is a noisy copy of the label.
  In production, price_plus_noise would not be available (you don't know
  the price before you predict it), so the model would be useless.
""")


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 9 — Histogram of test R² across 50 random seeds
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("EXERCISE 9 — Stability of a single split (50 seeds)")
print("=" * 60)

scores_50 = []
for seed in range(50):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=seed)
    scores_50.append(r2_score(yte, LinearRegression().fit(Xtr, ytr).predict(Xte)))

print(f"  Mean test R² : {np.mean(scores_50):.4f}")
print(f"  Std           : {np.std(scores_50):.4f}")
print(f"  Min           : {min(scores_50):.4f}")
print(f"  Max           : {max(scores_50):.4f}")
print("""
  The standard deviation shows how much the reported score depends on
  which random split was chosen. A single split can be misleading.
  Cross-validation (Day 14) removes this variance by averaging many splits.
""")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(scores_50, bins=12, color="#4A90D9", edgecolor="black", alpha=0.8)
ax.axvline(np.mean(scores_50), color="#E74C3C", linewidth=1.8,
           linestyle="--", label=f"Mean = {np.mean(scores_50):.3f}")
ax.set_xlabel("Test R²")
ax.set_ylabel("Frequency")
ax.set_title("Distribution of Test R² Across 50 Random Splits")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig("day-02/solutions/ex9_histogram.png", dpi=120, bbox_inches="tight")
print("  Plot saved → day-02/solutions/ex9_histogram.png")


# ═══════════════════════════════════════════════════════════════════════════
# Exercise 10 — shuffle=False with sorted data
# ═══════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("EXERCISE 10 — shuffle=False with sorted data")
print("=" * 60)

# Sort by price (ascending)
order  = np.argsort(y)
X_sort = X[order]
y_sort = y[order]

# No shuffle: last 20% = most expensive houses
split_idx = int(0.8 * n)
Xtr_ns, Xte_ns = X_sort[:split_idx], X_sort[split_idx:]
ytr_ns, yte_ns = y_sort[:split_idx], y_sort[split_idx:]
m_ns = LinearRegression().fit(Xtr_ns, ytr_ns)
r2_ns = r2_score(yte_ns, m_ns.predict(Xte_ns))

# Shuffled split for comparison
Xtr_sh, Xte_sh, ytr_sh, yte_sh = train_test_split(
    X_sort, y_sort, test_size=0.2, random_state=42)
m_sh = LinearRegression().fit(Xtr_sh, ytr_sh)
r2_sh = r2_score(yte_sh, m_sh.predict(Xte_sh))

print(f"  shuffle=False R² (test on top 20% prices) : {r2_ns:.4f}")
print(f"  shuffle=True  R² (random 20%)             : {r2_sh:.4f}")
print("""
  Without shuffling, the test set contains only high-price houses not well
  represented in training. The model has not seen the full price range during
  training, so it extrapolates poorly — test R² is typically lower.
  Always shuffle unless the data has a time ordering (time series) that must
  be preserved.
""")

print("\n" + "=" * 60)
print("  DAY 2 SOLUTIONS COMPLETE!")
print("=" * 60)
