"""
=======================================================
  DAY 1 — Introduction to Predictive Modeling
  Topic : Linear Regression from scratch + with sklearn
  Run   : python code.py
=======================================================
"""

# ── 0. Imports ──────────────────────────────────────────────────────────────
import numpy as np
import matplotlib
matplotlib.use("Agg")          # save plots to file (no display needed)
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ═══════════════════════════════════════════════════════════════════════════
# PART 1 — Understanding the problem with a simple analogy
# ═══════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART 1 — THE STUDY-HOURS ANALOGY")
print("=" * 60)

# Our tiny dataset: hours studied vs exam score
hours_studied = [1, 2, 3, 4, 5, 6, 7, 8]
exam_scores   = [52, 58, 65, 70, 78, 83, 90, 95]

print("\nHours studied | Exam Score")
print("-" * 28)
for h, s in zip(hours_studied, exam_scores):
    bar = "█" * int(s / 5)
    print(f"  {h} hour(s)     |  {s}   {bar}")

# ───────────────────────────────────────────────────────────────────────────
# PART 2 — Manual Linear Regression (so you see the math!)
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 2 — LINEAR REGRESSION MATH (manual)")
print("=" * 60)

x = np.array(hours_studied, dtype=float)
y = np.array(exam_scores,   dtype=float)

# Formulas for slope (m) and intercept (b):
#   m = Σ((xi - x̄)(yi - ȳ)) / Σ((xi - x̄)²)
#   b = ȳ - m * x̄

x_mean = np.mean(x)
y_mean = np.mean(y)

numerator   = np.sum((x - x_mean) * (y - y_mean))
denominator = np.sum((x - x_mean) ** 2)

m = numerator / denominator   # slope
b = y_mean - m * x_mean       # intercept

print(f"\n  x̄ (mean hours) = {x_mean:.2f}")
print(f"  ȳ (mean score) = {y_mean:.2f}")
print(f"\n  Slope  (m) = {m:.4f}")
print(f"  Intercept (b) = {b:.4f}")
print(f"\n  Learned formula: score = {m:.2f} * hours + {b:.2f}")

# Make predictions with our manual formula
y_pred_manual = m * x + b

print("\n  Predictions vs Actual:")
print(f"  {'Hours':>7} | {'Actual':>7} | {'Predicted':>9} | {'Error':>7}")
print("  " + "-" * 38)
for xi, yi, yp in zip(x, y, y_pred_manual):
    error = yi - yp
    print(f"  {xi:>7.0f} | {yi:>7.1f} | {yp:>9.2f} | {error:>+7.2f}")

# ───────────────────────────────────────────────────────────────────────────
# PART 3 — Linear Regression with scikit-learn (the real-world way)
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 3 — LINEAR REGRESSION WITH SCIKIT-LEARN")
print("=" * 60)

# sklearn expects a 2D array for X
X = x.reshape(-1, 1)   # shape: (8, 1)

model = LinearRegression()
model.fit(X, y)         # <-- this is where "learning" happens

print(f"\n  sklearn slope     : {model.coef_[0]:.4f}")
print(f"  sklearn intercept : {model.intercept_:.4f}")
print("  (Should match our manual calculation above!)")

# Evaluate the model
y_pred_sklearn = model.predict(X)
mse = mean_squared_error(y, y_pred_sklearn)
r2  = r2_score(y, y_pred_sklearn)

print(f"\n  Mean Squared Error (MSE) : {mse:.4f}")
print(f"  R² Score                 : {r2:.4f}")
print("\n  MSE  → lower is better (0 = perfect)")
print("  R²   → closer to 1.0 is better (1.0 = perfect fit)")

# ───────────────────────────────────────────────────────────────────────────
# PART 4 — Making New Predictions (the exciting part!)
# ───────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("PART 4 — PREDICTING NEW SCORES")
print("=" * 60)

new_hours = [9, 10, 0.5]
print(f"\n  {'Hours':>7} | {'Predicted Score':>15}")
print("  " + "-" * 26)
for h in new_hours:
    pred = model.predict([[h]])[0]
    pred = max(0, min(100, pred))  # clamp to 0-100 range
    print(f"  {h:>7.1f} | {pred:>15.1f}")

# ───────────────────────────────────────────────────────────────────────────
# PART 5 — Visualise (saves a PNG you can open!)
# ───────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Day 1 — Linear Regression: Study Hours vs Exam Score",
             fontsize=14, fontweight="bold")

# Plot 1: scatter + regression line
ax1 = axes[0]
x_line = np.linspace(0, 10, 100)
y_line = model.coef_[0] * x_line + model.intercept_
ax1.scatter(x, y, color="#4A90D9", s=80, zorder=5, label="Actual data")
ax1.plot(x_line, y_line, color="#E74C3C", linewidth=2, label="Regression line")
ax1.set_xlabel("Hours Studied")
ax1.set_ylabel("Exam Score")
ax1.set_title("Data + Best-Fit Line")
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: residuals (errors)
ax2 = axes[1]
residuals = y - y_pred_sklearn
ax2.bar(x, residuals, color=["#2ECC71" if r >= 0 else "#E74C3C" for r in residuals],
        alpha=0.8, edgecolor="black", linewidth=0.5)
ax2.axhline(0, color="black", linewidth=1)
ax2.set_xlabel("Hours Studied")
ax2.set_ylabel("Error (Actual - Predicted)")
ax2.set_title("Residuals (Errors)")
ax2.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("day-01/day01_plot.png", dpi=120, bbox_inches="tight")
print("\n  Plot saved → day-01/day01_plot.png")

print("\n" + "=" * 60)
print("  DAY 1 COMPLETE! Read exercises.md for your challenge.")
print("=" * 60)
