import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------------
# Dataset
# Hours studied vs exam score (8 observations)
# ---------------------------------------------------------------------------
x = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=float)
y = np.array([52, 58, 65, 70, 78, 83, 90, 95], dtype=float)


# ---------------------------------------------------------------------------
# Part 1: OLS closed-form solution
#
# w1 = sum((xi - x_mean)(yi - y_mean)) / sum((xi - x_mean)^2)
# w0 = y_mean - w1 * x_mean
# ---------------------------------------------------------------------------
x_mean = x.mean()
y_mean = y.mean()

w1 = np.sum((x - x_mean) * (y - y_mean)) / np.sum((x - x_mean) ** 2)
w0 = y_mean - w1 * x_mean

y_pred_ols = w1 * x + w0

print("OLS (manual)")
print(f"  w1 (slope)     : {w1:.6f}")
print(f"  w0 (intercept) : {w0:.6f}")
print(f"  equation       : y = {w1:.4f} * x + {w0:.4f}")

print("\n  x   |  y actual  |  y predicted  |  residual")
print("  " + "-" * 46)
for xi, yi, yp in zip(x, y, y_pred_ols):
    print(f"  {xi:.0f}   |  {yi:8.2f}  |  {yp:11.4f}  |  {yi - yp:+.4f}")


# ---------------------------------------------------------------------------
# Part 2: Verification with scikit-learn
# ---------------------------------------------------------------------------
X = x.reshape(-1, 1)
sk_model = LinearRegression()
sk_model.fit(X, y)
y_pred_sk = sk_model.predict(X)

print("\nsklearn LinearRegression")
print(f"  coef_      : {sk_model.coef_[0]:.6f}")
print(f"  intercept_ : {sk_model.intercept_:.6f}")
print("  (must match OLS above)")


# ---------------------------------------------------------------------------
# Part 3: Evaluation metrics
# ---------------------------------------------------------------------------
mse  = mean_squared_error(y, y_pred_sk)
rmse = np.sqrt(mse)
mae  = np.mean(np.abs(y - y_pred_sk))
r2   = r2_score(y, y_pred_sk)

rss = np.sum((y - y_pred_sk) ** 2)
tss = np.sum((y - y_mean) ** 2)
r2_manual = 1 - rss / tss

print("\nEvaluation")
print(f"  MSE         : {mse:.6f}")
print(f"  RMSE        : {rmse:.6f}")
print(f"  MAE         : {mae:.6f}")
print(f"  R^2         : {r2:.6f}")
print(f"  R^2 (manual): {r2_manual:.6f}")


# ---------------------------------------------------------------------------
# Part 4: Extrapolation — predicting outside training range
# Note: linear models assume the trend continues indefinitely.
# This is often unrealistic. Always know your model's valid range.
# ---------------------------------------------------------------------------
new_x = np.array([0, 9, 10, 12]).reshape(-1, 1)
new_preds = sk_model.predict(new_x)

print("\nPredictions on new inputs")
print(f"  {'x':>5} | {'y_hat':>10}")
print("  " + "-" * 20)
for xi, yp in zip(new_x.flatten(), new_preds):
    print(f"  {xi:>5.1f} | {yp:>10.4f}")


# ---------------------------------------------------------------------------
# Part 5: Plots
# Figure 1: data + regression line
# Figure 2: residual plot (check for patterns — patterns indicate model misfit)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

x_line = np.linspace(0, 10, 200)
y_line = w1 * x_line + w0

ax = axes[0]
ax.scatter(x, y, color="#2563EB", s=60, zorder=5, label="Observations")
ax.plot(x_line, y_line, color="#DC2626", linewidth=1.8, label=f"OLS fit: y={w1:.2f}x+{w0:.2f}")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Exam Score")
ax.set_title("Linear Regression — OLS Fit")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1]
residuals = y - y_pred_sk
ax.scatter(x, residuals, color="#2563EB", s=60, zorder=5)
ax.axhline(0, color="#DC2626", linewidth=1.2, linestyle="--")
ax.set_xlabel("Hours Studied")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("day-01/plot.png", dpi=120, bbox_inches="tight")
print("\nPlot saved: day-01/plot.png")
