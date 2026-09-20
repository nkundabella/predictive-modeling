# Day 1 — Ordinary Least Squares Linear Regression

## What Is a Predictive Model?

A predictive model is a function f that maps input variables X to an output variable y:

    y_hat = f(X)

The function f is not hand-written. It is estimated from data — a set of (X, y) pairs called the training set.
The model is useful when f generalises: it produces accurate y_hat for inputs it has not seen before.

---

## Ordinary Least Squares (OLS) — The Math

For a single input variable, the linear model is:

    y_hat = w1 * x + w0

Where:
  w1 = slope (weight)
  w0 = intercept (bias)

The goal is to find w1 and w0 that minimise the Residual Sum of Squares (RSS):

    RSS = sum( (yi - y_hat_i)^2 )  for i in 1..n

Taking the derivative of RSS with respect to w1 and w0, setting both to zero, and solving gives the closed-form OLS solution:

    w1 = sum( (xi - x_mean)(yi - y_mean) ) / sum( (xi - x_mean)^2 )
    w0 = y_mean - w1 * x_mean

This is the Ordinary Least Squares estimator. It finds the unique line that minimises squared vertical distances between the data points and the line.

Why squared? Squaring penalises large errors more than small ones, and makes the loss function differentiable everywhere.

---

## Residuals

The residual for observation i is:

    e_i = y_i - y_hat_i

Residuals are what the model failed to explain. In a well-fitting model, residuals should be:
  - Small in magnitude
  - Randomly distributed (no pattern)
  - Approximately zero on average

If residuals show a pattern (e.g. they increase with x), the linear assumption is violated and a more complex model is needed.

---

## Evaluation Metrics

**Mean Squared Error (MSE)**

    MSE = (1/n) * sum( (yi - y_hat_i)^2 )

Penalises large errors strongly. Units are squared.

**Root Mean Squared Error (RMSE)**

    RMSE = sqrt(MSE)

Same units as the target variable. Easier to interpret.

**Mean Absolute Error (MAE)**

    MAE = (1/n) * sum( |yi - y_hat_i| )

Less sensitive to outliers than MSE.

**R-squared (Coefficient of Determination)**

    R^2 = 1 - (RSS / TSS)

    where TSS = sum( (yi - y_mean)^2 )

R^2 measures the proportion of variance in y explained by the model.
  R^2 = 1.0 → perfect fit
  R^2 = 0.0 → model explains nothing (predicting the mean every time)
  R^2 < 0   → model is worse than predicting the mean

---

## Assumptions of Linear Regression

1. Linearity — the relationship between X and y is linear
2. Independence — observations are independent of each other
3. Homoscedasticity — residuals have constant variance across all values of X
4. Normality of residuals — residuals are approximately normally distributed
5. No multicollinearity — input features are not highly correlated (for multiple regression)

Violating these assumptions does not necessarily break the model, but it does degrade the reliability of predictions and statistical inference.

---

## Files

- `linear_regression.py` — OLS implementation from scratch + sklearn verification + plots
- `exercises.md` — problems covering derivation, implementation, and interpretation
- `solutions/` — reference answers

## Next

Day 2 covers train/test splitting, data leakage, and why evaluating on training data is invalid.
