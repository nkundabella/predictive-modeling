# Day 4 Exercises — Evaluation Metrics in Regression

Work through these in order. Attempt every question before opening `solutions/`.

---

## Theory

**1.** Optimal Constant Predictor:
   - Prove mathematically that the constant prediction $c$ that minimises $\text{MSE}(c) = \frac{1}{n} \sum_{i=1}^n (y_i - c)^2$ is the sample mean $\bar{y}$.
   - Differentiate $\text{MAE}(c) = \frac{1}{n} \sum_{i=1}^n |y_i - c|$ (for non-identical $y_i$) and show that setting the derivative to zero requires the number of positive residuals to equal the number of negative residuals, proving $c$ must be the sample **median**.

**2.** Relationship Between MAE and RMSE:
   - Prove that $\text{MAE} \le \text{RMSE} \le \sqrt{n} \cdot \text{MAE}$.
   - Under what exact mathematical condition does $\text{RMSE} = \text{MAE}$?
   - What does a very large ratio $\frac{\text{RMSE}}{\text{MAE}} \gg 1.25$ tell you about the distribution of errors in your model?

**3.** Negative $R^2$:
   - For an unregularized OLS model with an intercept, can $R^2$ ever be negative on the **training set**? Why or why not?
   - Can $R^2$ be negative on a **test set**?
   - Give a concrete mathematical and practical example where a model produces $R^2 < 0$. What does this mean compared to predicting $\bar{y}$?

**4.** Adjusted $R^2$ Properties:
   - The formula for Adjusted $R^2$ is:
     $$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$
   - Can Adjusted $R^2$ be negative?
   - Can Adjusted $R^2$ be greater than standard $R^2$?
   - Under what condition does adding a new feature increase $R^2_{\text{adj}}$? (Hint: consider the $F$-statistic or $t$-statistic of the new feature).

**5.** Residual Diagnostics:
   - You inspect a plot of residuals ($y - \hat{y}$) versus fitted values ($\hat{y}$) and observe:
     a. A funnel/fan shape (variance increases as $\hat{y}$ increases).
     b. An inverted U-shaped parabola.
     c. A horizontal band with constant vertical thickness centered at 0.
   - For each case, state whether regression assumptions are satisfied and what corrective action you should take.

---

## Code

**6.** Write a standalone evaluation function `regression_report(y_true, y_pred, p=1)`:
   - Computes MAE, MSE, RMSE, $R^2$, Adjusted $R^2$, and the ratio $\frac{\text{RMSE}}{\text{MAE}}$.
   - Returns a structured dictionary and prints a formatted summary table.
   - Verify your function against `scikit-learn` metrics.

**7.** **Outlier Stress Test**:
   - Generate a clean synthetic dataset ($n=100$) following $y = 4x + 10 + \mathcal{N}(0, 4)$.
   - Fit an OLS model and record baseline test MAE, RMSE, and $R^2$.
   - Progressively corrupt $k$ test labels ($k \in [0, 1, 2, 5, 10]$) by adding an error spike of $+100$.
   - Plot MAE vs RMSE as a function of the number of outliers $k$. Which metric degrades faster?

**8.** **Demonstrate Negative $R^2$ on Test Data**:
   - Create a scenario where a model fitted on training data yields a negative $R^2$ on the test set.
   - (For example: train on a narrow domain $x \in [0, 5]$ with a high-degree polynomial or an ill-chosen slope, then evaluate on $x \in [10, 15]$).
   - Compute test $R^2$ and plot actual test points, model predictions, and the horizontal baseline $\bar{y}_{\text{test}}$. Explain visually why $R^2 < 0$.

**9.** **Heteroscedasticity Diagnostic Tool**:
   - Write a function `check_heteroscedasticity(y_pred, residuals)` that:
     1. Calculates the Spearman rank correlation between $|\text{residuals}|$ and $\hat{y}$ (a non-parametric proxy for the Breusch-Pagan test).
     2. Flags whether residual variance significantly depends on predictions ($p < 0.05$).
     3. Produces a residual diagnostic scatter plot.
   - Test your function on two synthetic datasets: one homoscedastic and one heteroscedastic ($y = 2x + \epsilon \cdot x$).

**10.** **Huber Loss Implementation**:
   - The Huber Loss is a hybrid metric: quadratic for small errors and linear for large errors:
     $$L_\delta(e) = \begin{cases} \frac{1}{2} e^2 & \text{if } |e| \le \delta \\ \delta (|e| - \frac{1}{2}\delta) & \text{if } |e| > \delta \end{cases}$$
   - Implement `huber_loss(y_true, y_pred, delta=1.35)`.
   - Compare MAE, RMSE, and Huber Loss across a clean test set and an outlier-contaminated test set. Explain why Huber Loss is often the preferred loss in robust regression.

---

## Reflection

Write a short paragraph (5–8 sentences):
Before today, which metric would you have naturally reached for first when evaluating a regression model?
How has your understanding of $R^2$ vs Adjusted $R^2$ changed?
Why is looking at summary metrics alone dangerous without inspecting residual plots?
Can you describe a business or engineering situation where a large RMSE is catastrophic, but an equal MAE would be acceptable?
