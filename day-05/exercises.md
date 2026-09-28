# Day 5 Exercises — Bias-Variance Tradeoff, Overfitting, and Regularization

Work through these in order. Attempt every question before opening `solutions/`.

---

## Theory

**1. Bias-Variance Decomposition Derivation:**
   - Let $y = f(x) + \epsilon$ with $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$, where $\epsilon$ is independent of $x$ and the training set $\mathcal{D}$.
   - Let $\hat{f}(x)$ be the estimator trained on $\mathcal{D}$.
   - Starting from $\mathbb{E}\left[ (y - \hat{f}(x))^2 \right]$, expand the square using:
     $$y - \hat{f}(x) = \left( f(x) - \mathbb{E}[\hat{f}(x)] \right) + \left( \mathbb{E}[\hat{f}(x)] - \hat{f}(x) \right) + \epsilon$$
   - Show step-by-step why each of the three cross-product terms evaluates to zero.
   - What happens to the bias and variance as model complexity increases?

**2. Ridge Regression Normal Equations & Strict Invertibility:**
   - Consider the Ridge loss function:
     $$J(w) = \frac{1}{2} \|y - Xw\|_2^2 + \frac{\lambda}{2} \|w\|_2^2$$
   - Compute the gradient $\nabla_w J(w)$ with respect to $w$, set it to zero, and solve for $w^*$.
   - Let $X \in \mathbb{R}^{n \times p}$. Prove that for any $\lambda > 0$, the matrix $(X^T X + \lambda I)$ is strictly positive definite, and therefore invertible, even if $p > n$ or if columns of $X$ are linearly dependent.

**3. The Geometry of Sparsity (Lasso vs Ridge):**
   - Formulate Ridge and Lasso as constrained optimization problems:
     $$\min_w \|y - Xw\|_2^2 \quad \text{subject to } \|w\|_p \le s$$
   - Sketch or describe the constraint boundary for $p=2$ (Ridge) vs $p=1$ (Lasso) in the $(w_1, w_2)$ plane.
   - Explain why the elliptical contours of the quadratic loss function are mathematically much more likely to hit the constraint region at a corner (where $w_1=0$ or $w_2=0$) for $p=1$, but at a smooth curve (where both $w_1 \neq 0, w_2 \neq 0$) for $p=2$.

**4. Feature Standardization & Unpenalized Intercept:**
   - Why is feature standardization mandatory before fitting Ridge or Lasso regression? What would happen to the coefficient of a feature measured in meters versus the same feature measured in millimeters?
   - Explain why the intercept $b$ (or $w_0$) should **never** be included in the penalty term $\Omega(w)$. What would happen if the target variable $y$ were shifted by adding a constant $+1000$ to all observations?

**5. Effective Degrees of Freedom:**
   - For an unregularized OLS model with full column rank, the degrees of freedom is $p$.
   - In Ridge regression, the effective degrees of freedom is defined as:
     $$\text{df}(\lambda) = \text{Tr}\left( X (X^T X + \lambda I)^{-1} X^T \right) = \sum_{j=1}^p \frac{\sigma_j^2}{\sigma_j^2 + \lambda}$$
     where $\sigma_j$ are the singular values of $X$.
   - Compute $\lim_{\lambda \to 0} \text{df}(\lambda)$ and $\lim_{\lambda \to \infty} \text{df}(\lambda)$.
   - Interpret how $\text{df}(\lambda)$ quantifies the "effective number of free parameters" as a continuous function of $\lambda$.

---

## Code

**6. Scratch Ridge Regression with Automated Scaling:**
   - Write a self-contained class or function `RidgeScratch(alpha=1.0)` that:
     1. Centers $y$ and standardizes $X$ (stores $\mu_X, \sigma_X, \bar{y}$).
     2. Solves the analytical Ridge normal equations.
     3. Back-transforms coefficients and computes the unregularized intercept.
     4. Provides a `.predict(X)` method.
   - Verify that your predictions and coefficients match `sklearn.linear_model.Ridge` to within $10^{-10}$.

**7. Coordinate Descent for Lasso from Scratch:**
   - For an $L_1$-penalized objective, gradient descent fails at $w_j = 0$ because the absolute value function is non-differentiable. Coordinate descent updates one coordinate at a time using the soft-thresholding operator:
     $$\mathcal{S}_{\tau}(z) = \text{sign}(z) \cdot \max(0, |z| - \tau)$$
   - Implement `lasso_coordinate_descent(X, y, alpha=0.1, max_iter=1000, tol=1e-5)`.
   - Run it on synthetic sparse data and compare your learned coefficients to `sklearn.linear_model.Lasso`.

**8. Multicollinearity Stress Test & Coefficient Stability:**
   - Generate synthetic data ($n=60$) where $x_2 = 2x_1 + \epsilon$ with $\epsilon \sim \mathcal{N}(0, 0.01)$ (severe collinearity).
   - Fit 100 bootstrap resamples using OLS and 100 bootstrap resamples using Ridge ($\alpha=10$).
   - Compute the empirical variance of the estimated coefficients $\text{Var}(\hat{w}_1)$ and $\text{Var}(\hat{w}_2)$ under OLS vs Ridge.
   - Plot a 2D scatter of $(\hat{w}_1, \hat{w}_2)$ across the bootstrap runs. Notice how OLS spreads out wildly along a diagonal line, while Ridge clusters tightly.

**9. Hyperparameter Tuning & Validation Curve:**
   - Create an overfitted polynomial dataset (degree 10 with moderate noise).
   - Implement 5-fold cross-validation from scratch to evaluate Ridge across 50 log-spaced $\alpha$ values from $10^{-4}$ to $10^4$.
   - Plot the training MSE curve and validation MSE curve against $\log_{10}(\alpha)$.
   - Identify the optimal $\alpha^*$ that minimizes validation error and report the test set performance improvement over unregularized OLS ($\alpha=0$).

**10. Elastic Net & Grouping Effect:**
   - Generate a dataset with two pairs of collinear features: $(x_1, x_2)$ are correlated with $r \approx 0.95$, and $(x_3, x_4)$ are correlated with $r \approx 0.95$, plus 6 noise features.
   - Train Lasso ($\alpha=0.2$) and Elastic Net ($\alpha=0.2, \text{l1\_ratio}=0.5$).
   - Show how Lasso arbitrarily keeps one feature and zeros out its partner, whereas Elastic Net retains both correlated features with balanced weights.

---

## Reflection

Write a short paragraph (5–8 sentences):
- In your own words, what is the fundamental tradeoff between bias and variance?
- Why does OLS tend to overfit when features are numerous or collinear, and why are exploding coefficients the primary symptom?
- Why does $L_1$ regularization produce exact zeros while $L_2$ only shrinks coefficients asymptotically?
- When you begin a real-world predictive modeling task tomorrow, how will you decide whether to reach for OLS, Ridge, Lasso, or Elastic Net?
