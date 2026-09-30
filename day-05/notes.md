# Day 5 — Bias-Variance Tradeoff, Overfitting, and Regularization

## Overview

In Day 1 through Day 4, we explored how Ordinary Least Squares (OLS) fits a line or hyperplane to minimize the Residual Sum of Squares (RSS), and how we measure its prediction quality.

However, minimizing training error alone is dangerous. A model with zero training error can fail catastrophically on unseen test data. This phenomenon—**overfitting**—is governed by one of the most fundamental principles in all of statistical machine learning: the **Bias-Variance Tradeoff**.

Today, we will:
1. Formally derive the **Bias-Variance Decomposition** of Mean Squared Error.
2. Understand what causes overfitting in linear models and why weights explode.
3. Discover **Regularization**: constraining model complexity by penalizing large weights.
4. Deeply analyze **Ridge Regression ($L_2$)** and its analytical closed-form solution.
5. Deeply analyze **Lasso Regression ($L_1$)**, the geometry of sparsity, and feature selection.
6. Explore **Elastic Net** and practical strategies for selecting the regularization strength $\lambda$ (or $\alpha$).

---

## 1. The Bias-Variance Decomposition

Suppose the true relationship between an input vector $x$ and a scalar response $y$ is:

$$y = f(x) + \epsilon$$

Where:
- $f(x)$ is the true, unknown data-generating function.
- $\epsilon$ is random noise with mean zero and variance $\sigma^2$: $\mathbb{E}[\epsilon] = 0$ and $\text{Var}(\epsilon) = \sigma^2$.
- The noise $\epsilon$ is independent of the input $x$.

Given a training dataset $\mathcal{D}$, we train a predictive model $\hat{f}(x; \mathcal{D})$, abbreviated as $\hat{f}(x)$. Notice that $\hat{f}(x)$ is a random variable because it depends on the random draw of training data $\mathcal{D}$.

For an unseen test point $x_0$ with label $y_0 = f(x_0) + \epsilon$, the **Expected Prediction Error (Expected Mean Squared Error)** across all possible training datasets $\mathcal{D}$ is:

$$\mathbb{E}_{\mathcal{D}, \epsilon} \left[ \left(y_0 - \hat{f}(x_0)\right)^2 \right]$$

### Full Step-by-Step Derivation

Let $\mathbb{E}[\hat{f}(x_0)]$ denote the average prediction of the model at $x_0$ over all possible training datasets $\mathcal{D}$.

Substitute $y_0 = f(x_0) + \epsilon$:

$$\mathbb{E}\left[ \left( f(x_0) + \epsilon - \hat{f}(x_0) \right)^2 \right]$$

Rearrange the terms inside the square to group deterministic and estimation terms:

$$= \mathbb{E}\left[ \left( \underbrace{f(x_0) - \mathbb{E}[\hat{f}(x_0)]}_{\text{Bias}} + \underbrace{\mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0)}_{\text{Estimation Variation}} + \epsilon \right)^2 \right]$$

Let:
- $A = f(x_0) - \mathbb{E}[\hat{f}(x_0)]$ (a deterministic constant)
- $B = \mathbb{E}[\hat{f}(x_0)] - \hat{f}(x_0)$ (mean-zero random variable depending on $\mathcal{D}$)
- $C = \epsilon$ (mean-zero random variable independent of $\mathcal{D}$)

Expanding $(A + B + C)^2$:

$$(A + B + C)^2 = A^2 + B^2 + C^2 + 2AB + 2AC + 2BC$$

Now take expectations term by term:
1. $\mathbb{E}[A^2] = A^2 = \left( f(x_0) - \mathbb{E}[\hat{f}(x_0)] \right)^2 = \mathbf{\text{Bias}\left[\hat{f}(x_0)\right]^2}$
2. $\mathbb{E}[B^2] = \mathbb{E}\left[ \left( \hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)] \right)^2 \right] = \mathbf{\text{Var}\left(\hat{f}(x_0)\right)}$
3. $\mathbb{E}[C^2] = \mathbb{E}[\epsilon^2] = \mathbf{\sigma^2}$ (Irreducible Noise Variance)
4. $\mathbb{E}[2AB] = 2A \cdot \mathbb{E}[B] = 2A \cdot \left( \mathbb{E}[\hat{f}(x_0)] - \mathbb{E}[\hat{f}(x_0)] \right) = 0$
5. $\mathbb{E}[2AC] = 2A \cdot \mathbb{E}[\epsilon] = 0$
6. $\mathbb{E}[2BC] = 2 \cdot \mathbb{E}[B] \cdot \mathbb{E}[\epsilon] = 0$ (since $\mathcal{D}$ and $\epsilon$ are independent)

All cross-terms vanish! We arrive at the celebrated **Bias-Variance Decomposition**:

$$\boxed{\mathbb{E}\left[ \left(y_0 - \hat{f}(x_0)\right)^2 \right] = \underbrace{\left( \mathbb{E}[\hat{f}(x_0)] - f(x_0) \right)^2}_{\text{Bias}^2} + \underbrace{\mathbb{E}\left[ \left(\hat{f}(x_0) - \mathbb{E}[\hat{f}(x_0)]\right)^2 \right]}_{\text{Variance}} + \underbrace{\sigma^2}_{\text{Irreducible Error}}}$$

### Understanding the Three Components

| Component | Formal Meaning | Visual Analogy (Target / Bullseye) | Under- or Overfitting? |
|---|---|---|---|
| **$\text{Bias}^2$** | Error caused by simplifying assumptions in the model. How far the *average* model prediction is from the true reality. | Shots clustered consistently off-center (systematic miss). | **High Bias = Underfitting** (Model too simple, e.g. fitting a straight line to a sine wave). |
| **$\text{Variance}$** | Sensitivity of the model to the specific training dataset. How much $\hat{f}(x)$ changes if trained on a different sample from the same distribution. | Shots widely scattered all over the target board. | **High Variance = Overfitting** (Model too complex, fitting noise instead of signal). |
| **$\sigma^2$** | Natural stochastic noise inherent in the process or measurement error. Cannot be reduced by any model. | Wind blowing arrows off-course at random. | **Irreducible Floor** (Best achievable MSE). |

```
High Error │   \                        /  Total Test Error = Bias² + Variance + σ²
           │    \                      /
           │     \      Sweet         /
           │      \      Spot        /  ▲ High Variance (Overfitting)
           │       \      |         /   │
           │        \     ▼        /    │ Variance
           │         `-----------´      │
           │  ......................... │ 
           │  \                       / │
           │   \                     /  │
           │    \                   /   │
           │     \                 /    ▼
           │      `------------------------ Bias² (Underfitting)
           │  - - - - - - - - - - - - - - - Irreducible Error (σ²)
           └────────────────────────────────────────►
                      Model Complexity
```

---

## 2. Why Overfitting Happens in Linear Models

A linear model $\hat{y} = w^T x + b$ may seem too simple to overfit. But overfitting happens frequently in linear modeling when:

1. **Polynomial Expansion:** When we add higher-order powers ($x, x^2, x^3, \dots, x^d$), a degree-15 polynomial can pass through nearly every training point, oscillating wildly between them.
2. **Multicollinearity:** When features are strongly correlated, $X^T X$ is near-singular (ill-conditioned). OLS estimates swing wildly to massive positive and negative values (e.g. $\hat{y} = 10^5 x_1 - 10^5 x_2$), canceling each other out on training points but exploding on slight perturbations in test data.
3. **High Dimensionality ($p \approx n$ or $p > n$):** When the number of features $p$ approaches or exceeds the number of observations $n$, the system of equations is underdetermined. $X^T X$ is not invertible, and infinite zero-error solutions exist that memorize training noise.

### The Signature of Overfitting: Weight Explosion

Notice how overfit models behave: to fit individual noisy data points, the coefficients $w_j$ balloon to astronomical magnitudes.

If we can **constrain the size of the weights $\|w\|$**, we prevent the model from swinging violently between points, forcing it to find smoother, more generalizable hypotheses.

---

## 3. Regularization: Penalizing Complexity

Regularization modifies the optimization objective by adding a **penalty term** $\Omega(w)$ that punishes complex or large weight vectors:

$$\min_{w, b} \mathcal{L}(w, b) = \underbrace{\frac{1}{2n} \sum_{i=1}^n \left( y_i - (w^T x_i + b) \right)^2}_{\text{Empirical Data Loss (Fit the Data)}} + \underbrace{\lambda \, \Omega(w)}_{\text{Complexity Penalty (Stay Simple)}}$$

Where:
- $\lambda \ge 0$ (in `scikit-learn`, denoted as `alpha`) is the **regularization hyperparameter**:
  - $\lambda = 0$: Unregularized OLS (pure empirical risk minimization).
  - $\lambda \to \infty$: Penalty dominates completely; all weights are driven toward $0$ (predicts constant mean).
  - $0 < \lambda < \infty$: Balances bias and variance to minimize generalization error.

> [!IMPORTANT]
> **Two Golden Rules of Regularization:**
> 1. **Always scale your features first!** The penalty $\Omega(w)$ penalizes the absolute size of coefficients. If feature $x_1$ is measured in dollars (0 to 1,000,000) and $x_2$ in years (0 to 10), their natural coefficients will differ by orders of magnitude. Without standardization, the regularizer unfairly punishes features with small numerical units.
> 2. **Never regularize the intercept $b$ (or $w_0$)!** The intercept represents the baseline value of $y$ when all centered inputs are zero. Penalizing $b$ forces the baseline toward zero, introducing artificial bias that depends purely on the arbitrary origin of $y$.

---

## 4. Ridge Regression ($L_2$ Regularization)

Ridge Regression (Hoerl & Kennard, 1970), also called **Tikhonov Regularization**, uses the squared $L_2$-norm of the weights as the penalty:

$$\min_w J(w) = \frac{1}{2n} \|y - Xw\|_2^2 + \frac{\lambda}{2} \|w\|_2^2 = \frac{1}{2n} \sum_{i=1}^n (y_i - w^T x_i)^2 + \frac{\lambda}{2} \sum_{j=1}^p w_j^2$$

*(Assuming centered data so $b=0$).*

### Analytical Closed-Form Solution

Let us derive the optimal weight vector $w^*$ by setting the matrix gradient with respect to $w$ to zero:

$$J(w) = \frac{1}{2n} (y - Xw)^T (y - Xw) + \frac{\lambda}{2} w^T w$$

Expand the quadratic matrix terms:

$$J(w) = \frac{1}{2n} \left( y^T y - 2 w^T X^T y + w^T X^T X w \right) + \frac{\lambda}{2} w^T w$$

Compute the gradient $\nabla_w J(w)$:

$$\nabla_w J(w) = \frac{1}{n} \left( -X^T y + X^T X w \right) + \lambda w = 0$$

Multiply through by $n$, letting $\gamma = n\lambda$:

$$-X^T y + X^T X w + \gamma w = 0$$

Factor out $w$:

$$(X^T X + \gamma I) w = X^T y$$

Solving for $w^*$:

$$\boxed{w_{\text{Ridge}}^* = (X^T X + \gamma I)^{-1} X^T y}$$

*(In standard convention where the loss is written as $\|y - Xw\|^2 + \lambda \|w\|^2$, the solution is simply $(X^T X + \lambda I)^{-1} X^T y$.)*

### Why Ridge Solves Multicollinearity and Non-Invertibility

1. In OLS, $X^T X$ is positive semi-definite (eigenvalues $\ge 0$). If features are collinear, some eigenvalues are zero, making $X^T X$ singular and non-invertible.
2. The Ridge matrix $(X^T X + \lambda I)$ adds $\lambda > 0$ to every diagonal entry.
3. If the eigenvalues of $X^T X$ are $\mu_1, \mu_2, \dots, \mu_p \ge 0$, then the eigenvalues of $(X^T X + \lambda I)$ are:
   $$\mu_j + \lambda > 0 \quad \text{for all } j$$
4. Since every eigenvalue is strictly positive, $(X^T X + \lambda I)$ is **guaranteed to be strictly positive definite and invertible**, even when $p > n$ or features are perfectly collinear!

### Shrinkage via Singular Value Decomposition (SVD)

Using the SVD $X = U \Sigma V^T$, the OLS prediction is $\hat{y}_{\text{OLS}} = \sum_{j=1}^p u_j u_j^T y$.

In Ridge regression, the prediction is:

$$\hat{y}_{\text{Ridge}} = \sum_{j=1}^p u_j \left( \frac{\sigma_j^2}{\sigma_j^2 + \lambda} \right) u_j^T y$$

Notice the **shrinkage factor**:

$$\rho_j = \frac{\sigma_j^2}{\sigma_j^2 + \lambda} < 1$$

- Directions with high variance (large singular value $\sigma_j^2 \gg \lambda$) are barely shrunk ($\rho_j \approx 1$).
- Directions with low variance (small $\sigma_j^2 \ll \lambda$, which represent noise and multicollinearity) are aggressively shrunk toward zero ($\rho_j \to 0$).

---

## 5. Lasso Regression ($L_1$ Regularization)

Lasso (**Least Absolute Shrinkage and Selection Operator**, Tibshirani, 1996) penalizes the sum of the absolute values of the weights ($L_1$-norm):

$$\min_w J(w) = \frac{1}{2n} \|y - Xw\|_2^2 + \lambda \|w\|_1 = \frac{1}{2n} \sum_{i=1}^n (y_i - w^T x_i)^2 + \lambda \sum_{j=1}^p |w_j|$$

### Key Distinction: Exact Sparsity

Unlike Ridge, which shrinks weights toward zero asymptotically, **Lasso drives weights exactly to zero**.
When a coefficient $w_j = 0$, that feature is completely removed from the model. Thus, Lasso performs **automatic feature selection**, producing sparse, interpretable models.

### Why Does Lasso Set Weights Exactly to Zero?

We can understand this through two complementary perspectives:

#### Perspective A: The Geometry of Constraint Regions

Both Ridge and Lasso can be formulated as constrained optimization problems:
- **Ridge:** $\min_w \text{RSS}(w)$ subject to $\sum w_j^2 \le s$ (an $L_2$ hypersphere / circle).
- **Lasso:** $\min_w \text{RSS}(w)$ subject to $\sum |w_j| \le s$ (an $L_1$ hyperdiamond / rotated square).

```
        Ridge (L2 Constraint)                      Lasso (L1 Constraint)
                w2                                         w2
                ▲                                          ▲
                │      / RSS Contours                      │  /\
             .──┼──.  /                                    │ /  \   / RSS Contours
           .'   │   '.                                     │/    \ /
          /     │     \                                    /      \
      ────┼─────┼─────┼────► w1                        ────┼──────┼──────► w1
          \     │     /                                    \      /
           '.   │   .'                                      \    /
             '──┼──'                                         \  /
                │  Smooth tangency:                           \/  Tangency hits sharp CORNER:
                │  w1 ≠ 0, w2 ≠ 0                             w1 = 0 (exact sparsity!)
```

The RSS contours are ellipses centered at the unconstrained OLS solution $\hat{w}_{\text{OLS}}$. As the ellipse expands from the center, the first point of contact with the constraint boundary is the optimal regularized solution:
- The **$L_2$ circle is smooth everywhere**. The ellipse almost always touches a smooth curved edge where both coordinates are non-zero.
- The **$L_1$ diamond has sharp, pointed corners aligned with the coordinate axes**. An expanding elliptical contour has a high geometric probability of contacting one of these sharp corners first. At a corner on an axis, the other coordinates are **identically zero**.

#### Perspective B: The Soft-Thresholding Operator

When features are orthonormal ($X^T X = I$), the Lasso optimization decouples into $p$ independent 1D problems. The analytical solution for each weight is given by the **soft-thresholding operator**:

$$\boxed{w_j^* = \mathcal{S}_{\lambda}(w_j^{\text{OLS}}) = \text{sign}(w_j^{\text{OLS}}) \cdot \max\left(0, |w_j^{\text{OLS}}| - \lambda\right)}$$

```
          w* (Lasso Weight)
                ▲
                │          / (Slope = 1)
                │         /
                │        /
      ──────────┼───────/─────────► w_OLS
         -λ     0      +λ
        /       │
       /        │
      /         │
```

- If $|w_j^{\text{OLS}}| \le \lambda$, the coefficient is **zeroed out completely**: $w_j^* = 0$.
- If $|w_j^{\text{OLS}}| > \lambda$, the coefficient is shrunk toward zero by a constant offset $\lambda$.

In contrast, Ridge in an orthonormal design applies **proportional shrinkage**:
$$w_j^{\text{Ridge}} = \frac{1}{1 + \lambda} w_j^{\text{OLS}}$$
Ridge scales down the coefficient by a fraction, but unless $w_j^{\text{OLS}} = 0$, $w_j^{\text{Ridge}}$ never reaches zero.

---

## 6. Elastic Net: The Best of Both Worlds

While Lasso is brilliant for feature selection, it has two known limitations:
1. When features are highly correlated, Lasso tends to arbitrarily pick one feature from the group and ignore the others.
2. In high dimensions where $p > n$, Lasso can select at most $n$ features before saturating.

**Elastic Net** (Zou & Hastie, 2005) combines both $L_1$ and $L_2$ penalties:

$$\min_w \frac{1}{2n} \|y - Xw\|_2^2 + \alpha \left( \rho \|w\|_1 + \frac{1 - \rho}{2} \|w\|_2^2 \right)$$

Where:
- $\alpha \ge 0$ controls total penalty strength.
- $\rho \in [0, 1]$ is the `l1_ratio`:
  - $\rho = 1 \implies$ Pure Lasso.
  - $\rho = 0 \implies$ Pure Ridge.
  - $0 < \rho < 1 \implies$ Hybrid that retains Lasso's sparsity while inheriting Ridge's ability to group correlated features.

---

## 7. Direct Comparison: OLS vs. Ridge vs. Lasso vs. Elastic Net

| Property | OLS | Ridge ($L_2$) | Lasso ($L_1$) | Elastic Net ($L_1 + L_2$) |
|---|---|---|---|---|
| **Penalty Term** | None | $\lambda \sum w_j^2$ | $\lambda \sum |w_j|$ | $\lambda_1 \sum |w_j| + \lambda_2 \sum w_j^2$ |
| **Closed-Form Solution?** | Yes: $(X^T X)^{-1} X^T y$ | Yes: $(X^T X + \lambda I)^{-1} X^T y$ | No (Iterative: Coordinate Descent) | No (Iterative: Coordinate Descent) |
| **Generates Sparse Weights?** | No | No (weights shrink, never 0) | **Yes** (automatic feature selection) | **Yes** (sparse with grouping) |
| **Handles $p > n$?** | No (fails / ill-conditioned) | Yes (always invertible) | Yes (selects at most $n$ features) | Yes (can select $> n$ features) |
| **Handles Collinear Features?** | Unstable, exploding weights | Excellent (shrinks together) | Picks one arbitrarily | Excellent (groups together) |
| **Computational Cost** | $O(np^2 + p^3)$ | $O(np^2 + p^3)$ | Fast (coordinate descent) | Fast (coordinate descent) |
| **Primary Use Case** | Baseline, $n \gg p$, no collinearity | Many small-to-moderate effects | Few dominant sparse effects | Correlated high-dimensional features |

---

## 8. Practical Guide: Choosing $\lambda$ ($\alpha$)

How do you pick the optimal penalty $\alpha$? You never guess. You find it systematically using **Validation Curves** or **K-Fold Cross-Validation**:

1. Define a log-spaced grid of candidates: e.g., `alphas = np.logspace(-4, 4, 100)`.
2. Evaluate test MSE (or out-of-fold validation MSE) across all $\alpha$.
3. As $\alpha$ increases:
   - Training error monotonically increases (more constrained).
   - Test error forms a U-shape: decreases as variance drops, reaches the global minimum, then increases as bias dominates.
4. Select $\alpha^*$ at the trough of the test error curve.

In `scikit-learn`:
```python
from sklearn.linear_model import RidgeCV, LassoCV

# Automated k-fold cross-validated hyperparameter search:
ridge = RidgeCV(alphas=np.logspace(-4, 4, 100), cv=5).fit(X_train_scaled, y_train)
best_alpha = ridge.alpha_
```
