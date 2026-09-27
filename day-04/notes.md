# Day 4 — Evaluation Metrics in Regression

## Why Metrics Matter

Training a model is only half the battle. Once a model produces predictions $\hat{y}$, you must quantify **how good those predictions are**.

No single evaluation metric tells the complete story. A model with an impressive $R^2$ might perform disastrously in production if its errors follow a heavy-tailed distribution, or if its residuals suffer from severe heteroscedasticity. Choosing the wrong evaluation metric can lead to selecting a model that optimizes for the wrong business or scientific objective.

---

## 1. Residuals: The Atomic Unit of Error

For any data point $(x_i, y_i)$, the **residual** $e_i$ is the difference between the true label and the model's prediction:

$$e_i = y_i - \hat{y}_i$$

* If $e_i > 0$: The model **under-predicted** (actual was higher than prediction).
* If $e_i < 0$: The model **over-predicted** (actual was lower than prediction).

Every evaluation metric in regression is a specific summary statistic of the residual vector $\mathbf{e} = [e_1, e_2, \dots, e_n]^T$.

---

## 2. Mean Absolute Error (MAE)

MAE measures the average absolute magnitude of the errors:

$$\text{MAE} = \frac{1}{n} \sum_{i=1}^n |y_i - \hat{y}_i| = \frac{1}{n} \sum_{i=1}^n |e_i|$$

### Properties
1. **Natural Units:** MAE is in the exact same units as the target variable $y$ (e.g., dollars, degrees, kg).
2. **Linear Penalty:** An error of $10$ is penalized exactly twice as much as an error of $5$.
3. **Robust to Outliers:** Because errors are not squared, a single massive outlier does not disproportionately dominate the score.
4. **Optimal Constant Prediction:** The constant $c$ that minimizes MAE on a dataset is the **median** of $y$.

---

## 3. Mean Squared Error (MSE) & Root Mean Squared Error (RMSE)

### Mean Squared Error (MSE)
$$\text{MSE} = \frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2 = \frac{1}{n} \sum_{i=1}^n e_i^2$$

### Root Mean Squared Error (RMSE)
$$\text{RMSE} = \sqrt{\text{MSE}} = \sqrt{\frac{1}{n} \sum_{i=1}^n (y_i - \hat{y}_i)^2}$$

### Properties
1. **Quadratic Penalty:** Squaring errors heavily penalizes large mistakes. An error of $10$ contributes $100$ to the sum, whereas an error of $5$ contributes only $25$ (a $4\times$ penalty for a $2\times$ error).
2. **Units:**
   - MSE is in squared units ($y^2$), making direct interpretation difficult.
   - RMSE takes the square root, restoring the metric back to the **original units of $y$**.
3. **Sensitivity to Outliers:** RMSE is significantly more sensitive to outliers than MAE. If your dataset contains erroneous extreme values, RMSE will blow up.
4. **Optimal Constant Prediction:** The constant $c$ that minimizes MSE is the **mean** of $y$ ($\bar{y}$).
5. **Relationship to MAE:**
   $$\text{MAE} \le \text{RMSE} \le \sqrt{n} \cdot \text{MAE}$$
   - $\text{RMSE} = \text{MAE}$ if and only if all errors have identical absolute magnitude ($|e_1| = |e_2| = \dots = |e_n|$).
   - The larger the gap $\text{RMSE} - \text{MAE}$, the higher the variance of the individual errors (i.e., presence of extreme errors/outliers).

---

## 4. Coefficient of Determination ($R^2$)

While MAE and RMSE depend on the scale of $y$, $R^2$ is **scale-independent** (dimensionless). It quantifies the proportion of variance in $y$ explained by the model:

$$R^2 = 1 - \frac{\text{RSS}}{\text{TSS}} = 1 - \frac{\sum_{i=1}^n (y_i - \hat{y}_i)^2}{\sum_{i=1}^n (y_i - \bar{y})^2}$$

Where:
- $\text{RSS} = \sum (y_i - \hat{y}_i)^2$ is the **Residual Sum of Squares** (unexplained variance).
- $\text{TSS} = \sum (y_i - \bar{y})^2$ is the **Total Sum of Squares** (variance of a naive baseline model that always predicts $\bar{y}$).

### Interpreting $R^2$
| Value | Meaning |
|---|---|
| $R^2 = 1.0$ | Perfect predictions. Every $\hat{y}_i = y_i$; $\text{RSS} = 0$. |
| $R^2 = 0.0$ | The model performs no better than predicting the sample mean $\bar{y}$ for every observation. |
| $0 < R^2 < 1$ | The model explains $100 \times R^2\%$ of the variation in $y$. |
| $R^2 < 0.0$ | **Possible on test data!** The model performs *worse* than the simple mean predictor $\bar{y}$. This indicates severe overfitting or a fundamentally inappropriate model. |

---

## 5. The Danger of $R^2$ and Adjusted $R^2$

### Why $R^2$ is Flawed for Multi-Feature Models
In linear regression, adding **any** new feature to the model—even purely random Gaussian noise—will **never decrease training $R^2$**, and almost always slightly increases it. This occurs because the extra column gives the OLS optimization additional degrees of freedom to fit sample-specific noise.

### Adjusted $R^2$ ($R^2_{\text{adj}}$)
Adjusted $R^2$ penalizes the addition of useless features by dividing RSS and TSS by their respective degrees of freedom:

$$R^2_{\text{adj}} = 1 - \left[ \frac{(1 - R^2)(n - 1)}{n - p - 1} \right]$$

Where:
- $n$ is the number of observations.
- $p$ is the number of features (excluding intercept).

* If a new feature improves $R^2$ by more than expected by random chance, $R^2_{\text{adj}}$ increases.
* If a new feature adds marginal or no predictive power, the penalty term $\frac{n - 1}{n - p - 1}$ dominates, and $R^2_{\text{adj}}$ **decreases**.

---

## 6. Summary Comparison: Which Metric When?

| Metric | Formula | Units | Outlier Sensitivity | Best Used When... |
|---|---|---|---|---|
| **MAE** | $\frac{1}{n}\sum \|e_i\|$ | Same as $y$ | Low (linear penalty) | Large errors are not exponentially worse; dataset has outliers. |
| **MSE** | $\frac{1}{n}\sum e_i^2$ | $y^2$ | High (quadratic penalty) | Mathematical optimization (smooth gradient); large errors are catastrophic. |
| **RMSE** | $\sqrt{\text{MSE}}$ | Same as $y$ | High | You need human-interpretable units while penalizing large errors. |
| **$R^2$** | $1 - \frac{\text{RSS}}{\text{TSS}}$ | None (0 to 1) | High (inherits from MSE) | Comparing models across different datasets/scales. |
| **$R^2_{\text{adj}}$** | $1 - \frac{(1-R^2)(n-1)}{n-p-1}$ | None ($< 1$) | High | Comparing feature subsets or models with different numbers of inputs. |

---

## 7. Residual Diagnostics: Beyond Summary Numbers

Scalar metrics condense thousands of predictions into a single number, hiding structural model failures. Always inspect **residual plots**:

1. **Residuals vs. Fitted Values ($\hat{y}$ vs $e$):**
   - **Ideal:** A horizontal cloud centered around $e=0$ with constant vertical spread.
   - **Curve / U-shape:** Indicates **non-linearity** (e.g., model needs polynomial features or log transform).
   - **Fan / Funnel shape:** Indicates **heteroscedasticity** (residual variance increases/decreases with $\hat{y}$).
2. **Q-Q Plot / Residual Histogram:**
   - Evaluates whether residuals are approximately normally distributed (important for hypothesis tests, $p$-values, and confidence intervals).
3. **Residuals vs. Features ($x_j$ vs $e$):**
   - If residuals correlate with any feature, that feature's functional form is misspecified.

---

## Summary Workflow

1. Report **RMSE** and **MAE** together: their gap reveals error dispersion.
2. Use **$R^2$** for high-level explanatory power, but rely on **$R^2_{\text{adj}}$** when comparing models with different feature counts.
3. Never trust metrics alone—always visualize **residuals vs. fitted values**.
