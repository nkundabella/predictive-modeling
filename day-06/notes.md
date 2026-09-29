# Day 6 — Logistic Regression and the Sigmoid Function

## Overview

Welcome to **Week 2: Classification**!

In Week 1 (Days 1–5), we explored regression: predicting a continuous scalar target $y \in \mathbb{R}$ (such as house prices, stock returns, or temperatures). In regression, error was measured by squared or absolute distances between real numbers.

Today, we transition to **classification**, where the target variable $y$ is categorical. In binary classification, $y \in \{0, 1\}$ represents qualitative outcomes:
- Is an email spam ($y=1$) or legitimate ($y=0$)?
- Does a patient have a specific condition ($y=1$) or not ($y=0$)?
- Will a borrower default on a loan ($y=1$) or repay ($y=0$)?

Despite its historical name containing the word "regression", **Logistic Regression** is the foundational, indispensable parametric algorithm for **binary classification**.

Today, we will cover:
1. Why Ordinary Least Squares (Linear Regression) fails for classification.
2. The **Sigmoid (Logistic) Function**: geometry, properties, and calculus.
3. Odds, Log-Odds, and the **Logit Link Function**.
4. The Bernoulli likelihood and the derivation of **Binary Cross-Entropy (Log Loss)**.
5. Why Mean Squared Error fails with Sigmoid activations (non-convexity and vanishing gradients).
6. Analytical derivation of the **Log Loss Gradient** via the chain rule.
7. Optimization: Gradient Descent and Second-Order Newton-Raphson (**IRLS**).
8. Decision thresholds, the decision boundary geometry, and cost-sensitive classification.
9. The **Perfect Separation Problem** and why regularization ($L_2$/$L_1$) is vital.
10. Multi-class extensions: **One-vs-Rest (OvR)** and **Softmax (Multinomial Logistic Regression)**.

---

## 1. Why Linear Regression Fails for Classification

A natural initial thought is to treat the binary labels $\{0, 1\}$ as numbers, fit an Ordinary Least Squares (OLS) line $\hat{y} = w^T x + b$, and apply a threshold at $0.5$:
$$\hat{y}_{\text{class}} = \begin{cases} 1 & \text{if } w^T x + b \ge 0.5 \\ 0 & \text{if } w^T x + b < 0.5 \end{cases}$$

This naive approach breaks down in three catastrophic ways:

```
Linear Regression for Classification Failure:
  y
  1 ┼              ●  ●  ●  ●        ● [Extreme Outlier with y=1]
    │             /
0.5 ┼────────────/─────────── New Shifted Boundary ──
    │           / 
    │          /  Original Decision Boundary
    │         /       │
  0 ┼  ○  ○  ○        │
    └─────────────────┼───────────────────────────────► x
                      ▲
  The OLS line tilts heavily to accommodate the far-right outlier,
  misclassifying valid points near the original boundary!
```

### Failure 1: Extreme Sensitivity to Outliers
OLS minimizes squared residual distance $(y - \hat{y})^2$. If an observation with $y=1$ has a very large positive feature value $x$, its linear prediction might be $\hat{y} = 4.0$. 
In classification, $\hat{y} = 4.0$ is a "wildly confident correct positive." But OLS considers this an error of $(1 - 4)^2 = 9.0$! To minimize this artificial squared penalty, OLS rotates the regression line toward the distant outlier. This tilts the decision threshold and causes points near the boundary to be misclassified.

### Failure 2: Unbounded Predictions Outside $[0, 1]$
Probabilities must satisfy $0 \le P(Y=1|X) \le 1$. A linear function $w^T x + b$ has range $(-\infty, +\infty)$. For extreme feature values, linear regression outputs probabilities of $-0.4$ or $+1.8$, which are mathematically meaningless as probabilities.

### Failure 3: Heteroscedasticity and Violation of Normality
OLS assumes the residuals $\epsilon = y - \hat{y}$ are normally distributed with constant variance $\sigma^2$ (homoscedasticity).
For a binary variable $Y \in \{0, 1\}$ with $P(Y=1) = p$, the variance of $Y$ is:
$$\text{Var}(Y) = p(1 - p)$$
The variance depends directly on the mean prediction $p$, inherently violating the homoscedasticity assumption of linear regression.

---

## 2. The Sigmoid (Logistic) Function

To model a probability $p = P(Y=1|x) \in (0, 1)$, we need a smooth, monotonic mathematical function that maps the real line $\mathbb{R} \in (-\infty, +\infty)$ into the open interval $(0, 1)$.

The standard **Sigmoid** (or **logistic**) function $\sigma: \mathbb{R} \to (0, 1)$ is defined as:

$$\boxed{\sigma(z) = \frac{1}{1 + e^{-z}} = \frac{e^z}{e^z + 1}}$$

where $z = w^T x + b$ is the linear combination of inputs (often called the **logit** or **activation**).

```
   σ(z)
    1.0 ┼                                 ───────── Asymptote at 1.0
        │                           . · ´
        │                        . ´
    0.5 ┼─────────────────────┼──────────────────── Inflection point at z=0, σ(0)=0.5
        │                 . ´
        │           . · ´
    0.0 ┼─────────                                 Asymptote at 0.0
        └───────────┬─────────┬─────────┬──────────► z = w^T x + b
                   -4        -2         0         2         4
```

### Fundamental Mathematical Properties of Sigmoid

1. **Range and Domain**:
   - Domain: $z \in (-\infty, +\infty)$
   - Range: $\sigma(z) \in (0, 1)$

2. **Limits**:
   $$\lim_{z \to +\infty} \sigma(z) = 1, \quad \lim_{z \to -\infty} \sigma(z) = 0$$

3. **Symmetry About the Center $(0, 0.5)$**:
   $$\sigma(-z) = 1 - \sigma(z)$$
   *Proof*:
   $$\sigma(-z) = \frac{1}{1 + e^{-(-z)}} = \frac{1}{1 + e^z} = \frac{e^{-z}}{e^{-z}(1 + e^z)} = \frac{e^{-z}}{e^{-z} + 1} = \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} = 1 - \frac{1}{1 + e^{-z}} = 1 - \sigma(z) \quad \blacksquare$$

4. **The Derivative of the Sigmoid Function**:
   The derivative has a closed-form formula expressed directly in terms of $\sigma(z)$:
   $$\boxed{\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))}$$

   *Step-by-Step Derivation*:
   Using the quotient rule or power rule on $\sigma(z) = (1 + e^{-z})^{-1}$:
   $$\begin{aligned}
   \frac{d}{dz}\sigma(z) &= \frac{d}{dz} (1 + e^{-z})^{-1} \\
   &= -1 \cdot (1 + e^{-z})^{-2} \cdot \frac{d}{dz}(1 + e^{-z}) \\
   &= -(1 + e^{-z})^{-2} \cdot (-e^{-z}) \\
   &= \frac{e^{-z}}{(1 + e^{-z})^2} \\
   &= \left( \frac{1}{1 + e^{-z}} \right) \cdot \left( \frac{e^{-z}}{1 + e^{-z}} \right) \\
   &= \sigma(z) \cdot \left( \frac{(1 + e^{-z}) - 1}{1 + e^{-z}} \right) \\
   &= \sigma(z) \cdot \left( 1 - \frac{1}{1 + e^{-z}} \right) \\
   &= \sigma(z) \cdot (1 - \sigma(z)) \quad \blacksquare
   \end{aligned}$$

   *Significance*: Evaluating the derivative requires no expensive recalculation of $e^{-z}$—once the forward pass computes $\hat{y} = \sigma(z)$, the derivative is simply $\hat{y}(1 - \hat{y})$. Its maximum value is at $z=0$, where $\sigma'(0) = 0.5 \times 0.5 = 0.25$.

---

## 3. Odds, Log-Odds, and the Logit Link

To understand why the sigmoid function is not an arbitrary choice, let us view it through the lens of **Generalized Linear Models (GLMs)** and probability odds.

### Probability vs. Odds
Let $p = P(Y=1|x)$ be the probability of success. The **odds** of an event is the ratio of the probability that the event occurs to the probability that it does not occur:

$$\text{Odds} = \frac{p}{1 - p}$$

| Probability $p$ | Odds $\frac{p}{1-p}$ | Verbal Meaning |
|---|---|---|
| $0.10$ | $\frac{0.1}{0.9} = \frac{1}{9} \approx 0.111$ | 1 to 9 odds against |
| $0.50$ | $\frac{0.5}{0.5} = 1.0$ | Even odds (1 to 1) |
| $0.80$ | $\frac{0.8}{0.2} = 4.0$ | 4 to 1 odds in favor |
| $0.99$ | $\frac{0.99}{0.01} = 99.0$ | 99 to 1 odds in favor |

While probability is bounded within $[0, 1]$, odds are bounded within $[0, +\infty)$.

### The Logit Function (Log-Odds)
Taking the natural logarithm of the odds produces the **logit** function:

$$\text{logit}(p) = \ln\left( \frac{p}{1 - p} \right)$$

The logit maps $(0, 1) \to (-\infty, +\infty)$. Now the range is the entire real line!
Therefore, we can model the log-odds as a linear combination of features:

$$\boxed{\ln\left( \frac{p}{1 - p} \right) = w^T x + b}$$

Solving this equation for $p$:
$$\begin{aligned}
\frac{p}{1 - p} &= e^{w^T x + b} \\
p &= (1 - p) e^{w^T x + b} \\
p &= e^{w^T x + b} - p \cdot e^{w^T x + b} \\
p(1 + e^{w^T x + b}) &= e^{w^T x + b} \\
p &= \frac{e^{w^T x + b}}{1 + e^{w^T x + b}} = \frac{1}{1 + e^{-(w^T x + b)}} = \sigma(w^T x + b)
\end{aligned}$$

**Conclusion**: The sigmoid function is the direct algebraic inverse of the logit link function! 
$$\sigma = \text{logit}^{-1}$$

### Interpreting Coefficients (Odds Ratios)
In linear regression, $w_j$ represents the expected change in $y$ for a 1-unit increase in $x_j$.
In logistic regression:
$$\ln(\text{Odds}_{x_j + 1}) - \ln(\text{Odds}_{x_j}) = w_j$$
Exponentiating both sides:
$$\frac{\text{Odds}(x_j + 1)}{\text{Odds}(x_j)} = e^{w_j}$$

- $e^{w_j}$ is the **Odds Ratio (OR)**.
- If $w_j = 0.693$, then $e^{w_j} \approx 2.0$: a 1-unit increase in $x_j$ doubles the odds of the positive outcome (holding all other features constant).
- If $w_j = 0$, then $e^{w_j} = 1$: feature $x_j$ has no effect on the odds.
- If $w_j < 0$, then $e^{w_j} < 1$: increasing $x_j$ decreases the odds.

---

## 4. Maximum Likelihood Estimation & Binary Cross-Entropy

### The Bernoulli Likelihood
For a single observation $(x^{(i)}, y^{(i)})$ where $y^{(i)} \in \{0, 1\}$, let:
$$P(Y = 1 | x^{(i)}; w) = \hat{y}^{(i)} = \sigma(w^T x^{(i)} + b)$$
$$P(Y = 0 | x^{(i)}; w) = 1 - \hat{y}^{(i)} = 1 - \sigma(w^T x^{(i)} + b)$$

We can express these two conditional statements compactly as a single **Bernoulli probability mass function**:

$$P(Y = y^{(i)} | x^{(i)}; w) = \left( \hat{y}^{(i)} \right)^{y^{(i)}} \cdot \left( 1 - \hat{y}^{(i)} \right)^{1 - y^{(i)}}$$

Notice:
- If $y^{(i)} = 1$: $P = (\hat{y}^{(i)})^1 (1 - \hat{y}^{(i)})^0 = \hat{y}^{(i)}$.
- If $y^{(i)} = 0$: $P = (\hat{y}^{(i)})^0 (1 - \hat{y}^{(i)})^1 = 1 - \hat{y}^{(i)}$.

Assuming all $n$ training observations are **independent and identically distributed (i.i.d.)**, the joint probability of observing the entire dataset (the **Likelihood** $\mathcal{L}(w)$) is the product of individual probabilities:

$$\mathcal{L}(w) = \prod_{i=1}^n P(Y = y^{(i)} | x^{(i)}; w) = \prod_{i=1}^n \left( \hat{y}^{(i)} \right)^{y^{(i)}} \left( 1 - \hat{y}^{(i)} \right)^{1 - y^{(i)}}$$

### The Log-Likelihood
Products of small probabilities underflow numerically. Taking the natural log converts the product into a sum:

$$\ell(w) = \ln \mathcal{L}(w) = \sum_{i=1}^n \left[ y^{(i)} \ln \hat{y}^{(i)} + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

### The Binary Cross-Entropy Loss Function (Log Loss)
In machine learning, optimization frameworks are standardly framed as **minimizing loss** rather than maximizing utility. We multiply the log-likelihood by $-\frac{1}{n}$ to define the **Binary Cross-Entropy (BCE) Loss** or **Log Loss** $J(w)$:

$$\boxed{J(w) = -\frac{1}{n} \ell(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y^{(i)} \ln \left(\hat{y}^{(i)}\right) + (1 - y^{(i)}) \ln\left(1 - \hat{y}^{(i)}\right) \right]}$$

```
Loss penalty for a single sample:
  Loss
  ∞ ┼ \
    │  \
    │   \   When true y = 1: Loss = -ln(ŷ)
    │    \
    │     `.
  0 ┼───────`─────────────► ŷ (Predicted Probability)
    0.0     0.5     1.0
  Predicting ŷ=0 when y=1 incurs an INFINITE loss penalty!
```

---

## 5. Why Not Mean Squared Error for Logistic Regression?

What if we used MSE with the sigmoid function?
$$J_{\text{MSE}}(w) = \frac{1}{2n} \sum_{i=1}^n \left( y^{(i)} - \sigma(w^T x^{(i)} + b) \right)^2$$

There are two major reasons MSE is not used with sigmoid classification:

1. **Non-Convexity (Local Minima)**:
   In linear regression, MSE is a quadratic function of $w$, creating a bowl-shaped convex surface with a single global minimum.
   When $w$ is nested inside the nonlinear sigmoid function $\sigma(w^T x)$, the composition $(y - \sigma(w^T x))^2$ is **non-convex**. It creates plateaus, saddle points, and multiple local minima where gradient descent can get stuck.
   In contrast, **Binary Cross-Entropy is strictly convex** with respect to $w$, guaranteeing that any stationary point ($\nabla J = 0$) is the unique global minimum!

2. **Vanishing Gradient / Slow Learning**:
   Taking the derivative of $J_{\text{MSE}}$ with respect to $w_j$ introduces $\sigma'(z) = \sigma(z)(1 - \sigma(z))$:
   $$\frac{\partial J_{\text{MSE}}}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (\sigma(z^{(i)}) - y^{(i)}) \cdot \underbrace{\sigma(z^{(i)})(1 - \sigma(z^{(i)}))}_{\to 0 \text{ when } |z| \text{ is large}} \cdot x_j^{(i)}$$
   If the model makes a confident yet completely wrong prediction (e.g. $z = 10 \implies \hat{y} \approx 0.9999$ when true $y = 0$), then $\sigma(z)(1 - \sigma(z)) \approx 0$. The gradient vanishes, and gradient descent will barely update the weights!
   Under Log Loss, this vanishing term cancels out perfectly, as shown below.

---

## 6. Analytical Derivation of the Log Loss Gradient

Let us compute the gradient of the binary cross-entropy loss function with respect to weight parameter $w_j$.

Let $z^{(i)} = w^T x^{(i)} + b = \sum_{k=1}^p w_k x_k^{(i)} + b$ and $\hat{y}^{(i)} = \sigma(z^{(i)})$.

For a single observation $i$, the loss is:
$$L^{(i)} = - \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$

By the multivariate chain rule:
$$\frac{\partial L^{(i)}}{\partial w_j} = \frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} \cdot \frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} \cdot \frac{\partial z^{(i)}}{\partial w_j}$$

Let us compute each factor:

### Step 1: Derivative of Loss with respect to $\hat{y}^{(i)}$
$$\frac{\partial L^{(i)}}{\partial \hat{y}^{(i)}} = - \left( \frac{y^{(i)}}{\hat{y}^{(i)}} - \frac{1 - y^{(i)}}{1 - \hat{y}^{(i)}} \right) = - \frac{y^{(i)}(1 - \hat{y}^{(i)}) - (1 - y^{(i)})\hat{y}^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})} = - \frac{y^{(i)} - \hat{y}^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)}} = \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)}}$$

### Step 2: Derivative of Sigmoid with respect to $z^{(i)}$
As derived earlier:
$$\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}} = \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

### Step 3: Derivative of Linear Combination with respect to $w_j$
$$\frac{\partial z^{(i)}}{\partial w_j} = \frac{\partial}{\partial w_j}\left( \sum_{k=1}^p w_k x_k^{(i)} + b \right) = x_j^{(i)}$$

### Step 4: Multiply the Factors
Notice the cancellation:
$$\begin{aligned}
\frac{\partial L^{(i)}}{\partial w_j} &= \left( \frac{\hat{y}^{(i)} - y^{(i)}}{\hat{y}^{(i)}(1 - \hat{y}^{(i)})} \right) \cdot \left( \hat{y}^{(i)}(1 - \hat{y}^{(i)}) \right) \cdot x_j^{(i)} \\
&= (\hat{y}^{(i)} - y^{(i)}) \cdot x_j^{(i)}
\end{aligned}$$

Summing over all $n$ samples and dividing by $n$:
$$\boxed{\frac{\partial J(w)}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n \left( \hat{y}^{(i)} - y^{(i)} \right) x_j^{(i)}}$$

For the bias term $b$ (where $x_0^{(i)} = 1$):
$$\boxed{\frac{\partial J(w)}{\partial b} = \frac{1}{n} \sum_{i=1}^n \left( \hat{y}^{(i)} - y^{(i)} \right)}$$

### Vectorized Matrix Form
Let $X \in \mathbb{R}^{n \times p}$ be the feature matrix, $y \in \mathbb{R}^n$ the true binary labels, and $\hat{y} = \sigma(Xw + b) \in \mathbb{R}^n$ the predicted probabilities:

$$\boxed{\nabla_w J(w) = \frac{1}{n} X^T (\hat{y} - y)}$$

> [!NOTE]
> Look closely at this result: the gradient formula $\frac{1}{n} X^T (\hat{y} - y)$ is **identical in functional form to the OLS gradient** from Day 1 and Day 3! 
> The only difference is that for OLS, $\hat{y} = Xw$, whereas for logistic regression, $\hat{y} = \sigma(Xw)$. This universal elegance is a hallmark of Generalized Linear Models with canonical link functions.

---

## 7. Optimization Algorithms

Unlike linear regression ($X^T X w = X^T y$), the equation $\frac{1}{n} X^T (\sigma(Xw) - y) = 0$ is **transcendental** (nonlinear in $w$). There is **no closed-form analytical solution**. We must find $w^*$ numerically.

### 1. Batch Gradient Descent (First-Order)
We update weights iteratively in the opposite direction of the gradient:
$$w \leftarrow w - \alpha \nabla_w J(w) = w - \frac{\alpha}{n} X^T (\hat{y} - y)$$
$$b \leftarrow b - \alpha \frac{\partial J}{\partial b} = b - \frac{\alpha}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})$$
where $\alpha > 0$ is the learning rate.

### 2. Newton-Raphson and IRLS (Second-Order)
To converge in far fewer iterations (often 4–8 iterations), we can use Newton's method:
$$w^{(t+1)} = w^{(t)} - H^{-1} \nabla J(w^{(t)})$$
where $H$ is the **Hessian matrix** of second partial derivatives:
$$H_{jk} = \frac{\partial^2 J}{\partial w_j \partial w_k} = \frac{1}{n} \sum_{i=1}^n x_j^{(i)} x_k^{(i)} \hat{y}^{(i)}(1 - \hat{y}^{(i)})$$

In matrix form:
$$H = \frac{1}{n} X^T D X$$
where $D = \text{diag}\left( \hat{y}^{(1)}(1 - \hat{y}^{(1)}), \dots, \hat{y}^{(n)}(1 - \hat{y}^{(n)}) \right)$ is an $n \times n$ diagonal weight matrix.

Because $\hat{y}^{(i)} \in (0, 1)$, all diagonal entries $D_{ii} = \hat{y}^{(i)}(1 - \hat{y}^{(i)}) > 0$.
For any non-zero vector $v \in \mathbb{R}^p$:
$$v^T H v = \frac{1}{n} v^T X^T D X v = \frac{1}{n} (Xv)^T D (Xv) = \frac{1}{n} \sum_{i=1}^n D_{ii} (Xv)_i^2 \ge 0$$

Thus, the Hessian $H$ is **positive semi-definite everywhere**, proving mathematically that **Binary Cross-Entropy is convex**.

In statistics, Newton's method for logistic regression is known as **Iteratively Reweighted Least Squares (IRLS)** because at each iteration $t$, finding the update is equivalent to solving a weighted least-squares problem where weights are given by $D$.

---

## 8. Decision Boundaries and Threshold Tuning

### Geometry of the Decision Boundary
By default, we assign an observation to class 1 if its predicted probability meets or exceeds $0.5$:
$$\hat{y}_{\text{pred}} = \begin{cases} 1 & \text{if } \sigma(w^T x + b) \ge 0.5 \\ 0 & \text{if } \sigma(w^T x + b) < 0.5 \end{cases}$$

Recall that $\sigma(z) \ge 0.5 \iff z \ge 0$.
Therefore, the **decision boundary** is the geometric surface where $z = 0$:
$$w^T x + b = 0$$

For a 2D problem ($x_1, x_2$):
$$w_1 x_1 + w_2 x_2 + b = 0 \implies x_2 = -\frac{w_1}{w_2} x_1 - \frac{b}{w_2}$$
This is a straight line in 2D space (a hyperplane in $p$-dimensional space). Hence, **Logistic Regression is a linear classifier**.

```
2D Feature Space:
  x2 ▲
     │       ● (Class 1)     ●           ●
     │              ●            ●
     │   w^T x + b > 0  (ŷ > 0.5)
     │─────────────────────────────────── Decision Boundary: w^T x + b = 0 (ŷ = 0.5)
     │   w^T x + b < 0  (ŷ < 0.5)
     │         ○            ○     ○
     │      ○       ○ (Class 0)
     └───────────────────────────────────► x1
```

### Threshold Tuning for Asymmetric Costs
The default threshold $\tau = 0.5$ assumes False Positives (predicting 1 when true 0) and False Negatives (predicting 0 when true 1) have identical real-world costs.

In practice, costs are rarely symmetric:
- **Medical Screening (e.g. Cancer Detection)**: A False Negative (missing cancer) can be fatal. A False Positive (extra biopsy) is stressful but manageable. We lower the threshold (e.g., $\tau = 0.1$ or $0.2$) to maximize recall.
- **Spam Filtering**: A False Positive (sending an important job offer to spam) is far worse than a False Negative (seeing a spam email in the inbox). We raise the threshold (e.g., $\tau = 0.8$ or $0.9$).

Given a cost matrix:
$$\text{Expected Cost}(\tau) = C_{\text{FP}} \cdot \text{FP}(\tau) + C_{\text{FN}} \cdot \text{FN}(\tau)$$
We can sweep $\tau \in [0.01, 0.99]$ on validation data to pick the cost-optimal operating point.

---

## 9. The Perfect Separation Problem & Regularization

### Why Separation Causes Weights to Blow Up
Suppose the training data is **linearly separable**—there exists a hyperplane that classifies every single training point with 100% accuracy.

For every point $i$:
- If $y^{(i)} = 1$, then $w^T x^{(i)} + b > 0$.
- If $y^{(i)} = 0$, then $w^T x^{(i)} + b < 0$.

What happens if we multiply the weight vector by a scalar constant $c > 1$: $w' = c \cdot w$?
The sign of $w'^T x + b'$ does not change, so accuracy remains 100%.
However, as $c \to \infty$:
- For $y^{(i)} = 1$: $z^{(i)} \to +\infty \implies \hat{y}^{(i)} \to 1 \implies \ln(\hat{y}^{(i)}) \to 0$.
- For $y^{(i)} = 0$: $z^{(i)} \to -\infty \implies \hat{y}^{(i)} \to 0 \implies \ln(1 - \hat{y}^{(i)}) \to 0$.

The loss $J(c \cdot w) \to 0$ asymptotically, but it **never reaches 0 for finite $w$**!
As a result, unregularized gradient descent will push the weights toward $\pm \infty$, resulting in numerical overflow and severe overfitting.

### Regularization in Logistic Regression
To prevent weight explosion and reduce variance, we add an $L_2$ or $L_1$ penalty:

$$J_{\text{reg}}(w) = -\frac{1}{n} \sum_{i=1}^n \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right] + \frac{\lambda}{2} \|w\|_2^2$$

In `scikit-learn`, `LogisticRegression` uses parameter $C$, the **inverse regularization strength**:
$$C = \frac{1}{\lambda}$$
$$\min_w C \cdot \text{Loss}(w) + \frac{1}{2} \|w\|_2^2$$
- Small $C$ (e.g. $C=0.01$): Heavy regularization, smaller weights, higher bias, lower variance.
- Large $C$ (e.g. $C=1000$): Weak regularization, fits training data closely, potential high variance.

---

## 10. Multi-Class Classification

Logistic regression is fundamentally binary, but generalizes to $K > 2$ classes in two standard ways:

### Method 1: One-vs-Rest (OvR / One-vs-All)
- Train $K$ separate binary logistic regression models.
- Model $k$ is trained with positive class $y=k$ and negative class "all other classes".
- For a new query $x$, evaluate all $K$ models: $\hat{p}_k = \sigma(w_k^T x + b_k)$.
- Predict the class with the highest probability: $\hat{y} = \arg\max_k \hat{p}_k$.

### Method 2: Multinomial Logistic Regression (Softmax)
Instead of $K$ independent classifiers, we model the joint probability distribution using the **Softmax function**:

$$P(Y = k | x) = \frac{e^{w_k^T x + b_k}}{\sum_{j=1}^K e^{w_j^T x + b_j}}$$

The loss function becomes the **Categorical Cross-Entropy**:
$$J(W) = -\frac{1}{n} \sum_{i=1}^n \sum_{k=1}^K \mathbb{I}(y^{(i)} = k) \ln\left( P(Y = k | x^{(i)}) \right)$$

---

## Summary Cheat Sheet

| Concept | Mathematical Formula | Key Note |
|---|---|---|
| **Sigmoid Function** | $\sigma(z) = \frac{1}{1 + e^{-z}}$ | Maps $(-\infty, \infty) \to (0, 1)$; derivative $\sigma'(z) = \sigma(z)(1 - \sigma(z))$. |
| **Log-Odds (Logit)** | $\ln\left(\frac{p}{1-p}\right) = w^T x + b$ | Linear model on log-odds; $e^{w_j}$ is the Odds Ratio for feature $j$. |
| **Hypothesis** | $\hat{y} = P(Y=1\|x) = \sigma(w^T x + b)$ | Output is interpreted as a calibrated posterior class probability. |
| **Log Loss (BCE)** | $J(w) = -\frac{1}{n}\sum [y \ln \hat{y} + (1-y)\ln(1-\hat{y})]$ | Strictly convex loss derived from Bernoulli likelihood. |
| **Gradient** | $\nabla_w J(w) = \frac{1}{n} X^T (\hat{y} - y)$ | Same functional form as OLS, but with $\hat{y} = \sigma(Xw+b)$. |
| **Decision Boundary** | $w^T x + b = 0$ (for $\tau = 0.5$) | Hyperplane partitioning the feature space into two half-spaces. |
| **Regularization ($C$)** | $C = \frac{1}{\lambda}$ | Prevents weights from diverging to $\pm\infty$ on separable datasets. |
