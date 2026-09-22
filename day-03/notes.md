# Day 3 — Gradient Descent

## Why Do We Need Gradient Descent?

On Day 1 we solved linear regression using the **closed-form OLS formula** — a single equation that gives the exact answer in one step. That works beautifully for linear regression, but it has limits:

1. **Matrix inversion is expensive.** The OLS solution requires computing `(XᵀX)⁻¹`, which costs O(p³) where p is the number of features. With thousands of features this becomes prohibitive.
2. **Not all loss functions have a closed-form solution.** Logistic regression, neural networks, and most modern models have no formula you can write down and solve algebraically.

We need an iterative method that can minimise *any* differentiable loss function. That method is **gradient descent**.

---

## The Core Idea

Imagine you are standing on a hilly landscape in thick fog. You want to reach the lowest valley (the minimum of the loss function). You can't see the whole landscape — you can only feel the slope under your feet. The strategy:

1. Feel the slope at your current position (compute the gradient).
2. Take a step in the downhill direction (move opposite to the gradient).
3. Repeat until the ground feels flat (gradient ≈ 0).

This is gradient descent.

---

## Loss Function: Mean Squared Error

For linear regression with parameters `w` (weights) and `b` (bias):

    y_hat_i = w · x_i + b

    MSE(w, b) = (1/n) * sum( (y_i - y_hat_i)² )

MSE is a smooth, convex bowl in the parameter space. Gradient descent is guaranteed to find the global minimum for this loss.

---

## The Gradient

The gradient of MSE with respect to `w` and `b` tells us the slope of the loss surface in every direction.

For a single weight w and bias b (one feature):

    ∂MSE/∂w = (-2/n) * sum( (y_i - y_hat_i) * x_i )
    ∂MSE/∂b = (-2/n) * sum( (y_i - y_hat_i) )

In matrix form (p features):

    ∇_w MSE = (-2/n) * Xᵀ(y - ŷ)
    ∂MSE/∂b = (-2/n) * sum(y - ŷ)

The negative sign means the gradient points *uphill*. To go downhill, we move in the **negative gradient direction**.

---

## The Update Rule

At each step (called an **epoch** or **iteration**):

    w ← w - α * ∂MSE/∂w
    b ← b - α * ∂MSE/∂b

Where **α (alpha)** is the **learning rate** — a small positive number controlling step size.

This is the single equation that defines gradient descent. Every deep learning framework (PyTorch, TensorFlow, JAX) ultimately reduces to this update applied millions of times.

---

## The Learning Rate

The learning rate α is the most critical hyperparameter in gradient descent.

    α too large  → steps overshoot the minimum; loss oscillates or diverges
    α too small  → convergence is very slow; you need many more iterations
    α just right → smooth, fast convergence to the minimum

Typical starting values: α = 0.01 or 0.001.

There is no universal correct learning rate — it depends on the scale of the data, the loss function, and the model. We will cover learning rate schedules in later days.

---

## Convergence

Training stops when:

1. A fixed number of epochs is reached (most common in practice), or
2. The gradient norm falls below a threshold (`||∇L|| < ε`), or
3. The loss stops improving between epochs (early stopping — covered later).

On a convex loss like MSE, gradient descent always converges to the global minimum, provided α is small enough.

---

## Batch, Stochastic, and Mini-Batch Gradient Descent

The gradient can be computed on different amounts of data per step:

| Variant               | Data used per update     | Characteristics                                           |
|-----------------------|--------------------------|-----------------------------------------------------------|
| **Batch GD**          | All n training samples   | Exact gradient; slow per epoch for large n                |
| **Stochastic GD (SGD)** | 1 random sample        | Very noisy; fast; can escape local minima                 |
| **Mini-Batch GD**     | k samples (e.g. 32–256) | Balance of stability and speed; used by all modern DL     |

In this course, Day 3 uses **batch gradient descent** for clarity. Mini-batch SGD is covered on Day 16.

---

## Visualising the Loss Surface

For a single-feature problem we can plot loss as a function of w and b:

    - x-axis: values of w
    - y-axis: values of b  
    - z-axis / colour: MSE

The trajectory of gradient descent on this surface is a sequence of points spiralling into the minimum. Plotting this trajectory is a standard diagnostic tool.

A **loss curve** (loss vs. epoch number) should:
- Decrease monotonically (for batch GD with a good α)
- Level off as it approaches the minimum
- Never increase by a large amount (if it does, α is too large)

---

## Gradient Descent vs OLS — When to Use Which

| Criterion                  | OLS (closed-form)            | Gradient Descent              |
|----------------------------|------------------------------|-------------------------------|
| Problem type               | Linear regression only       | Any differentiable loss       |
| Solution quality           | Exact global minimum         | Approximate (depends on α, epochs) |
| Computational cost         | O(p³) — bad for large p      | O(n·p) per epoch — scalable   |
| Dataset size               | Moderate (p ≤ ~10,000)       | Any size, including streaming |
| Implementation complexity  | Low                          | Higher (needs tuning)         |

For small linear regression problems, OLS is always preferable. Gradient descent becomes essential when p is large, when the dataset doesn't fit in memory, or when the model is not linear.

---

## Feature Scaling Matters

Gradient descent is sensitive to the scale of input features. If features differ greatly in magnitude (e.g. age in years vs. income in pounds), the loss surface becomes elongated and narrow, and gradient descent struggles.

    Without scaling: slow zigzag descent on an elongated bowl
    With scaling   : fast, direct descent into a circular bowl

Always standardise features before gradient descent:

    x_scaled = (x - mean(x)) / std(x)

(This is another reason the split-first rule from Day 2 matters: compute the mean and std on the training set only.)

---

## Files

- `gradient_descent_demo.py` — batch gradient descent on MSE from scratch, comparison with OLS, convergence plots
- `exercises.md` — problems
- `solutions/` — reference answers

## Next

Day 4 covers multiple linear regression — extending from one input feature to many, including the normal equation in matrix form and interpreting coefficients.
