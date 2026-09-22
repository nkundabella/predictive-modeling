# Day 3 Exercises — Gradient Descent

Work through these in order. Attempt every question before opening `solutions/`.

---

## Theory

**1.** Write out the MSE loss function for a model `y_hat = w*x + b` with n data points.
Then derive `∂MSE/∂w` and `∂MSE/∂b` step-by-step using the chain rule.
Show every line — do not skip steps.

**2.** Explain in plain English what the gradient tells you.
If `∂MSE/∂w = -5.2`, which direction should w move to reduce the loss? By how much (given α = 0.1)?

**3.** Why does gradient descent require many iterations while OLS needs only one step?
Under what condition would you *prefer* gradient descent over OLS even for a linear model?

**4.** Describe what happens to the loss curve when the learning rate α is:
   a. Exactly right
   b. 10× too large
   c. 1000× too small

**5.** Explain why feature scaling improves gradient descent convergence.
Draw (or describe in words) the shape of the loss surface with and without scaling for a 2-feature dataset where one feature ranges from 0–1 and the other from 0–100,000.

---

## Code

**6.** Copy the `gradient_descent` function from `gradient_descent_demo.py`.
Add an **early stopping** mechanism: if the loss decreases by less than `tol=1e-6` for five consecutive epochs, stop early and return.
How many epochs does it take to converge on the demo dataset?

**7.** Implement gradient descent for **multiple features** (not just one x, but a vector x of p features).
The update rule becomes:

    w ← w - α * (-2/n) * Xᵀ(y - Xw - b)
    b ← b - α * (-2/n) * sum(y - Xw - b)

Use the house-price dataset from Day 2 (3 features). Compare the parameters you get after 500 epochs (α=0.05) with sklearn's `LinearRegression`. How close are they?

**8.** Run gradient descent 5 times with five different random initialisations for w and b (e.g. random values from N(0, 10)).
Does the final loss change? Does the final (w, b) change?
What does this tell you about MSE as a loss function?

**9.** Implement **stochastic gradient descent (SGD)** — update w and b using one randomly chosen sample per epoch instead of all n samples.
Compare the loss curve of SGD vs batch GD on the same dataset.
Plot both on the same axes. Which is noisier? Which gets closer to the minimum in 300 "data-touches" (i.e. 300 individual sample evaluations)?

**10.** The gradient descent update can be written as a matrix operation:

    θ ← θ - α * ∇L(θ)

where `θ = [w, b]` is the parameter vector.

Vectorise your single-feature GD implementation so that w and b are updated in a single numpy line (no separate `dw`, `db` variables). Verify it produces identical results.

---

## Reflection

Write a short paragraph (5–8 sentences):
Before this session, how did you think models "learn" their parameters?
Has your mental model changed?
Gradient descent is described as following the negative gradient — can you think of a real-world analogy where you use a similar strategy (following local information iteratively to reach a global goal)?
What would happen if the loss surface were non-convex (had multiple valleys)?
