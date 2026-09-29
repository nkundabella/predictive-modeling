# Day 6 Exercises — Logistic Regression and the Sigmoid Function

Work through these in order. Attempt every question before opening `solutions/`.

---

## Theory

**1. Sigmoid Function Symmetry & Analytical Derivative:**
   - The standard logistic sigmoid is $\sigma(z) = \frac{1}{1 + e^{-z}}$.
   - Prove algebraically that $\sigma(-z) = 1 - \sigma(z)$ for all $z \in \mathbb{R}$.
   - Prove step-by-step using calculus (quotient rule or power rule) that:
     $$\frac{d\sigma(z)}{dz} = \sigma(z) \cdot (1 - \sigma(z))$$
   - Find the value of $z$ that maximizes $\sigma'(z)$, and compute this maximum value. Explain why this maximum value is important in neural networks (hint: vanishing gradients).

**2. Complete Chain Rule Derivation of the Log Loss Gradient:**
   - Consider the Binary Cross-Entropy loss for $n$ samples:
     $$J(w, b) = -\frac{1}{n} \sum_{i=1}^n \left[ y^{(i)} \ln(\hat{y}^{(i)}) + (1 - y^{(i)}) \ln(1 - \hat{y}^{(i)}) \right]$$
     where $\hat{y}^{(i)} = \sigma(z^{(i)})$ and $z^{(i)} = \sum_{j=1}^p w_j x_j^{(i)} + b$.
   - Write out the three chain rule factors: $\frac{\partial J}{\partial \hat{y}^{(i)}}$, $\frac{\partial \hat{y}^{(i)}}{\partial z^{(i)}}$, and $\frac{\partial z^{(i)}}{\partial w_j}$.
   - Show in full algebraic detail how the $\hat{y}^{(i)}(1 - \hat{y}^{(i)})$ denominator cancels out, yielding:
     $$\frac{\partial J}{\partial w_j} = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)}) x_j^{(i)}$$
   - Write the full parameter gradient in vectorized matrix notation.

**3. The Hessian Matrix & Proof of Convexity:**
   - Compute the second partial derivatives:
     $$H_{jk} = \frac{\partial^2 J}{\partial w_j \partial w_k}$$
   - Express the Hessian matrix $H \in \mathbb{R}^{p \times p}$ in compact matrix form using $X$ and a diagonal matrix $D$. What are the diagonal entries $D_{ii}$?
   - Prove that for any vector $v \in \mathbb{R}^p$, $v^T H v \ge 0$.
   - What does this prove about the geometry of the Binary Cross-Entropy loss surface and the existence of local minima?

**4. Odds Ratios & Statistical Interpretation:**
   - A credit scoring model estimates the probability of loan default using:
     $$\ln\left(\frac{p}{1-p}\right) = -2.5 + 0.8 \cdot (\text{DTI}) - 0.04 \cdot (\text{FICO\_score}) + 1.4 \cdot (\text{Prior\_Default})$$
     where $\text{DTI}$ is the debt-to-income ratio (e.g. 0.45), $\text{Prior\_Default} \in \{0, 1\}$, and $\text{FICO\_score}$ is a credit score (e.g. 700).
   - Interpret the coefficient $1.4$ for `Prior_Default` in terms of the Odds Ratio ($e^{1.4}$).
   - If applicant A and applicant B have identical DTI and FICO scores, but applicant A has a prior default while applicant B does not, how many times higher are applicant A's odds of defaulting?
   - Compute the exact default probability $p$ for an applicant with $\text{DTI} = 0.50$, $\text{FICO\_score} = 650$, and $\text{Prior\_Default} = 1$.

**5. The Perfect Separation Pathology:**
   - Suppose a binary dataset is strictly linearly separable: there exists $(w, b)$ such that $w^T x^{(i)} + b > 0$ for all $y^{(i)} = 1$, and $w^T x^{(i)} + b < 0$ for all $y^{(i)} = 0$.
   - Show why the unregularized maximum likelihood estimator does not have a finite solution (i.e. $\|w\|_2 \to \infty$).
   - Explain how adding an $L_2$ regularization penalty $\frac{\lambda}{2} \|w\|_2^2$ guarantees a unique, finite solution.

---

## Code

**6. Newton-Raphson / IRLS (Iteratively Reweighted Least Squares) from Scratch:**
   - Implement `LogisticRegressionIRLS(n_epochs=20, tol=1e-6, l2_reg=1e-4)`.
   - At each step $t$, compute:
     $$\hat{y} = \sigma(Xw + b)$$
     $$\nabla J = \frac{1}{n} X^T (\hat{y} - y) + \lambda w$$
     $$D_{ii} = \hat{y}_i (1 - \hat{y}_i)$$
     $$H = \frac{1}{n} X^T D X + \lambda I$$
     $$w \leftarrow w - H^{-1} \nabla J$$
   - Fit this model on a 2-feature dataset and count how many iterations Newton's method requires to converge compared to First-Order Gradient Descent.

**7. Cost-Sensitive Cutoff Threshold Optimizer:**
   - Write a function `optimize_threshold(y_true, y_prob, cost_fp, cost_fn)` that:
     1. Sweeps through thresholds $\tau \in [0.01, 0.99]$ in increments of $0.01$.
     2. Computes the Confusion Matrix (TN, FP, FN, TP) at each threshold.
     3. Calculates the total financial cost: $\text{Cost}(\tau) = \text{FP}(\tau) \cdot C_{\text{FP}} + \text{FN}(\tau) \cdot C_{\text{FN}}$.
     4. Returns the optimal threshold $\tau^*$, the minimum cost, and the cost at the naive $\tau=0.5$ baseline.
   - Run this on synthetic fraud data where $C_{\text{FN}} = \$1,000$ (fraudulent transaction missed) and $C_{\text{FP}} = \$25$ (customer verification SMS/call).

**8. One-vs-Rest (OvR) Multi-Class Classifier from Scratch:**
   - Implement a generic `OneVsRestClassifierScratch` class:
     1. Accepts any binary classifier class (such as `LogisticRegressionScratch`).
     2. In `.fit(X, y)`, detects unique classes $K$, creates $K$ binary label vectors $y_k = (y == k).astype(int)$, and fits $K$ binary models.
     3. In `.predict_proba(X)`, gets positive probability from each model and normalizes probabilities across rows so they sum to 1.
     4. In `.predict(X)`, returns $\arg\max_k P(Y=k|x)$.
   - Test on a 3-class synthetic dataset and compare performance against `sklearn.linear_model.LogisticRegression(multi_class='ovr')`.
