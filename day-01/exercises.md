# Day 1 Exercises — Linear Regression

Work through these in order. The conceptual questions test understanding; the coding questions test application. Attempt all of them before checking solutions/.

---

## Theory

**1.** Derive the OLS formula for w1 from first principles.
Start with RSS = sum((yi - (w1*xi + w0))^2), differentiate with respect to w1 and w0, set both to zero, and solve the resulting system of two equations. Show each step.

**2.** Why does OLS minimise *squared* residuals rather than absolute residuals?
Give two reasons — one mathematical, one practical.

**3.** Given these five points:

    x = [1, 2, 3, 4, 5]
    y = [2, 4, 5, 4, 5]

Compute w1 and w0 by hand. Show your working.
Then state: does a line fit this data well? How do you know without computing R^2?

**4.** R^2 is defined as 1 - RSS/TSS. What is TSS, and what model does it represent?
In other words, what are you comparing your model against when you compute R^2?

**5.** Explain what it means for R^2 to be negative. Is that possible? When would it occur?

---

## Code

**6.** Add a ninth observation to the dataset: x=9, y=98. Re-run the script.
Report: how did w1, w0, and R^2 change? Explain why in one paragraph.

**7.** Fit a linear model on this dataset:

    x (temperature °C): [15, 20, 25, 30, 35]
    y (ice cream sales): [30, 50, 80, 120, 150]

Compute w1 and w0 manually (paper or code — no sklearn yet).
Then fit with sklearn and verify the values match.
Plot the data and the fitted line. Does linear regression seem appropriate here?

**8.** What happens when all y-values are identical?

    y = [60, 60, 60, 60, 60, 60, 60, 60]

Without running the code, predict: what will w1 be? What will R^2 be?
Run the script to confirm. Explain the result mathematically.

**9.** For a perfectly linear dataset:

    x = [0, 1, 2, 3, 4]
    y = [0, 2, 4, 6, 8]

What are w1 and w0? What is R^2? State your answers before running the code, then verify.

**10.** Examine the residual plot produced by the script. If residuals showed a clear curve (e.g., they start negative, go positive in the middle, then negative again), what would that tell you about the linear model? What would you do next?

---

## Reflection

Write a short paragraph (5–8 sentences) on the following:
What assumptions did you find most surprising? Which part of the derivation was hardest to follow? What is one question you still have that the notes did not answer?

This is not graded — it is to build the habit of identifying gaps in understanding before they compound.
