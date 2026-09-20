# Day 1 — Solutions

---

## Theory

**1. OLS Derivation**

RSS = sum( (yi - w1*xi - w0)^2 )

Partial derivative with respect to w0:
  dRSS/dw0 = -2 * sum(yi - w1*xi - w0) = 0
  => sum(yi) - w1*sum(xi) - n*w0 = 0
  => w0 = y_mean - w1 * x_mean          ... (i)

Partial derivative with respect to w1:
  dRSS/dw1 = -2 * sum( xi*(yi - w1*xi - w0) ) = 0
  => sum(xi*yi) - w1*sum(xi^2) - w0*sum(xi) = 0

Substitute (i) into the second equation and simplify:
  sum(xi*yi) - w1*sum(xi^2) - (y_mean - w1*x_mean)*sum(xi) = 0
  sum(xi*yi) - y_mean*sum(xi) = w1*(sum(xi^2) - x_mean*sum(xi))

Left side  = sum( xi*(yi - y_mean) ) = sum( (xi - x_mean)*(yi - y_mean) )
Right side = w1 * sum( (xi - x_mean)^2 )

Therefore:
  w1 = sum( (xi - x_mean)*(yi - y_mean) ) / sum( (xi - x_mean)^2 )


**2. Why squared residuals?**

Mathematical reason: the squared loss is differentiable everywhere, which allows calculus-based optimisation (gradient descent, closed-form solutions). The absolute value function has a non-differentiable kink at zero.

Practical reason: squaring penalises large errors more than small ones. In many applications, a single large error is worse than several small ones. The OLS solution corresponds to the maximum likelihood estimator under a Gaussian noise model, which has strong theoretical justification.


**3. Manual OLS on the five-point dataset**

x = [1, 2, 3, 4, 5], y = [2, 4, 5, 4, 5]
x_mean = 3, y_mean = 4

(xi - x_mean): [-2, -1, 0, 1, 2]
(yi - y_mean): [-2,  0, 1, 0, 1]

Numerator   = (-2*-2) + (-1*0) + (0*1) + (1*0) + (2*1) = 4+0+0+0+2 = 6
Denominator = 4+1+0+1+4 = 10

w1 = 6/10 = 0.6
w0 = 4 - 0.6*3 = 4 - 1.8 = 2.2

Line: y_hat = 0.6*x + 2.2

The fit is imperfect — y is not strictly increasing with x (it drops from 5 to 4).
Visual inspection is enough: the scatter shows no clean trend, so residuals will be noticeable.


**4. What is TSS?**

TSS = sum( (yi - y_mean)^2 )

This is the total variance in y. It represents the error of the simplest possible model: predicting y_mean for every observation (a horizontal line).

R^2 = 1 - RSS/TSS measures how much of that baseline variance your model eliminates.


**5. Can R^2 be negative?**

Yes. If RSS > TSS, then R^2 < 0. This means the model predicts worse than simply predicting the mean for every observation. It can occur when the model is constrained (e.g. forced through the origin) or when a model trained on different data is evaluated on a new dataset.

---

## Code

**6.** Adding x=9, y=98:

The slope w1 will increase slightly because the new point is consistent with the trend and lies slightly above the existing line. R^2 will also increase slightly because the variance explained grows. The intercept w0 may shift marginally. The exact values depend on how much the new point deviates from the existing fit.

**7.** Ice cream dataset manual solution:

x = [15, 20, 25, 30, 35], y = [30, 50, 80, 120, 150]
x_mean = 25, y_mean = 86

Numerator   = (-10*-56) + (-5*-36) + (0*-6) + (5*34) + (10*64)
            = 560 + 180 + 0 + 170 + 640 = 1550
Denominator = 100 + 25 + 0 + 25 + 100 = 250

w1 = 1550/250 = 6.2
w0 = 86 - 6.2*25 = 86 - 155 = -69

y_hat = 6.2*x - 69

A linear fit is plausible here. The data increases roughly linearly with temperature, though a residual plot would confirm whether a nonlinear term would improve fit.

**8.** Constant y:

y_mean = 60. Every deviation (yi - y_mean) = 0, so the numerator of the w1 formula is 0.
w1 = 0, w0 = 60. The model predicts 60 regardless of x.

TSS = sum((60-60)^2) = 0. R^2 = 1 - RSS/TSS is undefined (0/0).
sklearn returns NaN or 0 depending on version. This is a degenerate case — no variance in y means there is nothing for the model to explain.

**9.** Perfect linear dataset:

x = [0,1,2,3,4], y = [0,2,4,6,8] follows y = 2x exactly.

w1 = 2, w0 = 0, R^2 = 1.0 (no residuals).

**10.** Curved residual pattern:

A curved pattern in the residual plot indicates the linear model is misspecified — the true relationship is nonlinear. The model is systematically over-predicting and under-predicting in different regions of x. Next steps: add a polynomial feature (x^2) and refit, or apply a nonlinear model.
