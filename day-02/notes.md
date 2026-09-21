# Day 2 — Features, Labels, Train/Test Split, and Data Leakage

## Features and Labels

Every supervised learning problem has two sides:

- **Features** (also called inputs, predictors, X): the variables the model is given.
- **Label** (also called target, output, y): the variable the model is asked to predict.

Example — predicting house prices:

    Features X: square footage, number of bedrooms, postcode, age of property
    Label    y: sale price (£)

Choosing good features is often more important than choosing a fancy model. A well-chosen feature set with a simple linear model regularly beats a complex model on a poor feature set.

---

## The Generalisation Problem

On day 1 we fitted a model on all eight observations, then evaluated it on the same eight observations. This is fundamentally wrong.

**Why?** Any model can memorise its training data perfectly if it is complex enough. The number that actually matters is how well it performs on data it has *never seen*.

This is called **generalisation** — the model's ability to produce accurate predictions on new, unseen inputs.

Evaluating a model on its own training data always overestimates performance. The model has already "seen the answers."

---

## Train / Test Split

The standard fix is to hold out a portion of the data before training and never touch it until the very end.

    Full dataset (n rows)
    ├── Training set   → model is fitted on this   (~70–80% of rows)
    └── Test set       → model is evaluated on this (~20–30% of rows)

The split is done *randomly* so neither set is systematically biased.

**Rule**: the test set is locked away. You look at it exactly once — after training is complete.

Typical split ratios:

    80/20  — common general-purpose choice
    70/30  — used when the dataset is small (more test data = more reliable estimate)
    90/10  — used when the dataset is large and you can afford a smaller test set

### Scikit-learn: train_test_split

    from sklearn.model_selection import train_test_split

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,    # 20% held out for testing
        random_state=42   # seed for reproducibility
    )

    model.fit(X_train, y_train)           # train ONLY on training data
    y_pred = model.predict(X_test)        # evaluate on test data
    score  = r2_score(y_test, y_pred)     # true generalisation score

---

## Data Leakage

**Data leakage** occurs when information from outside the training set "leaks" into the model during training, causing it to appear better than it actually is.

There are two main types:

### 1. Target Leakage

A feature is included that is directly or indirectly derived from the label.

    Example: predicting whether a patient will be hospitalised,
    and including "number of hospital prescriptions" as a feature.
    If a patient is hospitalised, they receive prescriptions *because*
    they are ill — the feature encodes the outcome.

On training data the model achieves near-perfect accuracy.
In production (where you don't yet know the outcome), the feature
has a different meaning and the model collapses.

### 2. Train-Test Contamination

Preprocessing steps that "learn" from the full dataset are applied
before the split, smuggling test-set information into the training process.

    Bad (leaky) workflow:
        1. Compute mean and std of full dataset X
        2. Standardise full X using those stats
        3. Split into train/test
        4. Train model on X_train, evaluate on X_test

    The problem: step 1 used test-set rows to compute the mean and std.
    The model has indirectly seen the test set before evaluation.

    Correct workflow:
        1. Split into X_train and X_test
        2. Compute mean and std on X_train ONLY
        3. Standardise X_train with those stats
        4. Standardise X_test with the SAME stats (from training set)
        5. Train on X_train, evaluate on X_test

The test set should be treated as if it does not exist until the final evaluation.

---

## Why random_state Matters

Without fixing a random seed, every run of train_test_split produces a different split.
This means reported scores change on every run, making experiments non-reproducible.

Always set `random_state` to any integer when reporting results. Colleagues must be able
to reproduce your exact numbers.

---

## Comparing Train vs Test Score

After fitting, always compute the score on both sets:

    train_score = r2_score(y_train, model.predict(X_train))
    test_score  = r2_score(y_test,  model.predict(X_test))

| Outcome                    | Interpretation                      |
|----------------------------|-------------------------------------|
| train ~= test, both high   | Good fit — model generalises well   |
| train >> test              | Overfitting — model memorised data  |
| train ~= test, both low    | Underfitting — model too simple     |

This is the first hint of the bias-variance tradeoff (covered on day 5).

---

## Files

- `train_test_split_demo.py` — splitting, fitting, evaluating, and demonstrating leakage
- `exercises.md` — problems
- `solutions/` — reference answers

## Next

Day 3 covers gradient descent — iterative parameter learning without a closed-form solution.
