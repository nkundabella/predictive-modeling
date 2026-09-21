# Day 2 Exercises — Features, Labels, Train/Test Split, Data Leakage

Work through these in order. Attempt every question before opening `solutions/`.

---

## Theory

**1.** What is the difference between a feature and a label?
Give an example pair from a domain other than house prices or exam scores.

**2.** Why is it invalid to evaluate a model on the same data it was trained on?
Construct a concrete (even silly) example where a model scores 100% on training
data but would obviously fail on new data.

**3.** You have 200 observations. A colleague suggests using 190 for training and
10 for testing to "give the model as much data as possible."
What is the problem with a test set of only 10 rows?

**4.** A test set should only be looked at once. Why?
What goes wrong if you repeatedly evaluate different model versions on the same
test set and pick the best one?

**5.** Explain train-test contamination (leakage type 2) in plain English.
Why does fitting a `StandardScaler` on the full dataset before splitting count as leakage?

---

## Code

**6.** Copy the house-price dataset from `train_test_split_demo.py` (Part 1).
Fit a `LinearRegression` model three times, each time using a different
`test_size`: 0.1, 0.2, and 0.4. Record the test R² for each split.
What trend do you observe as the test set grows? Why?

**7.** Take the same dataset and do the following *deliberately wrong* thing:
   - Compute the mean of each feature column across the entire dataset.
   - Subtract the mean from every column (mean-centering).
   - Then split into train/test and fit a LinearRegression.
   - Compare the test R² to the correct workflow (split first, then centre).

Does the score differ? By how much? Write one paragraph explaining what
information leaked and why it matters more with larger preprocessing effects.

**8.** Introduce a leaky feature:
   - To the house-price dataset, add a column: `price_plus_noise = price + N(0, 1)`.
   - Fit a LinearRegression with this extra column included.
   - Report the train R² and test R².
   - Now remove the leaky column and refit. Report both scores again.
   - What happened, and why?

**9.** Run 50 different random splits (random_state 0–49) of the house-price data,
recording test R² each time. Plot a histogram of those 50 scores.
What is the standard deviation? What does this tell you about the reliability
of a single train/test split?

**10.** The `train_test_split` function has a `shuffle=True` default.
Imagine the house-price rows were sorted by price (cheapest first).
What would happen if you set `shuffle=False` and used the last 20% as the test set?
Would the test score be higher or lower than with shuffling? Why?
(You don't have to run this — reason through it first, then verify.)

---

## Reflection

Write a short paragraph (5–8 sentences):
At what point in your mental model did the concept of "leakage" feel unintuitive?
Is there a real-world analogy outside of machine learning where the same problem occurs?
What is the single most important rule you would tell a junior colleague about the
order of operations when preparing a dataset?
