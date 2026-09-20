# Day 1 Notes — What Is Predictive Modeling?

Date: Day 1
Topic: Introduction to Predictive Modeling + Linear Regression
Goal: Understand what a model does and write your first prediction in Python.

---

## 1. What Is Predictive Modeling?

Predictive modeling is teaching a computer to learn from past data so it can make smart guesses about the future.

Think of it like this:
  "I have seen 1,000 houses with their sizes and prices.
   If you show me a new house, I can guess its price."

The computer finds PATTERNS in data. Those patterns become a MODEL.
The model then PREDICTS new things.

---

## 2. Key Vocabulary

| Word             | Plain English                                          |
|------------------|--------------------------------------------------------|
| Feature (X)      | The input(s) we give the model. e.g., house size       |
| Label (y)        | The output we want to predict. e.g., house price       |
| Training         | Showing the model examples so it can learn             |
| Prediction       | The model answer for a new input                       |
| Model            | A mathematical rule the computer learned               |

---

## 3. The Simplest Predictive Model: Linear Regression

Linear Regression draws the best-fit straight line through your data.

Formula:
    y = m * x + b

Where:
  y = what we are predicting (price)
  x = our input (house size in sqft)
  m = slope (how much y changes per unit of x)
  b = intercept (y-value when x = 0)

The computer finds the best m and b automatically!

---

## 4. Real-World Analogy

You notice:
  - A student who studies 1 hour  scores ~50
  - A student who studies 2 hours scores ~60
  - A student who studies 3 hours scores ~70

Linear regression would learn: score = 10 * hours_studied + 40

Now if someone studies 5 hours: predicted score = 10*5 + 40 = 90

---

## 5. The ML Workflow (Every Day)

  1. Collect Data
  2. Clean Data
  3. Train Model
  4. Evaluate
  5. Predict

Today we focus on steps 3 and 5 using a toy dataset.

---

## 6. What To Do Today

1. Read these notes carefully
2. Run code.py and study each printed output
3. Complete exercises.md (try before looking at solutions!)
4. Commit your work:
       git add .
       git commit -m "Day 1: Introduction to predictive modeling"

---

## Key Takeaways

- A model is a mathematical function that maps inputs to outputs
- Linear Regression is the "Hello World" of predictive modeling
- The model learns by minimizing ERROR between predictions and real answers
- Tomorrow: we dig into features, labels, and the train/test split

---

"Start simple. Understand deeply. Build on top."
