# 12-Minute Presentation Script

## Slide 1 — Title
Hello everyone. Our project is an early-warning system for student dropout and academic success using machine learning. The goal is to predict whether a student will drop out, remain enrolled, or graduate.

## Slide 2 — Practical Problem
Student dropout is important because if we detect risk too late, support actions may not help. Our system is designed as a support tool for early intervention, not as an automatic decision-maker.

## Slide 3 — Dataset
We use the UCI/Kaggle student dropout dataset. It contains 4,424 student records, 36 features, and three labels: Dropout, Enrolled, and Graduate.

## Slide 4 — Class Imbalance
The dataset is imbalanced. Graduate is the largest class, then Dropout, then Enrolled. Because of this, we do not rely only on accuracy. We focus on Macro-F1 and Dropout Recall.

## Slide 5 — Early-Warning Design
We compare two settings. The first setting uses only admission-time information. The second setting adds semester-1 academic information. We exclude semester-2 information because it is too late for early warning.

## Slide 6 — Preprocessing
Some columns are numerical, while many columns are categorical codes. The categorical codes are not real numbers, so we use one-hot encoding. Numerical features are imputed and standardized. Everything is inside a scikit-learn pipeline.

## Slide 7 — Models
We compare four models: DummyClassifier, Logistic Regression, Decision Tree, and Random Forest. Random Forest is the main model because it can learn nonlinear interactions while still being easy to run on CPU.

## Slide 8 — Experimental Protocol
We use a stratified 80/20 train-test split. Model selection is done only on the training set using stratified cross-validation. The test set is used once at the end.

## Slide 9 — Metrics
Macro-F1 treats all classes equally. Dropout Recall measures how many actual dropout students the model can detect. This is important because missing a high-risk student is the most serious mistake.

## Slide 10 — Demo System
The Streamlit demo has three functions: predict one student, predict from a batch CSV, and view model reports. This turns the project from just a notebook into a usable system.

## Slide 11 — Results
After running the full experiment, we will show model comparison, admission-only versus semester-1 performance, the confusion matrix, and feature importance.

## Slide 12 — Conclusion
In conclusion, this project provides a complete machine learning workflow: dataset, preprocessing, model comparison, evaluation, interpretation, and demo. The main question is whether semester-1 information improves early dropout prediction.
