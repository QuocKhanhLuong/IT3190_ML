# Slide Outline

This outline matches the 12-minute presentation deck.

## Slide 1 — Title
Student Dropout Early Warning System.

## Slide 2 — Practical Problem
Student support only works when risk is detected early.

## Slide 3 — Dataset
UCI/Kaggle student dropout dataset: 4,424 records, 36 features, 3 target classes.

## Slide 4 — Target Distribution
Show class imbalance: Graduate 2,209; Dropout 1,421; Enrolled 794.

## Slide 5 — Early-Warning Design
Compare admission-only features vs admission + semester-1 features. Exclude semester-2 variables to avoid late leakage.

## Slide 6 — Preprocessing Pipeline
Numerical features: median imputation + standardization. Categorical coded features: one-hot encoding.

## Slide 7 — Model Comparison
DummyClassifier, Logistic Regression, Decision Tree, Random Forest.

## Slide 8 — Experimental Protocol
Stratified 80/20 split; 5-fold CV on training; Random Forest tuning; final test once.

## Slide 9 — Evaluation Metrics
Macro-F1 and Dropout Recall are the main metrics. Also report accuracy, balanced accuracy, confusion matrix and feature importance.

## Slide 10 — Demo System
Streamlit app: single-student prediction, batch CSV prediction, model-report view.

## Slide 11 — Results After Training
Fill with artifacts after running the real dataset: model comparison, feature-setting comparison, confusion matrix and permutation importance.

## Slide 12 — Conclusion
The project delivers a complete ML workflow: dataset, preprocessing, model selection, evaluation, interpretation and demo.
