# IT3190 Project Report Outline

## Title
Early Prediction of Student Dropout and Academic Success Using Machine Learning

## 1. Introduction
- Student dropout is a practical problem in higher education.
- The goal is to build an early-warning machine learning system that predicts whether a student is likely to be `Dropout`, `Enrolled`, or `Graduate`.
- The system is intended to support early intervention, not to make automatic academic decisions.

## 2. Problem Formulation
- **Task type:** multi-class classification.
- **Input:** admission/background/socioeconomic features and, in the early-warning setting, semester-1 academic features.
- **Output:** predicted class and class probabilities for `Dropout`, `Enrolled`, and `Graduate`.
- **Main research question:** How much does semester-1 information improve early prediction compared with admission-time information only?

## 3. Dataset
- Dataset: `Predict Students' Dropout and Academic Success`.
- Source: UCI Machine Learning Repository / Kaggle mirror.
- Size: 4,424 student records.
- Features: 36 input attributes.
- Target classes: `Dropout`, `Enrolled`, `Graduate`.
- The class distribution is imbalanced, so accuracy alone is not sufficient.

## 4. Feature Settings
### 4.1 Admission-only setting
Uses information available at or near enrollment. Excludes semester-1 and semester-2 academic variables.

### 4.2 Semester-1 early-warning setting
Uses admission/background features plus semester-1 academic variables. Excludes semester-2 variables because they are too late for an early-warning system.

## 5. Preprocessing
- Categorical code columns are one-hot encoded.
- Numerical columns are median-imputed and standardized.
- Preprocessing is fitted only on the training data inside a scikit-learn pipeline.
- The same fitted pipeline is used for validation, test and Streamlit inference.

## 6. Methods
Models compared:
- DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest

The final model is Random Forest because it can model nonlinear interactions between academic, financial and demographic factors while remaining CPU-friendly.

## 7. Experimental Protocol
- Stratified 80/20 train-test split.
- 5-fold stratified cross-validation on the training set.
- RandomizedSearchCV for Random Forest hyperparameter tuning.
- The hold-out test set is used once for final evaluation.

## 8. Evaluation Metrics
Main metrics:
- Macro-F1
- Dropout Recall

Additional metrics:
- Accuracy
- Balanced Accuracy
- Macro Precision
- Macro Recall
- Confusion Matrix
- Per-class Precision/Recall/F1

Macro-F1 is selected because all classes should matter, especially the smaller `Enrolled` class. Dropout Recall is important because missing at-risk students is costly.

## 9. Results
Fill this section after running the notebook/script on the real dataset.

### 9.1 Model comparison
Insert `artifacts/model_comparison_cv.csv` and `artifacts/model_comparison_cv.png`.

### 9.2 Admission-only vs Semester-1 setting
Insert `artifacts/feature_setting_test_results.csv`.

### 9.3 Final model result
Insert:
- `artifacts/classification_report_semester1.csv`
- `artifacts/confusion_matrix_semester1.png`

### 9.4 Feature importance
Insert:
- `artifacts/permutation_importance_semester1.csv`
- `artifacts/permutation_importance_semester1.png`

## 10. System Demo
The Streamlit app supports:
- Single-student prediction.
- Batch CSV prediction.
- Model report view.

Run:

```bash
streamlit run app/app.py
```

## 11. Difficulties and Solutions
Possible points to discuss:
- Categorical columns are integer-coded but not ordinal, so one-hot encoding is required.
- The target distribution is imbalanced, so Macro-F1 and Dropout Recall are emphasized.
- To avoid data leakage, semester-2 variables are excluded from the final early-warning setting.
- Test data is not used during model selection.

## 12. Conclusion
- The project builds a complete classical ML system for student dropout early warning.
- The key comparison is admission-only vs semester-1 prediction.
- The final deliverable includes code, notebook, trained model artifacts, report and Streamlit demo.
