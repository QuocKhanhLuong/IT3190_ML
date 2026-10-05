# IT3190 — Student Dropout Early Warning System

Course project for **IT3190 — Introduction to Machine Learning and Data Mining**.

## Project
Early prediction of student dropout and academic success using classical machine learning.

The system predicts one of three labels:

- `Dropout`
- `Enrolled`
- `Graduate`

The project is designed as an **early-warning decision-support system**, not an automatic academic decision-maker.

## Main experiment settings
1. **Admission-only**: use information available at enrollment; exclude semester 1 and semester 2 academic features.
2. **Semester-1 early warning**: include semester 1 academic features; exclude semester 2 features.

The central project question is:

> How much does semester-1 academic information improve early dropout prediction compared with admission-time information only?

## Models
- DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest is the final model after training-only stratified cross-validation and hyperparameter search.

## Evaluation
Main metrics:

- Macro F1
- Dropout Recall

Additional metrics:

- Accuracy
- Balanced Accuracy
- Per-class Precision / Recall / F1
- Confusion Matrix
- Permutation Importance

## Dataset
**Predict Students' Dropout and Academic Success**

- Kaggle: `mattop/predict-students-dropout-and-academic-success`
- UCI Machine Learning Repository: dataset ID 697
- 4,424 samples
- 36 input features
- Target: `Dropout`, `Enrolled`, `Graduate`

On Kaggle, add the dataset as an Input. The training script automatically checks the standard Kaggle input path.

## Repository structure

```text
IT3190_ML/
├── app/
│   ├── app.py
│   └── README.md
├── notebooks/
│   └── IT3190_student_dropout_full.ipynb
├── presentation/
│   ├── SLIDE_OUTLINE.md
│   └── speaker_script.md
├── reports/
│   └── REPORT_OUTLINE.md
├── train_student_dropout.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Train

Quick smoke run:

```bash
python train_student_dropout.py --quick
```

Full run:

```bash
python train_student_dropout.py --cv 5 --n-iter 24 --output-dir artifacts
```

Using a local CSV:

```bash
python train_student_dropout.py --data data/dataset.csv --output-dir artifacts
```

## Notebook
Open:

```text
notebooks/IT3190_student_dropout_full.ipynb
```

The notebook contains the complete experiment workflow for local or Kaggle execution.

## Streamlit demo
Run the training pipeline first so that `artifacts/` contains the trained model and metadata.

Then launch:

```bash
streamlit run app/app.py
```

Demo functions:

1. Single-student prediction.
2. Batch CSV prediction.
3. Model report with generated artifacts.

## Outputs
Training outputs are written to `artifacts/`, including:

- `model_comparison_cv.csv`
- `model_comparison_cv.png`
- `feature_setting_test_results.csv`
- `classification_report_semester1.csv`
- `confusion_matrix_semester1.png`
- `permutation_importance_semester1.csv`
- `permutation_importance_semester1.png`
- `metrics_and_metadata.json`
- `student_dropout_semester1_rf.joblib`

## Report and presentation
- Report outline: `reports/REPORT_OUTLINE.md`
- Slide outline: `presentation/SLIDE_OUTLINE.md`
- Speaking script: `presentation/speaker_script.md`

The PowerPoint deck can be generated/exported separately from the slide outline. The results slide should be filled after running the full experiment on the real dataset.

## Reproducibility
The project uses a fixed random seed (`42`) and a stratified 80/20 train-test split. Hyperparameter selection is performed only on the training set.
