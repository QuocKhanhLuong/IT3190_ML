# IT3190 — Student Dropout Early Warning System

Course project for **IT3190 — Introduction to Machine Learning and Data Mining**.

## Project
Early prediction of student dropout and academic success using classical machine learning.

### Main experiment settings
1. **Admission-only**: use information available at enrollment; exclude semester 1 and semester 2 academic features.
2. **Semester-1 early warning**: include semester 1 academic features; exclude semester 2 features.

### Models
- DummyClassifier
- Logistic Regression
- Decision Tree
- Random Forest

Random Forest is tuned using training-only stratified cross-validation.

### Evaluation
- Macro F1
- Accuracy
- Balanced Accuracy
- Per-class Precision / Recall / F1
- Dropout Recall
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

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

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

`notebooks/IT3190_student_dropout_full.ipynb`

The notebook contains the complete experiment workflow for local or Kaggle execution.

## Outputs
Training outputs are written to `artifacts/`, including model-comparison tables, test metrics, confusion matrices, permutation importance and the serialized best model.

## Reproducibility
The project uses a fixed random seed (`42`) and a stratified 80/20 train-test split. Hyperparameter selection is performed only on the training set.
