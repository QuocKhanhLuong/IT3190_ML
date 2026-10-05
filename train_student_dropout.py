#!/usr/bin/env python3
"""IT3190 - Early Warning System for Student Dropout.

A complete scikit-learn training pipeline for the UCI/Kaggle dataset
"Predict Students' Dropout and Academic Success".

Main experiments
----------------
1) Admission-time prediction: no semester-1/semester-2 academic features.
2) Early-warning prediction after semester 1: includes semester-1 features,
   excludes semester-2 features.

The script compares several baseline classifiers, tunes Random Forest using
training-only cross-validation, evaluates once on a stratified hold-out test
set, saves plots/tables, and serializes the final semester-1 model.

Kaggle dataset:
    mattop/predict-students-dropout-and-academic-success
Expected Kaggle path (when attached to a notebook):
    /kaggle/input/predict-students-dropout-and-academic-success/dataset.csv

Example
-------
python train_student_dropout.py --data data/dataset.csv --output-dir artifacts
python train_student_dropout.py --quick
"""

from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_validate,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

RANDOM_STATE = 42
TARGET_COL = "Target"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]

# Integer codes in these columns are category IDs, not continuous quantities.
CATEGORICAL_COLUMNS = [
    "Marital status",
    "Application mode",
    "Course",
    "Daytime/evening attendance",
    "Previous qualification",
    "Nacionality",  # spelling in the original dataset
    "Nationality",  # support cleaned mirrors too
    "Mother's qualification",
    "Father's qualification",
    "Mother's occupation",
    "Father's occupation",
    "Displaced",
    "Educational special needs",
    "Debtor",
    "Tuition fees up to date",
    "Gender",
    "Scholarship holder",
    "International",
]

SEM1_PREFIX = "Curricular units 1st sem"
SEM2_PREFIX = "Curricular units 2nd sem"


def set_seed(seed: int = RANDOM_STATE) -> None:
    random.seed(seed)
    np.random.seed(seed)


def _read_csv_robust(path: Path) -> pd.DataFrame:
    """Read comma- or semicolon-delimited mirrors of the dataset."""
    df = pd.read_csv(path)
    if df.shape[1] == 1:
        df = pd.read_csv(path, sep=";")
    return df


def find_local_dataset(explicit_path: str | None = None) -> Path | None:
    """Find the dataset locally, including the standard Kaggle input path."""
    candidates: List[Path] = []
    if explicit_path:
        candidates.append(Path(explicit_path))
    if os.getenv("DATASET_PATH"):
        candidates.append(Path(os.environ["DATASET_PATH"]))

    candidates.extend(
        [
            Path("/kaggle/input/predict-students-dropout-and-academic-success/dataset.csv"),
            Path("/kaggle/input/predict-students-dropout-and-academic-success/data.csv"),
            Path("data/dataset.csv"),
            Path("data/data.csv"),
            Path("dataset.csv"),
            Path("data.csv"),
        ]
    )

    for path in candidates:
        if path.is_file():
            return path

    # Last-resort Kaggle search, restricted to this dataset slug.
    kaggle_dir = Path("/kaggle/input/predict-students-dropout-and-academic-success")
    if kaggle_dir.exists():
        csvs = sorted(kaggle_dir.rglob("*.csv"))
        if csvs:
            return csvs[0]
    return None


def load_dataset(explicit_path: str | None = None) -> Tuple[pd.DataFrame, str]:
    """Load from a local/Kaggle CSV, otherwise fall back to ucimlrepo (UCI id 697)."""
    path = find_local_dataset(explicit_path)
    if path is not None:
        df = _read_csv_robust(path)
        source = str(path)
    else:
        try:
            from ucimlrepo import fetch_ucirepo
        except ImportError as exc:
            raise FileNotFoundError(
                "Dataset not found locally. Attach the Kaggle dataset, pass --data, "
                "or install ucimlrepo with: pip install ucimlrepo"
            ) from exc

        ds = fetch_ucirepo(id=697)
        X = ds.data.features.copy()
        y = ds.data.targets.copy()
        if isinstance(y, pd.Series):
            y = y.to_frame(name=TARGET_COL)
        elif TARGET_COL not in y.columns and y.shape[1] == 1:
            y.columns = [TARGET_COL]
        df = pd.concat([X.reset_index(drop=True), y.reset_index(drop=True)], axis=1)
        source = "UCI ML Repository via ucimlrepo (dataset id=697)"

    df.columns = [str(c).strip() for c in df.columns]
    unnamed = [c for c in df.columns if c.lower().startswith("unnamed")]
    if unnamed:
        df = df.drop(columns=unnamed)

    if TARGET_COL not in df.columns:
        matches = [c for c in df.columns if c.strip().lower() == "target"]
        if matches:
            df = df.rename(columns={matches[0]: TARGET_COL})
        else:
            raise ValueError(f"Target column not found. Columns: {list(df.columns)}")

    # Normalize target labels across mirrors.
    target_map = {
        "dropout": "Dropout",
        "enrolled": "Enrolled",
        "graduate": "Graduate",
    }
    normalized = df[TARGET_COL].astype(str).str.strip().str.lower().map(target_map)
    if normalized.isna().any():
        bad = sorted(df.loc[normalized.isna(), TARGET_COL].astype(str).unique().tolist())
        raise ValueError(f"Unexpected target labels: {bad}")
    df[TARGET_COL] = normalized

    return df, source


def get_feature_sets(df: pd.DataFrame) -> Dict[str, List[str]]:
    all_features = [c for c in df.columns if c != TARGET_COL]
    admission = [
        c for c in all_features
        if not c.startswith(SEM1_PREFIX) and not c.startswith(SEM2_PREFIX)
    ]
    semester1 = [c for c in all_features if not c.startswith(SEM2_PREFIX)]
    return {
        "admission": admission,
        "semester1": semester1,
    }


def make_preprocessor(feature_columns: Iterable[str]) -> ColumnTransformer:
    feature_columns = list(feature_columns)
    categorical = [c for c in CATEGORICAL_COLUMNS if c in feature_columns]
    numerical = [c for c in feature_columns if c not in categorical]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numerical),
            ("cat", categorical_pipe, categorical),
        ],
        remainder="drop",
    )


def make_pipeline(feature_columns: Iterable[str], estimator) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(feature_columns)),
            ("model", estimator),
        ]
    )


def baseline_models() -> Dict[str, object]:
    return {
        "Dummy": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": LogisticRegression(
            max_iter=3000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "DecisionTree": DecisionTreeClassifier(
            max_depth=10,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            max_features="sqrt",
            class_weight="balanced_subsample",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }


def compare_models_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: List[str],
    feature_setting: str,
    cv_splits: int = 5,
) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_f1": "f1_macro",
    }
    rows = []
    for name, estimator in baseline_models().items():
        pipeline = make_pipeline(feature_columns, estimator)
        scores = cross_validate(
            pipeline,
            X_train[feature_columns],
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False,
        )
        rows.append(
            {
                "feature_setting": feature_setting,
                "model": name,
                "cv_accuracy_mean": scores["test_accuracy"].mean(),
                "cv_accuracy_std": scores["test_accuracy"].std(),
                "cv_balanced_accuracy_mean": scores["test_balanced_accuracy"].mean(),
                "cv_balanced_accuracy_std": scores["test_balanced_accuracy"].std(),
                "cv_macro_f1_mean": scores["test_macro_f1"].mean(),
                "cv_macro_f1_std": scores["test_macro_f1"].std(),
            }
        )
    return pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False)


def tune_random_forest(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    feature_columns: List[str],
    n_iter: int = 24,
    cv_splits: int = 5,
) -> RandomizedSearchCV:
    estimator = RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1)
    pipeline = make_pipeline(feature_columns, estimator)

    param_distributions = {
        "model__n_estimators": [200, 300, 500, 700],
        "model__max_depth": [None, 8, 12, 16, 24],
        "model__min_samples_split": [2, 5, 10],
        "model__min_samples_leaf": [1, 2, 4],
        "model__max_features": ["sqrt", "log2", 0.5],
        "model__class_weight": [None, "balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        scoring="f1_macro",
        refit=True,
        cv=cv,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
        return_train_score=False,
    )
    search.fit(X_train[feature_columns], y_train)
    return search


def evaluate_model(model: Pipeline, X: pd.DataFrame, y: pd.Series) -> Tuple[dict, pd.DataFrame, np.ndarray]:
    pred = model.predict(X)
    metrics = {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "macro_precision": float(precision_score(y, pred, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y, pred, average="macro", zero_division=0)),
        "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
    }
    for label in CLASS_ORDER:
        metrics[f"recall_{label.lower()}"] = float(
            recall_score(y == label, pred == label, zero_division=0)
        )
        metrics[f"f1_{label.lower()}"] = float(
            f1_score(y == label, pred == label, zero_division=0)
        )

    report = pd.DataFrame(
        classification_report(
            y,
            pred,
            labels=CLASS_ORDER,
            output_dict=True,
            zero_division=0,
        )
    ).T
    cm = confusion_matrix(y, pred, labels=CLASS_ORDER)
    return metrics, report, cm


def save_confusion_matrix(cm: np.ndarray, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm)
    ax.set_xticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER, rotation=25, ha="right")
    ax.set_yticks(range(len(CLASS_ORDER)), labels=CLASS_ORDER)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("Confusion Matrix - Semester 1 Random Forest")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_cv_plot(cv_results: pd.DataFrame, path: Path) -> None:
    plot_df = cv_results.copy()
    labels = plot_df["feature_setting"] + " / " + plot_df["model"]
    values = plot_df["cv_macro_f1_mean"].to_numpy()
    errors = plot_df["cv_macro_f1_std"].to_numpy()
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(labels))
    ax.barh(y_pos, values, xerr=errors)
    ax.set_yticks(y_pos, labels=labels)
    ax.invert_yaxis()
    ax.set_xlabel("5-fold CV Macro-F1")
    ax.set_title("Model comparison")
    ax.set_xlim(0, 1)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def permutation_feature_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_columns: List[str],
    n_repeats: int = 10,
) -> pd.DataFrame:
    result = permutation_importance(
        model,
        X_test[feature_columns],
        y_test,
        scoring="f1_macro",
        n_repeats=n_repeats,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    return (
        pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": result.importances_mean,
                "importance_std": result.importances_std,
            }
        )
        .sort_values("importance_mean", ascending=False)
        .reset_index(drop=True)
    )


def save_feature_importance_plot(importance_df: pd.DataFrame, path: Path, top_k: int = 20) -> None:
    top = importance_df.head(top_k).sort_values("importance_mean", ascending=True)
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(top["feature"], top["importance_mean"], xerr=top["importance_std"])
    ax.set_xlabel("Decrease in Macro-F1 after permutation")
    ax.set_title(f"Top {top_k} permutation importances")
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def json_safe_params(params: dict) -> dict:
    out = {}
    for key, value in params.items():
        if isinstance(value, (np.integer, np.floating)):
            value = value.item()
        out[key] = value
    return out


def run(args: argparse.Namespace) -> None:
    set_seed()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df, source = load_dataset(args.data)
    print(f"Loaded dataset from: {source}")
    print(f"Shape: {df.shape}")
    print("Target distribution:")
    print(df[TARGET_COL].value_counts())

    feature_sets = get_feature_sets(df)
    X_all = df.drop(columns=[TARGET_COL])
    y_all = df[TARGET_COL]

    train_idx, test_idx = train_test_split(
        np.arange(len(df)),
        test_size=args.test_size,
        stratify=y_all,
        random_state=RANDOM_STATE,
    )
    X_train, X_test = X_all.iloc[train_idx].copy(), X_all.iloc[test_idx].copy()
    y_train, y_test = y_all.iloc[train_idx].copy(), y_all.iloc[test_idx].copy()

    # 1) Baseline model comparison for both early-warning feature settings.
    all_cv = []
    for setting, features in feature_sets.items():
        print(f"\n=== CV comparison: {setting} ({len(features)} raw features) ===")
        cv_df = compare_models_cv(
            X_train,
            y_train,
            features,
            feature_setting=setting,
            cv_splits=args.cv,
        )
        print(cv_df.to_string(index=False))
        all_cv.append(cv_df)
    cv_results = pd.concat(all_cv, ignore_index=True)
    cv_results.to_csv(output_dir / "model_comparison_cv.csv", index=False)
    save_cv_plot(cv_results, output_dir / "model_comparison_cv.png")

    # 2) Tune Random Forest separately on training data for each feature setting.
    # This lets the admission-vs-semester1 ablation remain fair without touching the test set.
    test_rows = []
    searches: Dict[str, RandomizedSearchCV] = {}
    n_iter = min(args.n_iter, 8) if args.quick else args.n_iter

    for setting, features in feature_sets.items():
        print(f"\n=== Random Forest tuning: {setting} ===")
        search = tune_random_forest(
            X_train,
            y_train,
            features,
            n_iter=n_iter,
            cv_splits=args.cv,
        )
        searches[setting] = search
        print(f"Best CV Macro-F1: {search.best_score_:.4f}")
        print(f"Best params: {search.best_params_}")

        search_df = pd.DataFrame(search.cv_results_).sort_values("rank_test_score")
        keep_cols = [
            "rank_test_score",
            "mean_test_score",
            "std_test_score",
            "mean_fit_time",
            "params",
        ]
        search_df[keep_cols].to_csv(
            output_dir / f"random_forest_search_{setting}.csv", index=False
        )

        metrics, _, _ = evaluate_model(
            search.best_estimator_, X_test[features], y_test
        )
        test_rows.append(
            {
                "feature_setting": setting,
                "best_cv_macro_f1": float(search.best_score_),
                **metrics,
            }
        )

    test_results = pd.DataFrame(test_rows)
    test_results.to_csv(output_dir / "feature_setting_test_results.csv", index=False)
    print("\n=== Hold-out feature-setting comparison ===")
    print(test_results.to_string(index=False))

    # 3) Final model: semester-1 early-warning setting, defined a priori by project scope.
    final_features = feature_sets["semester1"]
    final_search = searches["semester1"]
    final_model: Pipeline = final_search.best_estimator_
    final_metrics, report, cm = evaluate_model(
        final_model, X_test[final_features], y_test
    )

    report.to_csv(output_dir / "classification_report_semester1.csv")
    save_confusion_matrix(cm, output_dir / "confusion_matrix_semester1.png")

    print("\n=== Final semester-1 model metrics ===")
    for key, value in final_metrics.items():
        print(f"{key:>24s}: {value:.4f}")

    # 4) Permutation importance on original raw features (interpretable feature-level analysis).
    repeats = 4 if args.quick else args.permutation_repeats
    importance_df = permutation_feature_importance(
        final_model,
        X_test,
        y_test,
        final_features,
        n_repeats=repeats,
    )
    importance_df.to_csv(output_dir / "permutation_importance_semester1.csv", index=False)
    save_feature_importance_plot(
        importance_df,
        output_dir / "permutation_importance_semester1.png",
        top_k=min(20, len(importance_df)),
    )

    # 5) Persist model and reproducibility metadata.
    joblib.dump(final_model, output_dir / "student_dropout_semester1_rf.joblib")
    metadata = {
        "dataset_source": source,
        "dataset_shape": list(df.shape),
        "target_distribution": df[TARGET_COL].value_counts().to_dict(),
        "random_state": RANDOM_STATE,
        "test_size": args.test_size,
        "cv_folds": args.cv,
        "feature_sets": {k: v for k, v in feature_sets.items()},
        "final_feature_setting": "semester1",
        "final_features": final_features,
        "best_cv_macro_f1": float(final_search.best_score_),
        "best_params": json_safe_params(final_search.best_params_),
        "test_metrics": final_metrics,
    }
    with open(output_dir / "metrics_and_metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    print(f"\nSaved all artifacts to: {output_dir.resolve()}")
    print("Final model:", output_dir / "student_dropout_semester1_rf.joblib")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to dataset.csv/data.csv. If omitted, auto-detect Kaggle/local paths, then UCI.",
    )
    parser.add_argument("--output-dir", type=str, default="artifacts")
    parser.add_argument("--test-size", type=float, default=0.20)
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument(
        "--n-iter",
        type=int,
        default=24,
        help="RandomizedSearchCV candidates per feature setting.",
    )
    parser.add_argument("--permutation-repeats", type=int, default=10)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Fast smoke-test mode: fewer search candidates and permutation repeats.",
    )
    return parser


if __name__ == "__main__":
    run(build_arg_parser().parse_args())