"""Streamlit demo for the IT3190 student dropout early-warning project.

Run:
    python train_student_dropout.py --output-dir artifacts
    streamlit run app/app.py
"""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "student_dropout_semester1_rf.joblib"
METADATA_PATH = ARTIFACT_DIR / "metrics_and_metadata.json"
CLASS_ORDER = ["Dropout", "Enrolled", "Graduate"]


def load_uploaded_model(uploaded_file) -> Any:
    suffix = Path(uploaded_file.name).suffix or ".joblib"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return joblib.load(tmp.name)


@st.cache_resource(show_spinner=False)
def load_local_model(path: str) -> Any:
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def load_local_metadata(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_preprocessor_info(model: Any) -> Tuple[List[str], List[str], Dict[str, List[Any]], Dict[str, float]]:
    """Read raw columns, category values and numerical medians from the fitted pipeline."""
    preprocess = model.named_steps["preprocess"]
    num_cols: List[str] = []
    cat_cols: List[str] = []
    cat_options: Dict[str, List[Any]] = {}
    numeric_defaults: Dict[str, float] = {}

    for name, transformer, columns in preprocess.transformers_:
        cols = list(columns)
        if name == "num":
            num_cols = cols
            imputer = transformer.named_steps.get("imputer")
            if imputer is not None:
                numeric_defaults = {c: float(v) for c, v in zip(cols, imputer.statistics_)}
        elif name == "cat":
            cat_cols = cols
            onehot = transformer.named_steps.get("onehot")
            if onehot is not None:
                for c, values in zip(cols, onehot.categories_):
                    cat_options[c] = [v.item() if hasattr(v, "item") else v for v in values]
    return num_cols, cat_cols, cat_options, numeric_defaults


def get_features(metadata: Dict[str, Any], model: Any) -> List[str]:
    if metadata.get("final_features"):
        return list(metadata["final_features"])
    num_cols, cat_cols, _, _ = get_preprocessor_info(model)
    return num_cols + cat_cols


def read_artifacts(model_file, metadata_file):
    model = None
    metadata: Dict[str, Any] = {}

    if model_file is not None:
        model = load_uploaded_model(model_file)
    elif MODEL_PATH.exists():
        model = load_local_model(str(MODEL_PATH))

    if metadata_file is not None:
        metadata = json.loads(metadata_file.getvalue().decode("utf-8"))
    elif METADATA_PATH.exists():
        metadata = load_local_metadata(str(METADATA_PATH))

    return model, metadata


def predict_df(model: Any, df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    missing = [c for c in features if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    X = df[features].copy()
    out = df.copy()
    out["prediction"] = model.predict(X)

    if hasattr(model, "predict_proba"):
        class_names = list(getattr(model, "classes_", CLASS_ORDER))
        proba = model.predict_proba(X)
        for i, name in enumerate(class_names):
            out[f"prob_{name}"] = proba[:, i]
    return out


def render_single_form(features: List[str], cat_options: Dict[str, List[Any]], numeric_defaults: Dict[str, float]) -> pd.DataFrame:
    row: Dict[str, Any] = {}
    groups = {
        "Admission/background": [c for c in features if not c.startswith("Curricular units") and c not in ["Unemployment rate", "Inflation rate", "GDP"]],
        "Semester 1 academic signals": [c for c in features if c.startswith("Curricular units 1st sem")],
        "Economic context": [c for c in features if c in ["Unemployment rate", "Inflation rate", "GDP"]],
    }

    for group_name, cols in groups.items():
        if not cols:
            continue
        with st.expander(group_name, expanded=group_name != "Economic context"):
            left, right = st.columns(2)
            for i, col in enumerate(cols):
                host = left if i % 2 == 0 else right
                with host:
                    if col in cat_options:
                        row[col] = st.selectbox(col, cat_options[col])
                    else:
                        row[col] = st.number_input(col, value=float(numeric_defaults.get(col, 0.0)), step=1.0, format="%.4f")
    return pd.DataFrame([row], columns=features)


def main() -> None:
    st.set_page_config(page_title="Student Dropout Early Warning", page_icon="🎓", layout="wide")
    st.title("Student Dropout Early Warning System")
    st.caption("IT3190 demo: predict Dropout / Enrolled / Graduate using the trained scikit-learn pipeline.")

    with st.sidebar:
        st.header("Model artifacts")
        st.write("Default local paths:")
        st.code("artifacts/student_dropout_semester1_rf.joblib\nartifacts/metrics_and_metadata.json")
        uploaded_model = st.file_uploader("Optional model upload", type=["joblib", "pkl"])
        uploaded_meta = st.file_uploader("Optional metadata upload", type=["json"])

    try:
        model, metadata = read_artifacts(uploaded_model, uploaded_meta)
    except Exception as exc:
        st.error(f"Could not load artifacts: {exc}")
        st.stop()

    if model is None:
        st.warning("No trained model found yet. Train first, then reload this app.")
        st.code("python train_student_dropout.py --cv 5 --n-iter 24 --output-dir artifacts\nstreamlit run app/app.py", language="bash")
        st.stop()

    features = get_features(metadata, model)
    _, _, cat_options, numeric_defaults = get_preprocessor_info(model)

    tab_one, tab_batch, tab_report = st.tabs(["Single student", "Batch CSV", "Model report"])

    with tab_one:
        st.subheader("Single-student prediction")
        st.info("Use raw dataset codes. For a clean demo, copy one real row from the UCI/Kaggle CSV.")
        single_df = render_single_form(features, cat_options, numeric_defaults)
        if st.button("Predict", type="primary"):
            result = predict_df(model, single_df, features)
            st.success(f"Prediction: {result.loc[0, 'prediction']}")
            proba_cols = [c for c in result.columns if c.startswith("prob_")]
            if proba_cols:
                chart = result[proba_cols].rename(columns=lambda c: c.replace("prob_", "")).T
                chart.columns = ["probability"]
                st.bar_chart(chart)
                st.dataframe(chart.style.format({"probability": "{:.2%}"}))

    with tab_batch:
        st.subheader("Batch prediction")
        template = pd.DataFrame([{c: numeric_defaults.get(c, 0) for c in features}])
        for c, opts in cat_options.items():
            if c in template.columns and opts:
                template.loc[0, c] = opts[0]
        st.download_button("Download CSV template", template.to_csv(index=False), "student_dropout_input_template.csv")

        csv_file = st.file_uploader("Upload CSV", type=["csv"], key="batch_csv")
        if csv_file is not None:
            try:
                batch_df = pd.read_csv(csv_file)
                pred_df = predict_df(model, batch_df, features)
                st.dataframe(pred_df.head(100))
                st.download_button("Download predictions", pred_df.to_csv(index=False), "student_dropout_predictions.csv")
            except Exception as exc:
                st.error(str(exc))

    with tab_report:
        st.subheader("Model report")
        if metadata:
            st.write("Dataset source:", metadata.get("dataset_source", "unknown"))
            st.write("Dataset shape:", metadata.get("dataset_shape", "unknown"))
            st.write("Final feature setting:", metadata.get("final_feature_setting", "unknown"))
            if "test_metrics" in metadata:
                metrics = metadata["test_metrics"]
                cols = st.columns(3)
                for i, key in enumerate(["accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "dropout_recall"]):
                    if key in metrics:
                        cols[i % 3].metric(key.replace("_", " ").title(), f"{float(metrics[key]):.3f}")
            with st.expander("Metadata JSON"):
                st.json(metadata)
        else:
            st.info("Metadata JSON not available.")

        for image_name in ["model_comparison_cv.png", "confusion_matrix_semester1.png", "permutation_importance_semester1.png"]:
            path = ARTIFACT_DIR / image_name
            if path.exists():
                st.image(str(path), caption=image_name, use_container_width=True)


if __name__ == "__main__":
    main()
