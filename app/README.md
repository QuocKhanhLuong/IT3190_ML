# Streamlit Demo

Run the training pipeline first:

```bash
python train_student_dropout.py --cv 5 --n-iter 24 --output-dir artifacts
```

Then launch the demo:

```bash
streamlit run app/app.py
```

The app expects:

```text
artifacts/student_dropout_semester1_rf.joblib
artifacts/metrics_and_metadata.json
```

Main demo functions:

1. **Single student prediction** — manually enter raw feature values and see class probabilities.
2. **Batch CSV prediction** — upload many students and download predictions.
3. **Model report** — view metrics and generated plots from training artifacts.

The form keeps the original dataset coding scheme because the fitted preprocessing pipeline expects raw UCI/Kaggle feature values.
