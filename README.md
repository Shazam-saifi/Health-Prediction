# HealthPredict AI

HealthPredict AI is a full machine learning project for chronic disease risk prediction based on the topic "Health Monitoring and Disease Prediction Using Machine Learning." It includes:

- Dual disease prediction workflows for heart disease and diabetes
- Data generation and ingestion utilities
- Preprocessing for missing values, encoding, scaling, and SMOTE balancing
- Explicit Logistic Regression vs Random Forest model comparison
- Explainability-ready prediction output
- A Streamlit app for interactive risk scoring

## Project Structure

```text
Health-Prediction/
  artifacts/
  data/
    raw/
  reports/
    plots/
    shap/
  tests/
  health_predict_ai/
```

Main source package:

```text
health_predict_ai/
  config.py
  data.py
  preprocessing.py
  explain.py
  features.py
  modeling.py
  pipeline.py
  plots.py
  predict.py
  run_explain.py
  run_plots.py
  train.py
app.py
tests/
```

## Quick Start

1. Create and activate a virtual environment.

Create

```bash
python3 -m venv venv
```

Activate

```bash
source venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Train models and generate reports:

```bash
python3 -m health_predict_ai.train
```

4. Launch the app:

```bash
streamlit run app.py
```

## Datasets

This repository ships with a reproducible synthetic data generator so the project runs without external downloads. The code is also structured to accept real CSV datasets later in:

- `data/raw/heart.csv`
- `data/raw/diabetes.csv`

Expected target columns:

- Heart disease: `target`
- Diabetes: `diabetes`

## Outputs

Training produces:

- `artifacts/heart_disease_bundle.joblib`
- `artifacts/diabetes_bundle.joblib`
- `artifacts/heart_disease_model.pkl`
- `artifacts/diabetes_model.pkl`
- `artifacts/model.pkl`
- `reports/model_comparison.json`
- `reports/metrics_summary.json`
- `reports/plots/heart_disease_roc_curve.png`
- `reports/plots/diabetes_roc_curve.png`
- `reports/shap/shap_summary.png`
- `reports/shap/shap_waterfall_row_0.png`

## Tests

The test suite is organized by responsibility:

- `tests/test_preprocessing.py`
- `tests/test_modeling.py`
- `tests/test_predict.py`

## Notes

- `streamlit` and `shap` are optional at development time but required for the full UI and explainability experience.
- The synthetic datasets are intended for academic demonstration and software validation, not clinical use.
