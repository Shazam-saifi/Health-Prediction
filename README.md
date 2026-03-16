# HealthPredict AI

HealthPredict AI is a full machine learning project for chronic disease risk prediction based on the topic "Health Monitoring and Disease Prediction Using Machine Learning." It includes:

- Dual disease prediction workflows for heart disease and diabetes
- Data generation and ingestion utilities
- Preprocessing, imbalance handling, model training, and evaluation
- Explainability-ready prediction output
- A Streamlit app for interactive risk scoring

## Project Structure

```text
health_predict_ai/
  config.py
  data.py
  explain.py
  features.py
  modeling.py
  pipeline.py
  predict.py
  train.py
app.py
tests/
```

## Quick Start

1. Create and activate a virtual environment.
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

- `artifacts/heart_bundle.joblib`
- `artifacts/diabetes_bundle.joblib`
- `reports/metrics_summary.json`

## Notes

- `streamlit` and `shap` are optional at development time but required for the full UI and explainability experience.
- The synthetic datasets are intended for academic demonstration and software validation, not clinical use.
