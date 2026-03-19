# HealthPredict AI

HealthPredict AI is a full machine learning project for chronic disease risk prediction based on the topic "Health Monitoring and Disease Prediction Using Machine Learning." It includes:

- Dual disease prediction workflows for heart disease and diabetes
- Data ingestion and reproducible demo dataset generation
- Preprocessing for missing values, encoding, scaling, and SMOTE balancing
- Explicit Logistic Regression vs Random Forest model comparison
- SHAP explainability and ROC curve visualization
- A Streamlit app for interactive risk scoring
- Simple local user registration and login for the app

## Project Structure

```text
Health-Prediction/
  app.py
  README.md
  requirements.txt
  .gitignore
  artifacts/
    heart_disease_bundle.joblib
    diabetes_bundle.joblib
    model.pkl
  data/
    raw/
      heart.csv
      diabetes.csv
    users.json
  reports/
    metrics_summary.json
    model_comparison.json
    plots/
      heart_disease_roc_curve.png
      diabetes_roc_curve.png
    shap/
      shap_summary.png
      shap_waterfall_row_0.png
  tests/
  health_predict_ai/
```

Main source package:

```text
health_predict_ai/
  __init__.py
  config.py
  data.py
  preprocessing.py
  features.py
  modeling.py
  pipeline.py
  predict.py
  explain.py
  plots.py
  train.py
  run_explain.py
  run_plots.py
```

## Core Features

- Heart disease and diabetes prediction from structured health inputs
- Training pipeline with preprocessing, model comparison, and artifact export
- Saved `.joblib` bundles and `.pkl` model files for presentation and reuse
- SHAP global and local explainability plots
- ROC curve plots and JSON evaluation summaries
- Streamlit dashboard with login, registration, and interactive prediction forms

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

4. Generate SHAP explainability outputs:

```bash
python3 -m health_predict_ai.run_explain
```

5. Generate ROC curve plots directly:

```bash
python3 -m health_predict_ai.run_plots
```

6. Launch the app:

```bash
streamlit run app.py
```

## Datasets

This repository is structured to work with CSV datasets stored in:

- `data/raw/heart.csv`
- `data/raw/diabetes.csv`

If those files are missing, the project can generate reproducible demo datasets automatically.

Expected target columns:

- Heart disease: `target`
- Diabetes: `diabetes`

App user credentials are stored locally in:

- `data/users.json`

## Training Workflow

The training pipeline now makes the full implementation visible in code:

1. Split the dataset into training and test sets
2. Detect numeric and categorical features
3. Impute missing values
4. Encode categorical variables
5. Scale numeric features
6. Apply SMOTE to the training data
7. Compare Logistic Regression and Random Forest
8. Save the best trained pipeline and evaluation reports
9. Generate ROC comparison plots

## Outputs And Reports

After training and report generation, the repository contains:

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

Run all tests with:

```bash
python3 -m unittest tests.test_preprocessing tests.test_modeling tests.test_predict
```

## Notes

- `streamlit` is required to run the dashboard.
- `shap` and `matplotlib` are required to generate explainability and evaluation plots.
- The synthetic datasets are intended for academic demonstration and software validation, not clinical use.
