# HealthPredict AI

HealthPredict AI is a machine learning project for **health monitoring and chronic disease risk prediction**. The system is designed to predict the likelihood of diseases such as **heart disease** and **diabetes** using structured health and lifestyle data. The project combines data preprocessing, feature engineering, model training, evaluation, and a Streamlit interface for interactive prediction. This aligns with the scope described in the Phase-2 report, which focuses on structured numerical and categorical health features rather than imaging or wearable data.  [oai_citation:2‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

## Project Objective

The main objective of this project is to apply machine learning techniques to a real-world healthcare problem by building a decision-support system that can:

- predict disease risk from structured patient data
- compare multiple machine learning models
- handle class imbalance in medical datasets
- provide interpretable and reproducible results
- support real-time prediction through a Streamlit application

## Problem Statement

Chronic diseases such as cardiovascular disease and diabetes are among the leading causes of death worldwide. Early prediction of these conditions can improve intervention and reduce long-term healthcare costs. Traditional diagnosis can be slow and dependent on manual interpretation, while machine learning can identify patterns in health data more efficiently. This project investigates whether common clinical and lifestyle indicators can be used to classify patients into **high-risk** and **low-risk** categories.  [oai_citation:3‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

## Scope of the Project

This project focuses on:

- tabular health datasets
- binary classification for disease risk prediction
- classical and ensemble machine learning models
- reproducible training and evaluation workflows
- interactive prediction through a lightweight web app

This project does **not** focus on:

- medical imaging
- wearable sensor streams
- clinical deployment
- real-time hospital integration

## Datasets

According to the Phase-2 report, the project uses:

- **UCI Heart Disease Dataset**
- **Kaggle Diabetes Health Indicators Dataset**  [oai_citation:4‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

The current repository is structured so it can run with a reproducible synthetic data generator, and it also supports placement of real CSV datasets in:

- `data/raw/heart.csv`
- `data/raw/diabetes.csv`  [oai_citation:5‡GitHub](https://github.com/Shazam-saifi/Health-Prediction)

### Expected Target Columns

- Heart disease dataset: `target`
- Diabetes dataset: `diabetes`  [oai_citation:6‡GitHub](https://github.com/Shazam-saifi/Health-Prediction)

## Features Used

The project works with health-related indicators such as:

- age
- blood pressure
- cholesterol
- BMI
- glucose level
- lifestyle-related indicators where available

Feature relevance is further explored through:

- correlation analysis
- feature importance
- domain-based interpretation from healthcare literature  [oai_citation:7‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

## Methodology

The project follows a standard machine learning pipeline.

### 1. Data Collection
Health datasets are collected and prepared for machine learning experiments.

### 2. Data Preprocessing
Preprocessing includes:

- missing value handling
- encoding of categorical variables
- feature scaling
- train-test splitting
- class imbalance mitigation using **SMOTE**
- exploratory data analysis (EDA)  [oai_citation:8‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

### 3. Feature Engineering
Feature engineering includes:

- correlation analysis
- feature selection support
- Random Forest feature importance
- domain relevance analysis  [oai_citation:9‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

### 4. Model Development
The main models used in Phase 2 are:

- **Logistic Regression** as an interpretable baseline
- **Random Forest** as a robust ensemble model  [oai_citation:10‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

### 5. Evaluation Strategy
Model performance is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- 5-fold cross-validation  [oai_citation:11‡Health Monitoring and Disease Prediction Using Machine Learning.docx](sediment://file_00000000ff3071fbaab8a71fdb71908e)

- Quick Start
Create and activate a virtual environment.
Install dependencies:
pip install -r requirements.txt
Train models and generate reports:
python3 -m health_predict_ai.train
Launch the app:
streamlit run app.py

## Repository Structure

The current repository includes the following main components:

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
data/
requirements.txt
README.md
