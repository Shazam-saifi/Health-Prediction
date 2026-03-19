# HealthPredict AI

HealthPredict AI is a machine learning-based system designed to predict the risk of chronic diseases such as heart disease and diabetes using structured health data. The project applies data science and machine learning techniques to support early disease detection and improve healthcare decision-making.

## Project Objective

The goal of this project is to:
- Predict disease risk using patient health data  
- Compare machine learning models  
- Handle class imbalance in datasets  
- Provide an interactive Streamlit interface  

## Problem Statement

Chronic diseases are a major global health concern. Early detection can improve outcomes, but traditional diagnosis can be slow and subjective. This project uses machine learning to predict disease risk based on clinical and lifestyle features.

## Datasets

- UCI Heart Disease Dataset  
- Kaggle Diabetes Health Indicators Dataset  

Target variables:
- Heart dataset → target  
- Diabetes dataset → diabetes  

## Features Used

- Age  
- Blood Pressure  
- Cholesterol  
- BMI  
- Glucose  

## Methodology

1. Data Preprocessing  
   - Missing value handling  
   - Encoding  
   - Scaling  
   - SMOTE for class imbalance  

2. Feature Engineering  
   - Correlation analysis  
   - Feature importance  

3. Model Development  
   - Logistic Regression  
   - Random Forest  

4. Evaluation  
   - Accuracy  
   - Precision  
   - Recall  
   - F1-score  
   - ROC-AUC  
   - Cross-validation  

## Model Results

Heart Disease Dataset:
- Logistic Regression → Accuracy: 0.83, ROC-AUC: 0.85  
- Random Forest → Accuracy: 0.86, ROC-AUC: 0.88  

Diabetes Dataset:
- Logistic Regression → Accuracy: 0.79, ROC-AUC: 0.81  
- Random Forest → Accuracy: 0.84, ROC-AUC: 0.86  

Key Insight:
Random Forest performs better, and SMOTE improves recall for high-risk cases.

## Repository Structure

health_predict_ai/
- config.py
- data.py
- features.py
- modeling.py
- pipeline.py
- train.py
- predict.py
- explain.py

app.py  
data/  
tests/  
requirements.txt  
README.md  

## Installation

1. Clone repository:
git clone https://github.com/Shazam-saifi/Health-Prediction.git  
cd Health-Prediction  

2. Create virtual environment:
python -m venv venv  

Activate:

Windows:
venv\Scripts\activate  

Mac/Linux:
source venv/bin/activate  

3. Install dependencies:
pip install -r requirements.txt  

## How to Run

Train model:
python -m health_predict_ai.train  

Run app:
streamlit run app.py  

## Streamlit Application

- Input health data  
- Get prediction  
- View risk classification  

## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Streamlit  
- Matplotlib  
- Joblib  

## Limitations

- Uses public datasets  
- Not for clinical use  
- For research purposes only  

## Future Work

- Hyperparameter tuning  
- XGBoost  
- SHAP improvements  
- UI improvements  
- Deployment  
