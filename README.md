# Customer Churn Prediction App

## Overview
An end-to-end machine learning pipeline designed to predict customer churn for banks, enabling proactive customer retention strategies.

## Features

### Machine Learning Models
- Multiple model implementations:
  - Logistic Regression
  - Naive Bayes
  - XGBoost
  - Random Forest Classifier
  - K-Nearest Neighbors (KNN)
  - XGBClassifier
  - Decision Tree Classifier
  - Support Vector Classifier (SVC)

- Advanced techniques:
  - Feature engineering
  - SMOTE for handling class imbalance
  - Voting Classifier
  - Hyperparameter tuning

### LLM Integration
- Llama 3.2 LLMs via GROQ API
  - Generate personalized retention emails
  - Provide prediction explanations

### Deployment
- Model APIs hosted on Render
- Web application for real-time predictions

## Dataset
- Kaggle Customer Churn Dataset-(https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
- Contents: Customer demographics, account information, usage patterns

## Development Workflow

### 1. Data Preparation
- Dataset cleaning
- Missing value handling
- Feature selection and engineering

### 2. Model Training
- Multi-model comparison
- SMOTE for class balance
- Hyperparameter optimization

### 3. Ensemble Learning
- Voting Classifier for improved predictions

### 4. AI-Powered Insights
- LLM-generated retention strategies

## Prerequisites
- Python 3.8+
- Libraries:
  - pandas
  - numpy
  - scikit-learn
  - xgboost
  - imbalanced-learn
  - flask
  - GROQ API library

## Getting Started

### Installation
```bash
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction
pip install -r requirements.txt
```

### Running the Application
```bash
flask run
```
Access at: `http://127.0.0.1:5000/`



## Notes
- Requires GROQ API and Render deployment credentials
