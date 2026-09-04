# ❤️ Heart Disease Prediction

An end-to-end machine learning project that predicts the likelihood of heart disease from patient clinical data. It includes data cleaning & model training, a FastAPI backend to serve predictions, and a Streamlit frontend for users to interact with the model.

## 📋 Overview

This project uses the UCI Heart Disease dataset to train and compare multiple classification models, then serves the best-performing one through a REST API with a simple web interface on top.

**Pipeline:**
- **Raw Data** — heart_disease_uci.csv
- **Cleaning & Preprocessing** — handle missing values, fix invalid entries, scale/encode features
- **Model Training & Comparison** — train Logistic Regression, Decision Tree, Random Forest, and KNN
- **Best Model Saved** — selected automatically based on F1 score
- **FastAPI Backend** — loads the saved model and serves predictions via a `/predict` endpoint
- **Streamlit Frontend** — user enters patient details and views the prediction result

## 🚀 Features

- Data cleaning: handles missing values, invalid entries (e.g. cholesterol = 0), and type conversions
- Preprocessing pipeline: `StandardScaler` for numerical features, `OneHotEncoder` for categorical features
- Trains and compares 4 models: Logistic Regression, Decision Tree, Random Forest, and KNN
- Automatically selects the best model based on F1 score
- REST API (FastAPI) with interactive Swagger docs
- Web UI (Streamlit) for entering patient details and viewing predictions with confidence scores

## 🛠️ Tech Stack

| Layer | Tools |
|---|---|
| Data & ML | Python, Pandas, NumPy, Scikit-learn |
| Model Persistence | Joblib |
| Backend API | FastAPI, Uvicorn |
| Frontend | Streamlit |

## 📁 Project Structure
- **heart_disease/** (root folder)
  - `heart_disease_prediction.ipynb` — Exploratory notebook (EDA + experiments)
  - `requirements.txt` — Python dependencies
  - **ml/**
    - `train.py` — Cleans data, trains models, saves the best one
    - `heart_disease_uci.csv` — Dataset
    - `heart_disease_model.pkl` — Saved best model (generated)
    - `preprocessor.pkl` — Saved preprocessing pipeline (generated)
    - `model_metadata.json` — Feature names & best model info (generated)
    - `model_comparison.csv` — Metrics for all trained models (generated)
  - **backend/**
    - `app.py` — FastAPI app serving predictions
  - **frontend/**
    - `streamlit_app.py` — Streamlit UI

## 📊 Dataset

The [UCI Heart Disease dataset](https://archive.ics.uci.edu/dataset/45/heart+disease), combining data from multiple hospitals, with 13 clinical features (age, chest pain type, resting blood pressure, cholesterol, ECG results, etc.) used to predict presence of heart disease.

## 📈 Model Performance

Model comparison (Accuracy, Precision, Recall, F1 Score) is generated automatically in `ml/model_comparison.csv` after running `train.py`, with the best model (by F1 score) selected and saved for deployment.
