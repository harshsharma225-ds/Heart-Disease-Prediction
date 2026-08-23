# ❤️ Heart Disease Prediction

A complete machine learning workflow for predicting the presence of heart disease in patients using clinical and demographic data, built in a single, well-documented Jupyter notebook.

## 📌 Project Overview

Cardiovascular disease is one of the leading causes of death worldwide. This project uses the classic **UCI Heart Disease dataset** (combining data from Cleveland, Hungary, Switzerland, and VA Long Beach hospitals) to build and compare multiple classification models that predict whether a patient has heart disease based on clinical measurements.

The notebook covers the entire ML pipeline end to end:

1. Importing Libraries
2. Loading & Inspecting Data
3. Data Cleaning
4. Exploratory Data Analysis (EDA) & Visualization
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Model Evaluation (with hyperparameter tuning)
9. Conclusion

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data handling:** pandas, numpy
- **Visualization:** matplotlib, seaborn
- **Modeling:** scikit-learn (Logistic Regression, Decision Tree, Random Forest, KNN)
- **Environment:** Jupyter Notebook

## 🔍 Workflow Summary

- **Cleaning:** Handled missing values (median imputation for numeric, mode for categorical), fixed invalid cholesterol entries, converted the multi-class target into a binary label.
- **EDA:** Visualized target balance, age distribution, chest pain type vs. disease, correlation heatmap, and more to surface key clinical risk patterns.
- **Preprocessing:** Built a `ColumnTransformer` + `Pipeline` combining standard scaling for numeric features and one-hot encoding for categorical features.
- **Modeling:** Trained and compared Logistic Regression, Decision Tree, Random Forest, and KNN.
- **Evaluation:** Compared models on Accuracy, Precision, Recall, F1-score.
