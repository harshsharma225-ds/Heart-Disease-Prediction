import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

import joblib
import json
import os

import warnings
warnings.filterwarnings("ignore")
print("Libraries Imported Successfully")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "heart_disease_uci.csv")  # CSV sits in the same ml/ folder as this script

df = pd.read_csv(DATA_PATH)
print(df.head())
print(df.shape)
df.info()
print(df.describe())
print(df.describe(include="object"))
print(df.isnull().sum())
print(df.duplicated().sum())
print(df["num"].value_counts().sort_index())

df_clean = df.drop(columns=["id"])

df_clean["target"] = (df_clean["num"] > 0).astype(int)
df_clean = df_clean.drop(columns=["num"])
print(df_clean["target"].value_counts())

df_clean["fbs"] = df_clean["fbs"].astype(bool)
df_clean["exang"] = df_clean["exang"].astype(bool)

df_clean["trestbps"] = df_clean["trestbps"].fillna(df_clean["trestbps"].median())
df_clean["chol"] = df_clean["chol"].fillna(df_clean["chol"].median())
df_clean["thalch"] = df_clean["thalch"].fillna(df_clean["thalch"].median())
df_clean["oldpeak"] = df_clean["oldpeak"].fillna(df_clean["oldpeak"].median())
df_clean["ca"] = df_clean["ca"].fillna(df_clean["ca"].median())

df_clean["restecg"] = df_clean["restecg"].fillna(df_clean["restecg"].mode()[0])
df_clean["slope"] = df_clean["slope"].fillna(df_clean["slope"].mode()[0])
df_clean["thal"] = df_clean["thal"].fillna(df_clean["thal"].mode()[0])

print(df_clean.isnull().sum())

print((df_clean['chol'] == 0).sum())
df_clean["chol"] = df_clean["chol"].replace(0, df_clean["chol"].median())
print((df_clean['chol'] == 0).sum())

print(df_clean.head())

df_model = df_clean.drop(columns=["dataset"])

X = df_model.drop(columns=["target"])
y = df_model["target"]

print(X.shape)
print(y.shape)

num_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
cat_features = X.select_dtypes(include=["object", "boolean"]).columns.tolist()

print("Numerical : ", num_features)
print("Categorical : ", cat_features)

for col in cat_features:
    X[col] = X[col].astype(str)

sc = StandardScaler()
ohe = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", sc, num_features),
        ("cat", ohe, cat_features)
    ]
)
print(preprocessor)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Training set size:", X_train.shape)
print("Testing set size:", X_test.shape)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Decision Tree ": DecisionTreeClassifier(random_state=42),
    "Random Forest ": RandomForestClassifier(n_estimators=200),
    "KNN": KNeighborsClassifier(n_neighbors=7)
}

pipelines = {}
for name, model in models.items():
    pipelines[name] = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])
print("Pipeline created for : ", list(pipelines.keys()))

for name, pipe in pipelines.items():
    pipe.fit(X_train, y_train)

print("All Models Trained on Training set")

result = []

for name, pipe in pipelines.items():
    y_pred = pipe.predict(X_test)

    result.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 score ": f1_score(y_test, y_pred)
    })

result_df = pd.DataFrame(result)
print(result_df)

best_model_name = result_df.loc[result_df["F1 score "].idxmax(), "Model"]
best_pipeline = pipelines[best_model_name]
print(f"Best model: {best_model_name}")

joblib.dump(best_pipeline, os.path.join(BASE_DIR, "heart_disease_model.pkl"))

joblib.dump(preprocessor, os.path.join(BASE_DIR, "preprocessor.pkl"))

joblib.dump(sc, os.path.join(BASE_DIR, "scaler.pkl"))
joblib.dump(ohe, os.path.join(BASE_DIR, "onehot_encoder.pkl"))

metadata = {
    "best_model": best_model_name,
    "input_columns": X.columns.tolist(),
    "numerical_features": num_features,
    "categorical_features": cat_features,
    "target_column": "target"
}
with open(os.path.join(BASE_DIR, "model_metadata.json"), "w") as f:
    json.dump(metadata, f, indent=2)

result_df.to_csv(os.path.join(BASE_DIR, "model_comparison.csv"), index=False)

print("Model, preprocessor, scaler, encoder and metadata saved successfully.")