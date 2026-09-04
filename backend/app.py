import os
import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel


# =======================================================
# STEP 1: Load the trained model
# =======================================================
# BASE_DIR = the folder this app.py file is in (e.g. "backend/")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# The model file was saved by ml/train.py, inside the "ml" folder
# which sits next to this "backend" folder.
MODEL_PATH = os.path.join(BASE_DIR, "..", "ml", "heart_disease_model.pkl")

# joblib.load() reads the saved pipeline back into memory.
# This pipeline already includes the scaler + one-hot encoder + the model,
# so we don't need to preprocess the data ourselves.
model = joblib.load(MODEL_PATH)

print("Model loaded successfully!")


# =======================================================
# STEP 2: Define what the input data should look like
# =======================================================
# This tells FastAPI: "a prediction request must have exactly these
# fields, with these types". FastAPI will automatically reject bad
# requests (e.g. missing a field, or sending text instead of a number).
#
# These 13 fields match the columns used to train the model
# (everything except the "target" column).

class PatientData(BaseModel):
    age: int              # e.g. 54
    sex: str              # "Male" or "Female"
    cp: str               # chest pain type, e.g. "typical angina"
    trestbps: float        # resting blood pressure
    chol: float             # cholesterol level
    fbs: bool              # fasting blood sugar > 120 mg/dl? True/False
    restecg: str           # resting ECG result, e.g. "normal"
    thalch: float           # max heart rate achieved
    exang: bool             # exercise induced angina? True/False
    oldpeak: float           # ST depression value
    slope: str               # slope of the ST segment, e.g. "flat"
    ca: float                 # number of major vessels (0-3)
    thal: str                 # thalassemia type, e.g. "normal"


# =======================================================
# STEP 3: Create the FastAPI app
# =======================================================
app = FastAPI(title="Heart Disease Prediction API")


# A simple "is it working" endpoint. Visit http://127.0.0.1:8000/ to check.
@app.get("/")
def home():
    return {"message": "Heart Disease Prediction API is running!"}


# The main prediction endpoint.
# It receives patient data (matching the PatientData shape above)
# and returns whether the model thinks they have heart disease.
@app.post("/predict")
def predict(patient: PatientData):

    # Convert the incoming data into a dictionary, e.g.:
    # {"age": 54, "sex": "Male", "cp": "typical angina", ...}
    patient_dict = patient.dict()

    # IMPORTANT: during training, fbs and exang were converted to
    # text ("True" / "False") before being fed into the model.
    # We must do the exact same thing here, or the model will be
    # confused by receiving actual booleans instead of text.
    patient_dict["fbs"] = str(patient_dict["fbs"])
    patient_dict["exang"] = str(patient_dict["exang"])

    # The model expects a table (DataFrame) of data, not a plain
    # dictionary — so we wrap it in a list to make a single row.
    input_data = pd.DataFrame([patient_dict])

    # Ask the model to predict: 0 = no disease, 1 = disease
    prediction = model.predict(input_data)[0]

    # Ask the model how confident it is (probability for each class)
    probabilities = model.predict_proba(input_data)[0]

    # Turn the raw prediction (0 or 1) into a readable message
    if prediction == 1:
        result_text = "Heart Disease Detected"
    else:
        result_text = "No Heart Disease"

    # Send the result back as JSON
    return {
        "prediction": int(prediction),
        "result": result_text,
        "confidence_no_disease": round(float(probabilities[0]), 2),
        "confidence_disease": round(float(probabilities[1]), 2),
    }