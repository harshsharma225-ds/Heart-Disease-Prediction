import streamlit as st
import requests


# =======================================================
# STEP 1: Page title and setup
# =======================================================
st.title("Heart Disease Prediction App")
st.write("Fill in the details below, then click the Predict button.")

# This is the web address of our FastAPI backend.
# Streamlit will send the form data here when the button is clicked.
BACKEND_URL = "http://127.0.0.1:8000/predict"


# =======================================================
# STEP 2: Collect input from the user
# =======================================================
# Each line below creates one input box or dropdown on the page.
# The value the user picks gets stored in a variable.

age = st.number_input("Age", min_value=1, max_value=120, value=54)

sex = st.selectbox("Sex", ["Male", "Female"])

cp = st.selectbox(
    "Chest Pain Type",
    ["typical angina", "atypical angina", "non-anginal", "asymptomatic"]
)

trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=50, max_value=250, value=130)

chol = st.number_input("Cholesterol (mg/dl)", min_value=50, max_value=600, value=246)

# For yes/no questions, we show the words "Yes" and "No" to the user
# (since that's easier to read than True/False), and convert their
# choice into an actual True/False value ourselves below.
fbs_choice = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", ["No", "Yes"])
if fbs_choice == "Yes":
    fbs = True
else:
    fbs = False

restecg = st.selectbox(
    "Resting ECG Result",
    ["normal", "st-t abnormality", "lv hypertrophy"]
)

thalch = st.number_input("Max Heart Rate Achieved", min_value=50, max_value=250, value=150)

exang_choice = st.selectbox("Exercise Induced Angina?", ["No", "Yes"])
if exang_choice == "Yes":
    exang = True
else:
    exang = False

oldpeak = st.number_input("ST Depression (oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)

slope = st.selectbox("Slope of Peak Exercise ST Segment", ["upsloping", "flat", "downsloping"])

ca = st.number_input("Number of Major Vessels (0-3)", min_value=0, max_value=3, value=0)

thal = st.selectbox("Thalassemia", ["normal", "fixed defect", "reversable defect"])


# =======================================================
# STEP 3: When the user clicks "Predict"
# =======================================================
predict_button = st.button("Predict")

if predict_button:

    # Put all the collected values into one dictionary.
    # The keys here must match what the FastAPI backend expects.
    patient_data = {
        "age": age,
        "sex": sex,
        "cp": cp,
        "trestbps": trestbps,
        "chol": chol,
        "fbs": fbs,
        "restecg": restecg,
        "thalch": thalch,
        "exang": exang,
        "oldpeak": oldpeak,
        "slope": slope,
        "ca": ca,
        "thal": thal,
    }

    # Send this data to the backend and wait for its reply.
    response = requests.post(BACKEND_URL, json=patient_data)

    # If the backend replied successfully (status code 200 means "OK")...
    if response.status_code == 200:

        # Convert the backend's reply from JSON into a Python dictionary
        result = response.json()

        # Pull out the pieces we want to show
        prediction_text = result["result"]
        confidence_disease = result["confidence_disease"]
        confidence_no_disease = result["confidence_no_disease"]

        st.subheader("Result")

        # Show a red warning box if disease is predicted,
        # or a green success box if not.
        if result["prediction"] == 1:
            st.error(prediction_text)
        else:
            st.success(prediction_text)

        st.write("Confidence of having heart disease:", confidence_disease)
        st.write("Confidence of NOT having heart disease:", confidence_no_disease)

    else:
        # If something went wrong (backend not running, bad data, etc.)
        st.write("Something went wrong. Please check that the backend is running.")