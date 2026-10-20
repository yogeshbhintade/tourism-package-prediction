"""
Streamlit app - Wellness Tourism Package purchase prediction for "Visit with Us".

Loads the model that the GitHub Actions pipeline committed to
tourism_project/deployment/, collects customer details, stores them in a
dataframe and predicts whether the customer is likely to buy the package.
"""
import json
import os

import joblib
import pandas as pd
import streamlit as st

# The model lives next to this file in the repository
APP_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(APP_DIR, "best_tourism_model_v1.joblib")
METADATA_PATH = os.path.join(APP_DIR, "model_metadata.json")


@st.cache_resource
def load_model():
    """Load the trained pipeline (preprocessing + XGBoost) and its threshold."""
    model = joblib.load(MODEL_PATH)
    threshold = 0.5
    if os.path.exists(METADATA_PATH):
        with open(METADATA_PATH) as f:
            threshold = json.load(f).get("classification_threshold", 0.5)
    return model, threshold


st.set_page_config(page_title="Wellness Package Predictor", layout="wide")
st.title("Wellness Tourism Package - Purchase Prediction")
st.write(
    "Enter the customer's details and the sales-interaction data to predict whether "
    "they are likely to purchase the **Wellness Tourism Package** before the sales team contacts them."
)

if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Run the GitHub Actions pipeline so it commits the trained model.")
    st.stop()

model, threshold = load_model()

# ---------------------------------------------------------------------------
# Collect user inputs
# ---------------------------------------------------------------------------
st.subheader("Customer details")
c1, c2, c3 = st.columns(3)
with c1:
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital status", ["Married", "Single", "Unmarried", "Divorced"])
    occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Freelancer"])
    designation = st.selectbox("Designation", ["Executive", "Manager", "Senior Manager", "AVP", "VP"])
with c2:
    monthly_income = st.number_input("Monthly income", min_value=0.0, max_value=500000.0, value=23000.0, step=500.0)
    city_tier = st.selectbox("City tier", [1, 2, 3])
    passport = st.selectbox("Has a valid passport?", ["Yes", "No"])
    own_car = st.selectbox("Owns a car?", ["Yes", "No"])
    number_of_trips = st.number_input("Average trips per year", min_value=0, max_value=30, value=3)
with c3:
    number_of_person_visiting = st.number_input("People visiting", min_value=1, max_value=10, value=3)
    number_of_children_visiting = st.number_input("Children (<5 yrs) visiting", min_value=0, max_value=5, value=1)
    preferred_property_star = st.selectbox("Preferred hotel rating", [3, 4, 5])

# Save the inputs into a single-row dataframe with the training column names
input_data = pd.DataFrame([{
    "Age": age,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children_visiting,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
}])

with st.expander("Show input dataframe"):
    st.dataframe(input_data, use_container_width=True)

# ---------------------------------------------------------------------------
# Predict
# ---------------------------------------------------------------------------
if st.button("Predict", type="primary"):
    probability = float(model.predict_proba(input_data)[0, 1])
    if probability >= threshold:
        st.success(f"Likely to purchase the Wellness Tourism Package (probability {probability:.1%}). "
                   "Prioritise this customer for outreach.")
    else:
        st.warning(f"Unlikely to purchase (probability {probability:.1%}). "
                   "Consider a lower-cost touchpoint instead of a full sales call.")
    st.caption(f"Decision threshold: {threshold:.2f}")
