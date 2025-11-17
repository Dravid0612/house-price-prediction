import streamlit as st
import joblib
import pandas as pd
import os

st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏡",
    layout="centered"
)

MODEL_PATH = "models/pipeline.joblib"

# Load Model
if not os.path.exists(MODEL_PATH):
    st.error("Model file not found. Train the model first.")
    st.stop()

model = joblib.load(MODEL_PATH)
feature_names = model.named_steps["preprocessor"].transformers_[0][2]

st.title("🏡 House Price Prediction App")
st.write("Enter property details below to get an estimated market price.")

user_inputs = {}
cols = st.columns(2)

for i, feature in enumerate(feature_names):
    with cols[i % 2]:

        # Meaningful input ranges
        if "bed" in feature.lower():
            user_inputs[feature] = st.slider(feature, 0, 10, 3)

        elif "bath" in feature.lower():
            user_inputs[feature] = st.slider(feature, 0, 8, 2)

        elif "sqft" in feature.lower():
            user_inputs[feature] = st.number_input(feature, 200, 15000, 2000)

        elif "lot" in feature.lower():
            user_inputs[feature] = st.number_input(feature, 300, 50000, 3000)

        elif "floor" in feature.lower():
            user_inputs[feature] = st.slider(feature, 1.0, 4.0, 1.0)

        elif "view" in feature.lower():
            user_inputs[feature] = st.slider(feature, 0, 4, 0)

        elif "condition" in feature.lower():
            user_inputs[feature] = st.slider(feature, 1, 5, 3)

        elif "yr_built" in feature.lower():
            user_inputs[feature] = st.number_input(feature, 1900, 2025, 1995)

        elif "yr_renovated" in feature.lower():
            user_inputs[feature] = st.number_input(feature, 0, 2025, 0)

        else:
            user_inputs[feature] = st.number_input(feature, 0.0, 10000.0, 1.0)

st.write("---")

if st.button("Predict Price", use_container_width=True):
    input_df = pd.DataFrame([user_inputs], columns=feature_names)
    price = model.predict(input_df)[0]
    st.success(f"Estimated House Price: **${price:,.2f}**")

st.write("---")
st.caption("Built with Streamlit & Scikit-Learn 💙")
