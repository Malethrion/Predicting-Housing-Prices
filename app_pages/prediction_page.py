import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Cache model loading to prevent reloading on every UI refresh
@st.cache_resource
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, feature_names, scaler

model, feature_names, scaler = load_model()

def predict_price(features):
    input_df = pd.DataFrame([features])

    # Ensure all required columns exist
    missing_cols = {col: 0 for col in feature_names if col not in input_df}
    input_df = pd.concat([input_df, pd.DataFrame([missing_cols])], axis=1)

    # Align columns to match model input
    input_df = input_df[feature_names]

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Predict price
    log_price = model.predict(input_scaled)
    predicted_price = np.expm1(log_price)

    return predicted_price[0]

# Streamlit UI
def app():
    st.title("Enter House Features")
    st.write("### Enter house features below to predict the price.")

    # Use unique keys to prevent duplicate ID errors
    features = {
        "GrLivArea": st.number_input("GrLivArea", value=1500, key="grlivarea_input_1"),
        "OverallQual": st.number_input("OverallQual", value=5, key="overallqual_input_1"),
        "GarageCars": st.number_input("GarageCars", value=2, key="garagecars_input_1"),
        "YearBuilt": st.number_input("YearBuilt", value=2000, key="yearbuilt_input_1"),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", value=1000, key="totalbsmt_input_1"),
    }

    # Predict button
    if st.button("Predict Price", key="predict_button_1"):
        price = predict_price(features)
        st.success(f"Predicted House Price: ${price:,.2f}")

print("Running Prediction Page...")
if __name__ == "__main__":
    app()
