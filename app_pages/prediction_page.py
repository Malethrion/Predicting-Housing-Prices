import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and scaler
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

def predict_price(features):
    """
    Takes user input as a dictionary and predicts house price.
    """
    # Convert input dictionary to dataframe
    input_df = pd.DataFrame([features])

    # Add missing columns all at once
    missing_cols = {col: 0 for col in feature_names if col not in input_df}
    input_df = input_df.assign(**missing_cols)  # Efficiently add missing columns

    # Align columns to match training data
    input_df = input_df[feature_names]

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Predict and transform back from log scale
    log_price = model.predict(input_scaled)
    predicted_price = np.exp(log_price)  # Convert log price back to normal scale

    return predicted_price[0]

# Streamlit UI
def app():
    st.title("Enter House Features")
    st.write("### Enter house features below to predict the price.")

    # Create input fields for user (Kept as per your request)
    features = {
        "GrLivArea": st.number_input("GrLivArea", value=1500),
        "OverallQual": st.number_input("OverallQual", value=5),
        "GarageCars": st.number_input("GarageCars", value=2),
        "YearBuilt": st.number_input("YearBuilt", value=2000),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", value=1000),
    }

    if st.button("Predict Price"):
        price = predict_price(features)
        st.success(f"Predicted House Price: ${price:,.2f}")

