import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and preprocessing tools
@st.cache_resource
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, feature_names, scaler

# Load model, features, and scaler
model, feature_names, scaler = load_model()

def predict_price(features):
    """Takes user input as a dictionary and predicts house price."""
    
    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([features])

    # Fill missing columns efficiently without causing fragmentation
    for col in feature_names:
        if col not in input_df:
            input_df[col] = 0  # Add missing columns with default 0

    # Reorder columns to match model training data
    input_df = input_df[feature_names]

    # Apply scaling
    input_scaled = scaler.transform(input_df)

    # Predict and transform back from log scale
    log_price = model.predict(input_scaled)
    predicted_price = np.expm1(log_price)  # Convert log1p back to normal scale

    return predicted_price[0]

# Streamlit UI
def app():
    st.title("Enter House Features")
    st.write("### Enter house features below to predict the price.")

    # User Input Fields
    features = {
        "GrLivArea": st.number_input("GrLivArea", value=1500, min_value=500, max_value=5000),
        "OverallQual": st.number_input("OverallQual", value=5, min_value=1, max_value=10),
        "GarageCars": st.number_input("GarageCars", value=2, min_value=0, max_value=5),
        "YearBuilt": st.number_input("YearBuilt", value=2000, min_value=1800, max_value=2023),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", value=1000, min_value=0, max_value=3000),
    }

    # Button to trigger prediction
    if st.button("Predict Price"):
        price = predict_price(features)
        st.success(f"Predicted House Price: ${price:,.2f}")

