import streamlit as st
import pandas as pd
import pickle
import os
import numpy as np

def app():
    st.title("House Price Prediction")

    model_path = "models/trained_model.pkl"
    feature_path = "models/feature_names.pkl"
    scaler_path = "models/scaler.pkl"

    # Ensure model and feature files exist
    if not os.path.exists(model_path) or not os.path.exists(feature_path) or not os.path.exists(scaler_path):
        st.error("Model, scaler, or feature file not found. Please train the model first.")
        return

    # Load trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Load feature names
    with open(feature_path, "rb") as file:
        feature_names = pickle.load(file)

    # Load scaler
    with open(scaler_path, "rb") as file:
        scaler = pickle.load(file)

    # Features for user input
    selected_features = ["GrLivArea", "OverallQual", "GarageCars", "YearBuilt", "TotalBsmtSF"]

    st.subheader("Enter House Features")

    user_input = {}
    for feature in selected_features:
        default_values = {
            "GrLivArea": 1500,
            "OverallQual": 5,
            "GarageCars": 2,
            "YearBuilt": 2000,
            "TotalBsmtSF": 1000,
        }
        min_values = {
            "GrLivArea": 500,
            "OverallQual": 1,
            "GarageCars": 0,
            "YearBuilt": 1800,
            "TotalBsmtSF": 0,
        }
        max_values = {
            "GrLivArea": 5000,
            "OverallQual": 10,
            "GarageCars": 4,
            "YearBuilt": 2023,
            "TotalBsmtSF": 3000,
        }

        user_input[feature] = st.number_input(
            f"{feature}", 
            min_value=min_values[feature], 
            max_value=max_values[feature], 
            value=default_values[feature]
        )

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure the input has the same features as the model
    missing_features = set(feature_names) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = 0

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Apply the same scaling as training
    input_scaled = scaler.transform(input_df)

    # Predict button
    if st.button("Predict Price"):
        try:
            predicted_price = model.predict(input_scaled)[0]

            # Apply inverse transformation if needed
            predicted_price = np.exp(predicted_price)

            st.success(f"Predicted House Price: ${predicted_price:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
