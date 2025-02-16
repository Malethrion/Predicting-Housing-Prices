import streamlit as st
import pandas as pd
import pickle
import os

def app():
    st.title("Prediction Page")

    model_path = "models/trained_model.pkl"
    feature_path = "models/feature_names.pkl"

    # Ensure model and feature file exist
    if not os.path.exists(model_path) or not os.path.exists(feature_path):
        st.error(f"Model or feature file not found. Train the model first.")
        return

    # Load trained model
    with open(model_path, "rb") as file:
        model = pickle.load(file)

    # Load feature names
    with open(feature_path, "rb") as file:
        feature_names = pickle.load(file)

    # Select a few features for user input
    selected_features = ["GrLivArea", "OverallQual", "GarageCars", "YearBuilt", "TotalBsmtSF"]

    st.write("### Enter House Features")

    user_input = {}
    for feature in selected_features:
        user_input[feature] = st.number_input(f"{feature}", value=0.0)

    # Convert input to DataFrame
    input_df = pd.DataFrame([user_input])

    # Ensure the input has the same features as the model
    missing_features = set(feature_names) - set(input_df.columns)
    for feature in missing_features:
        input_df[feature] = 0  # Fill missing features with 0

    # Ensure correct column order
    input_df = input_df[feature_names]

    # Predict button
    if st.button("Predict Price"):
        predicted_price = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${predicted_price:,.2f}")


