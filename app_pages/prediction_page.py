import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load trained model and scaler
@st.cache_data
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
    """Takes user input as a dictionary and predicts house price."""
    st.write("### Debug: User Input")
    st.json(features)  # Display user input for debugging

    # Convert input dictionary to DataFrame
    input_df = pd.DataFrame([features])

    # Ensure all required columns exist
    missing_cols = {col: 0 for col in feature_names if col not in input_df.columns}
    input_df = input_df.assign(**missing_cols)

    # Align columns to match the model’s expected feature order
    input_df = input_df[feature_names]

    # Debugging: Print transformed input before scaling
    st.write("### Debug: Transformed Input Before Scaling")
    st.dataframe(input_df)

    # Scale input data
    input_scaled = scaler.transform(input_df)

    # Debugging: Print scaled input
    st.write("### Debug: Transformed Input After Scaling")
    st.dataframe(pd.DataFrame(input_scaled, columns=feature_names))

    # Predict using the trained model
    log_price = model.predict(input_scaled)

    # Convert log-transformed prediction back to actual price
    predicted_price = np.expm1(log_price)[0]

    # Debugging: Print prediction
    st.write(f"### Debug: Raw Model Output (Log Scale): {log_price}")
    st.write(f"### Debug: Final Predicted Price: {predicted_price}")

    return predicted_price

# Streamlit UI
def app():
    st.title("Enter House Features")
    st.write("### Enter house features below to predict the price.")

    # User input fields
    features = {
        "GrLivArea": st.number_input("GrLivArea", min_value=500, max_value=10000, value=1500, step=100, key="grliv"),
        "OverallQual": st.number_input("OverallQual", min_value=1, max_value=10, value=5, step=1, key="qual"),
        "GarageCars": st.number_input("GarageCars", min_value=0, max_value=5, value=2, step=1, key="garage"),
        "YearBuilt": st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000, step=1, key="year"),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", min_value=0, max_value=5000, value=1000, step=100, key="bsmt"),
    }

    if st.button("Predict Price"):
        price = predict_price(features)
        st.success(f"Predicted House Price: ${price:,.2f}")

