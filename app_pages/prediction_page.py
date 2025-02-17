import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ✅ Load trained model, feature names, and preprocessor
@st.cache_resource
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/preprocessor.pkl", "rb") as f:
        preprocessor = pickle.load(f)
    return model, feature_names, preprocessor

# ✅ Load model components
model, feature_names, preprocessor = load_model()

# ✅ Extract only numerical feature names (Ignore categorical ones)
num_features = preprocessor.named_transformers_['num'].feature_names_in_

# ✅ Prediction function
def predict_price(features):
    """Takes user input as a dictionary and predicts house price."""
    try:
        st.write("🔄 Converting input to DataFrame...")
        input_df = pd.DataFrame([features])

        # ✅ Ensure all required columns exist efficiently
        missing_cols = set(num_features) - set(input_df.columns)
        missing_data = pd.DataFrame(0, index=[0], columns=missing_cols)
        input_df = pd.concat([input_df, missing_data], axis=1)

        # ✅ Align columns to match model training order
        input_df = input_df[num_features]

        # ✅ Transform numerical inputs
        input_transformed = preprocessor.named_transformers_['num'].transform(input_df)

        # ✅ Debugging Output
        st.write("🔍 **Feature Names (Input to Model):**", num_features)
        st.write("🔍 **Transformed Input Shape:**", input_transformed.shape)
        st.write("🔍 **First Row of Transformed Input:**", input_transformed[0])

        # ✅ Predict using the trained model
        log_price = model.predict(input_transformed)

        # ✅ Convert log-transformed prediction back to actual price
        predicted_price = np.expm1(log_price)[0]

        return predicted_price

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None

# ✅ Streamlit UI for the Prediction Page
def app():
    st.title("🏡 House Price Prediction")
    st.write("### Enter house features below to predict the price.")

    # ✅ User input fields (ONLY numerical)
    numerical_inputs = {
        "GrLivArea": st.number_input("GrLivArea", min_value=500, max_value=10000, value=1500, step=100),
        "OverallQual": st.number_input("OverallQual", min_value=1, max_value=10, value=5, step=1),
        "GarageCars": st.number_input("GarageCars", min_value=0, max_value=5, value=2, step=1),
        "YearBuilt": st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000, step=1),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", min_value=0, max_value=5000, value=1000, step=100),
    }

    # ✅ Ensure predictions update dynamically
    if st.button("🔍 Predict Price"):
        price = predict_price(numerical_inputs)
        if price:
            st.success(f"💰 **Predicted House Price:** ${price:,.2f}")
