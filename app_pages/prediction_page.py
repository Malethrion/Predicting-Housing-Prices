import streamlit as st
import pickle
import numpy as np
import pandas as pd

# ✅ Load trained model, feature names, and scaler
@st.cache_resource
def load_model():
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    return model, feature_names, scaler

# Load model components
model, feature_names, scaler = load_model()

def predict_price(features):
    """Takes user input as a dictionary and predicts house price."""
    try:
        # Convert dictionary to DataFrame
        input_df = pd.DataFrame([features])

        # ✅ Identify missing columns
        missing_cols = set(feature_names) - set(input_df.columns)

        # ✅ Efficiently add missing columns at once (Avoid Fragmentation Warning)
        if missing_cols:
            missing_data = pd.DataFrame(0, index=[0], columns=list(missing_cols))
            input_df = pd.concat([input_df, missing_data], axis=1)

        # ✅ Ensure correct column order
        input_df = input_df[feature_names].copy()

        # ✅ Apply scaling
        input_scaled = scaler.transform(input_df)

        # ✅ Predict and transform back from log scale
        log_price = model.predict(input_scaled)
        predicted_price = np.expm1(log_price)[0]  # Convert log1p back to normal scale

        return predicted_price

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")
        return None

# ✅ Streamlit UI
def app():
    st.title("🏡 Enter House Features")
    st.write("### Enter house features below to predict the price.")

    # ✅ Dynamic User Inputs
    features = {
        "GrLivArea": st.number_input("GrLivArea", value=1500, min_value=500, max_value=5000, step=100),
        "OverallQual": st.number_input("OverallQual", value=5, min_value=1, max_value=10, step=1),
        "GarageCars": st.number_input("GarageCars", value=2, min_value=0, max_value=5, step=1),
        "YearBuilt": st.number_input("YearBuilt", value=2000, min_value=1800, max_value=2023, step=1),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", value=1000, min_value=0, max_value=3000, step=100),
    }

    # ✅ Ensure predictions dynamically update
    if st.button("🔍 Predict Price"):
        price = predict_price(features)
        if price:
            st.success(f"💰 **Predicted House Price:** ${price:,.2f}")

if __name__ == "__main__":
    app()
