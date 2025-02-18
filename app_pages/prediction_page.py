import streamlit as st
import pickle
import numpy as np
import pandas as pd

def load_model():
    """Load trained model, feature names, and preprocessor."""
    try:
        with open("models/trained_model.pkl", "rb") as f:
            model = pickle.load(f)
        with open("models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
        with open("models/preprocessor.pkl", "rb") as f:
            preprocessor = pickle.load(f)
        return model, feature_names, preprocessor
    except Exception as e:
        st.error(f"Error loading model components: {e}")
        return None, None, None

# Load model components
model, feature_names, preprocessor = load_model()

# Define default values for missing categorical features
categorical_defaults = {
    "MSZoning": "RL",
    "Street": "Pave",
    "LotShape": "Reg",
    "LandContour": "Lvl",
    "Utilities": "AllPub",
    "Neighborhood": "NAmes",
    "Condition1": "Norm",
    "BldgType": "1Fam",
    "HouseStyle": "1Story",
    "RoofStyle": "Gable",
    "Foundation": "PConc",
    "Heating": "GasA",
    "CentralAir": "Y",
    "Electrical": "SBrkr",
    "Functional": "Typ",
    "GarageType": "Attchd",
    "SaleType": "WD",
    "SaleCondition": "Normal"
}

def predict_price(features):
    """Processes input and returns the predicted house price."""
    try:
        input_df = pd.DataFrame([features])

        # Ensure all required categorical and numerical columns exist
        missing_cols = list(set(feature_names) - set(input_df.columns))
        missing_data = pd.DataFrame(0, index=[0], columns=missing_cols)
        
        # Fill missing categorical values with default values
        for col, default_value in categorical_defaults.items():
            if col in missing_data.columns:
                missing_data[col] = default_value
        
        # Use pd.concat() to efficiently merge missing columns and avoid fragmentation
        input_df = pd.concat([input_df, missing_data], axis=1)
        input_df = input_df.copy()  # Ensure de-fragmented DataFrame

        # Align columns to match model training order
        input_df = input_df[feature_names]

        # Transform using preprocessor
        input_transformed = preprocessor.transform(input_df)

        # Predict and inverse log transformation
        log_price = model.predict(input_transformed)
        predicted_price = np.expm1(log_price)[0]

        return predicted_price
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return None

# Streamlit UI

def app():
    st.title("🏡 House Price Prediction")
    st.write("### Enter house features below to predict the price.")

    numerical_inputs = {
        "GrLivArea": st.number_input("GrLivArea", min_value=500, max_value=10000, value=1500, step=100),
        "OverallQual": st.number_input("OverallQual", min_value=1, max_value=10, value=5, step=1),
        "GarageCars": st.number_input("GarageCars", min_value=0, max_value=5, value=2, step=1),
        "YearBuilt": st.number_input("YearBuilt", min_value=1800, max_value=2025, value=2000, step=1),
        "TotalBsmtSF": st.number_input("TotalBsmtSF", min_value=0, max_value=5000, value=1000, step=100),
    }

    if st.button("🔍 Predict Price"):
        price = predict_price(numerical_inputs)
        if price:
            st.success(f"💰 **Predicted House Price:** ${price:,.2f}")

if __name__ == "__main__":
    app()
