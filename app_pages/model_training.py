import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

# 🔹 Function to train and save the model
@st.cache_resource  # Ensures Streamlit doesn't retrain unless necessary
def train_model():
    """Loads data, trains model, and saves it."""
    st.write("## Model Training")
    st.write("Training the model and saving it for predictions.")

    # Load dataset
    data = pd.read_csv("data/processed_train.csv")
    target = "SalePrice"

    # 🔹 Debugging: Show initial SalePrice values
    st.write("### First 5 SalePrice values BEFORE transformation:")
    st.write(data[[target]].head())

    # If SalePrice is too small, scale it back
    if data[target].max() < 10000:
        st.warning("⚠️ SalePrice values seem too small! Multiplying by 100,000...")
        data[target] *= 100000  

    # Remove invalid values
    data = data[data[target] > 0].copy()

    # 🔹 Debugging: Show SalePrice statistics
    st.write("### SalePrice Stats After Scaling:")
    st.write(data[[target]].describe())

    # Apply log1p transformation
    y = np.log1p(data[target])
    X = data.drop(columns=[target])

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save trained model, feature names, and scaler
    os.makedirs("models", exist_ok=True)
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("✅ Model training completed!")
    return model, X_train.columns, scaler

# 🔹 Fix: Add the missing app() function
def app():
    """Streamlit UI for training the model."""
    train_model()  # Calls the training function

# 🔹 Ensure the model is trained only once per session
if "model_trained" not in st.session_state:
    st.session_state["model_trained"] = True
    app()  # Run the training when the app starts
