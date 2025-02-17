import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def app():
    """Train model and save it for prediction."""
    st.title("Model Training")
    st.write("Training the model and saving it for predictions.")

    # Load dataset
    data = pd.read_csv("data/processed_train.csv")

    target = "SalePrice"

    # If SalePrice is too small, scale it back
    if data[target].max() < 10000:
        data[target] *= 100000  

    # Remove rows where SalePrice is invalid (≤ 0)
    data = data[data[target] > 0].copy()

    # Apply log1p transformation
    y = np.log1p(data[target])
    X = data.drop(columns=[target])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Save model, scaler, and feature names
    os.makedirs("models", exist_ok=True)
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("Model training completed!")

if __name__ == "__main__":
    app()
