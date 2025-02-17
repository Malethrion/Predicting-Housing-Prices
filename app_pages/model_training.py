import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print("🚀 Script started...")

def app():
    """Streamlit UI for training the model."""
    print("Starting app() function...")
    st.title("Model Training")
    st.write("Training the model and saving it for predictions.")

    # Load dataset
    print("Loading dataset...")
    data = pd.read_csv("data/processed_train.csv")
    print("Data loaded. Shape:", data.shape)

    target = "SalePrice"

    # Debug: Show first few SalePrice values
    print("First 5 SalePrice values BEFORE transformation:")
    print(data[target].head())

    # If SalePrice is too small, scale it back
    if data[target].max() < 10000:
        print("SalePrice values seem too small! Multiplying by 100,000...")
        data[target] *= 100000  

    # Remove rows where SalePrice is invalid (≤ 0)
    data = data[data[target] > 0].copy()

    # Debug: SalePrice statistics
    print("SalePrice Stats After Scaling:")
    print(data[target].describe())

    # Apply log1p transformation
    y = np.log1p(data[target])

    # Debug: Check transformed values
    print(f"Log-transformed SalePrice min: {y.min()}, max: {y.max()}")

    X = data.drop(columns=[target])

    # Train-test split
    print("Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Training samples: {len(y_train)}, Test samples: {len(y_test)}")

    # Scale features
    print("Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print("Scaling complete.")

    # Train RandomForest model
    print("Training RandomForest model...")
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)
    print("Model training completed!")

    # Save the trained models
    os.makedirs("models", exist_ok=True)
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(list(X_train.columns), f)
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    print("Model, feature names, and scaler saved.")
    st.success("Model training completed!")

print("Running app() function...")
app()
print("Script finished successfully!")
