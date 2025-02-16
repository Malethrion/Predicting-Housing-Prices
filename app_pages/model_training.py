import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import os
import numpy as np

def app():
    st.title("Model Training")

    # Load processed data
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please ensure preprocessing is completed.")
        return

    data = pd.read_csv(data_path)

    # Remove rows where SalePrice is zero or negative before log transformation
    data = data[data["SalePrice"] > 0]

    # Apply log transformation
    y = np.log(data["SalePrice"])
    X = data.drop(columns=["SalePrice"])

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Ensure model directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model, feature names, and scaler
    with open(f"{model_dir}/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    with open(f"{model_dir}/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.write("Model training completed. Saved model, feature names, and scaler.")

