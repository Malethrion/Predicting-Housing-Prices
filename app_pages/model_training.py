import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle
import os

def app():
    st.title("Model Training")

    # Load processed data
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please ensure preprocessing is completed.")
        return

    data = pd.read_csv(data_path)
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Ensure model directory exists
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    # Save model and feature names
    with open(f"{model_dir}/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    with open(f"{model_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    st.write("Model training completed. Saved model and feature names.")

