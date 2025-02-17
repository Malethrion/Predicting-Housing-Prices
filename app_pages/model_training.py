import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def app():
    """Train the model and save it for prediction."""
    st.title("Model Training")
    st.write("Training the model and saving it for predictions.")

    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run feature engineering first.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Ensure SalePrice exists and remove invalid values
    target = "SalePrice"
    if target not in data.columns:
        st.error(f"SalePrice column missing in {data_path}.")
        return
    
    data = data[data[target] > 0].copy()
    data[target] = np.log1p(data[target])  # Apply log transformation

    # Separate features and target
    X = data.drop(columns=[target])
    y = data[target]

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
        pickle.dump(list(X_train.columns), f)  # Save correct feature names
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("✅ Model training completed!")

if __name__ == "__main__":
    app()
