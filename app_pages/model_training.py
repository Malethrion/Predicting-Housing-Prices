import streamlit as st
import pickle
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def app():
    """Streamlit UI for training the model."""
    st.title("Model Training")
    st.write("Training the model and saving it for predictions.")

    # Load dataset
    data = pd.read_csv("data/processed_train.csv")

    target = "SalePrice"
    X = data.drop(columns=[target])
    y = np.log(data[target])  # Apply log transformation

    # Ensure no NaN values in y
    y.replace([np.inf, -np.inf], np.nan, inplace=True)  # Replace infinities
    y.dropna(inplace=True)  # Drop any NaN values

    # Ensure X and y have the same length after dropping NaN values
    X = X.loc[y.index]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Verify no NaN values exist before training
    st.write(f"Missing values in y_train after cleaning: {y_train.isnull().sum()}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train_scaled, y_train)

    # Ensure the models directory exists
    os.makedirs("models", exist_ok=True)

    # Save trained model
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    # Save feature names
    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    # Save scaler
    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    # Display success message
    st.success("Model training completed. Saved model, feature names, and scaler.")
