import streamlit as st
import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

def app():
    st.title("Model Training")
    st.write("Training the model and saving it for predictions.")

    # Load dataset
    data_path = "data/processed_train.csv"
    data = pd.read_csv(data_path)

    # Define target variable
    target = "SalePrice"

    # Ensure SalePrice has only positive values before log transformation
    data = data[data[target] > 0]  # Remove rows with zero or negative SalePrice

    # Apply log transformation
    data[target] = np.log(data[target])

    # Drop any remaining NaN values from features and target
    data = data.dropna()

    # Separate features (X) and target (y)
    X = data.drop(columns=[target])
    y = data[target]

    # Split dataset into training and testing sets (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save the trained model and necessary metadata
    os.makedirs("models", exist_ok=True)

    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/feature_names.pkl", "wb") as f:
        pickle.dump(X_train.columns.tolist(), f)

    # Save the scaler
    scaler = StandardScaler()
    scaler.fit(X_train)

    with open("models/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    st.success("Model training completed. Saved model, feature names, and scaler.")

