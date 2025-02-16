import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def app():
    st.title("Model Training")

    # Load dataset
    data = pd.read_csv("../data/processed_train.csv")

    # Train-test split
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("../models/trained_model.pkl", "wb") as file:
        pickle.dump(model, file)

    st.write("Model training complete. Model saved as `trained_model.pkl`.")
