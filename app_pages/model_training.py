import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle

def app():
    st.title("Model Training")

    # Load processed data
    data = pd.read_csv("data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save model
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.write("Model training completed. Saved model to 'trained_model.pkl'.")

