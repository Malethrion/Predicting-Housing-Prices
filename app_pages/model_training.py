import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pickle

def app():
    st.title("Model Training")

    # Load dataset
    data = pd.read_csv("../data/processed_train.csv")
    target = "SalePrice"

    X = data.drop(columns=[target])
    y = data[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Save trained model
    with open("../models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.write("Model Training Complete and Saved.")
