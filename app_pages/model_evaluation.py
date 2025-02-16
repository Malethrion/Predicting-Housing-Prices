import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def app():
    st.title("Model Evaluation")

    # Load test data
    data = pd.read_csv("../data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Load trained model
    with open("../models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Predictions
    y_pred = model.predict(X)

    # Compute metrics
    st.write(f"Mean Absolute Error: {mean_absolute_error(y, y_pred)}")
    st.write(f"Mean Squared Error: {mean_squared_error(y, y_pred)}")
    st.write(f"R-squared Score: {r2_score(y, y_pred)}")
