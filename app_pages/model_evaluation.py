import streamlit as st
import pandas as pd
import pickle
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def app():
    st.title("Model Evaluation")

    # Load trained model
    with open("../models/trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load test data
    data = pd.read_csv("../data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Make predictions
    y_pred = model.predict(X)

    # Compute metrics
    mae = mean_absolute_error(y, y_pred)
    mse = mean_squared_error(y, y_pred)
    r2 = r2_score(y, y_pred)

    st.write(f"MAE: {mae}")
    st.write(f"MSE: {mse}")
    st.write(f"R² Score: {r2}")

