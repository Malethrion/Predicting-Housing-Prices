import streamlit as st
import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor

def app():
    st.title("Hyperparameter Tuning")

    # Load dataset
    data = pd.read_csv("data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Define hyperparameter grid
    param_grid = {"n_estimators": [50, 100, 200]}

    # Grid search
    grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X, y)

    st.write(f"Best Parameters: {grid_search.best_params_}")
