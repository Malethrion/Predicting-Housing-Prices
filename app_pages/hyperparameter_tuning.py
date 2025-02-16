import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

def app():
    st.title("Hyperparameter Tuning")

    # Load dataset
    data = pd.read_csv("../data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Define grid search parameters
    param_grid = {"n_estimators": [50, 100, 200], "max_depth": [10, 20, 30]}

    # Perform grid search
    model = RandomForestRegressor(random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=3)
    grid_search.fit(X, y)

    st.write(f"Best parameters: {grid_search.best_params_}")
