import streamlit as st
import pandas as pd
import pickle
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np

def app():
    st.title("Hyperparameter Tuning")

    # Load dataset
    data = pd.read_csv("data/processed_train.csv")
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]

    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [50, 100, 200, 300],
        "max_depth": [None, 10, 20, 30],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
    }

    # Randomized Search for efficiency
    random_search = RandomizedSearchCV(
        RandomForestRegressor(), param_distributions=param_grid, n_iter=10, cv=3, n_jobs=-1, random_state=42
    )
    random_search.fit(X, y)

    # Save best parameters
    best_params = random_search.best_params_
    with open("models/best_hyperparameters.pkl", "wb") as f:
        pickle.dump(best_params, f)

    st.write("### Best Hyperparameters")
    st.write(best_params)

if __name__ == "__main__":
    app()
