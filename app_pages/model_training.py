import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model():
    """Train an XGBoost model with optimized hyperparameters and save it for house price prediction."""
    st.title("🔧 Model Training with XGBoost")
    st.write("Training the model using **XGBoost** with optimized hyperparameters.")

    data_path = "data/processed_train.csv"

    if not os.path.exists(data_path):
        st.error(f"❌ File not found: `{data_path}`. Run feature engineering first.")
        return

    st.success(f"✅ Data file found: `{data_path}`")

    # Load the processed dataset
    data = pd.read_csv(data_path)

    # Ensure SalePrice is present for training
    target = "SalePrice"
    if target not in data.columns:
        st.error("❌ `SalePrice` column is missing. Ensure processed_train.csv includes target values for training.")
        return

    y = data[target]  # Log-transformed target
    X = data.drop(columns=[target])

    # Load feature names to ensure correct column order
    feature_names_path = "models/feature_names.pkl"
    if not os.path.exists(feature_names_path):
        st.error(f"❌ Feature names file not found at `{feature_names_path}`. Run feature engineering first.")
        return

    with open(feature_names_path, "rb") as f:
        expected_features = pickle.load(f)

    # Ensure X columns match the expected features from training
    X = X[expected_features]  # Reorder and filter columns to match training

    # Debug: Check columns in X before training
    st.write("🔍 **Features in X Before Training:**", X.columns.tolist())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    st.write(f"📊 **Training Samples:** {X_train.shape[0]} | **Testing Samples:** {X_test.shape[0]}")

    # Load best parameters from hyperparameter tuning
    best_params_path = "models/best_params.pkl"
    if not os.path.exists(best_params_path):
        st.warning("⚠️ Best parameters not found. Using default hyperparameters.")
        best_params = {
            "n_estimators": 600,
            "max_depth": 5,
            "learning_rate": 0.03,
            "subsample": 0.9,
            "colsample_bytree": 0.7,
            "min_child_weight": 3,
            "gamma": 0.1,
            "random_state": 42,
            "enable_categorical": False
        }
    else:
        with open(best_params_path, "rb") as f:
            best_params = pickle.load(f)

    # Define and train XGBoost model with optimized parameters
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=True)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"📉 **Model RMSE on Log-Transformed Prices:** {rmse:.4f}")

    # Save trained model as optimized_model.pkl
    os.makedirs("models", exist_ok=True)
    with open("models/optimized_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("✅ **XGBoost Model Trained & Saved with Optimized Parameters!**")

def app():
    train_model()

if __name__ == "__main__":
    app()