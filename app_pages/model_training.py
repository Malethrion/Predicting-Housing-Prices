import streamlit as st
import pandas as pd
import os
import numpy as np
import pickle
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

def train_model():
    st.title("🔧 Model Training with XGBoost")
    st.write("Training the model using **XGBoost**.")

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

    y = data[target]  # Log-transformed target (already transformed in feature engineering)
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

    # Define and train XGBoost model
    model = xgb.XGBRegressor(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], early_stopping_rounds=50, verbose=True)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    st.write(f"📉 **Model RMSE:** {rmse:,.2f}")

    # Save trained model
    os.makedirs("models", exist_ok=True)
    with open("models/trained_model.pkl", "wb") as f:
        pickle.dump(model, f)

    st.success("✅ **XGBoost Model Trained & Saved!**")

def app():
    train_model()

if __name__ == "__main__":
    app()