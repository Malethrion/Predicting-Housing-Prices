import pandas as pd
import pickle
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import os


def train_model():
    """Train an XGBoost model with optimized hyperparameters and save it."""
    data_path = "data/processed_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"File not found: {data_path}. Run feature engineering first."
        )

    best_params_path = "models/best_params.pkl"
    if not os.path.exists(best_params_path):
        raise FileNotFoundError(
            f"Best parameters not found at `{best_params_path}`. Run hyperparameter "
            "tuning first."
        )

    print(f"Data file found: {data_path}")

    # Load the processed dataset
    data = pd.read_csv(data_path)

    # Ensure SalePrice is present for training
    target = "SalePrice"
    if target not in data.columns:
        raise KeyError(
            "`SalePrice` column is missing. Ensure processed_train.csv includes "
            "target values for training."
        )

    y = data[target]  # Log-transformed target
    X = data.drop(columns=[target])

    # Load feature names to ensure correct column order
    feature_names_path = "models/feature_names.pkl"
    if not os.path.exists(feature_names_path):
        raise FileNotFoundError(
            f"Feature names file not found at `{feature_names_path}`. Run feature "
            "engineering first."
        )

    with open(feature_names_path, "rb") as f:
        expected_features = pickle.load(f)

    # Ensure X columns match the expected features from training
    X = X[expected_features]  # Reorder and filter columns to match training

    # Debug: Check columns in X before training (hypothetical long line causing E501)
    print("Features in X Before Training:", X.columns.tolist())

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print(f"Training Samples: {X_train.shape[0]} | Testing Samples: {X_test.shape[0]}")

    # Load best parameters from hyperparameter tuning
    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)

    # Add early_stopping_rounds to the parameters
    best_params["early_stopping_rounds"] = 50  # Set early stopping rounds here
    best_params["eval_metric"] = "rmse"  # Specify evaluation metric for early stopping

    # Define and train XGBoost model with optimized parameters
    model = xgb.XGBRegressor(**best_params)
    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)

    # Model evaluation
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"Model RMSE on Log-Transformed Prices: {rmse:.4f}")

    # Save trained model as optimized_model.pkl
    os.makedirs("models", exist_ok=True)
    with open("models/optimized_model.pkl", "wb") as f:
        pickle.dump(model, f)

    print("XGBoost Model Trained & Saved with Optimized Parameters!")


if __name__ == "__main__":
    train_model()
