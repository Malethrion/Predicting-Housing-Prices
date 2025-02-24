import streamlit as st
import optuna
import xgboost as xgb
import pandas as pd
import numpy as np
import os
import pickle  # Add this line
from sklearn.model_selection import cross_val_score

def app():
    """Optimize XGBoost hyperparameters using Optuna and display results in Streamlit."""
    st.title("🔧 Hyperparameter Tuning with Optuna")

    data_path = "data/processed_train.csv"

    if not os.path.exists(data_path):
        st.error(f"❌ File not found: `{data_path}`. Run feature engineering first.")
        return

    st.success(f"✅ Data file found: `{data_path}`")

    # Load dataset
    data = pd.read_csv(data_path)
    X = data.drop(columns=["SalePrice"])
    y = data["SalePrice"]  # Log-transformed target

    def objective(trial):
        """Optimize XGBoost hyperparameters for house price prediction."""
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
            "max_depth": trial.suggest_int("max_depth", 3, 15),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0, 5),
            "random_state": 42,
            "enable_categorical": False  # Explicitly disable to avoid FutureWarnings
        }
        
        model = xgb.XGBRegressor(**params)
        scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
        return -np.mean(scores)  # Minimize RMSE

    # Run optimization with progress bar
    st.write("Optimizing hyperparameters with Optuna...")
    progress_bar = st.progress(0)
    study = optuna.create_study(direction="minimize")

    def on_trial_end(study, trial):
        progress_bar.progress((trial.number + 1) / 50)  # Update progress for 50 trials

    study.optimize(objective, n_trials=50, callbacks=[on_trial_end])

    # Display results
    best_params = study.best_params
    st.write("### Best Hyperparameters:")
    st.json(best_params)

    best_rmse = -study.best_value
    st.write(f"### Best Cross-Validation RMSE (Log-Transformed): {best_rmse:.4f}")

    # Save best parameters and optionally train a final model
    os.makedirs("models", exist_ok=True)
    with open("models/best_params.pkl", "wb") as f:
        pickle.dump(best_params, f)

    final_model = xgb.XGBRegressor(**best_params, enable_categorical=False)
    final_model.fit(X, y)
    
    with open("models/optimized_model.pkl", "wb") as f:
        pickle.dump(final_model, f)

    st.success("✅ Hyperparameters optimized and model saved with best parameters!")

if __name__ == "__main__":
    app()