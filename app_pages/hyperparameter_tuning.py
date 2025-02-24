import streamlit as st
import pickle
import os

def app():
    """Display results of hyperparameter tuning from saved parameters."""
    st.title("Hyperparameter Tuning with Optuna")

    best_params_path = "models/best_params.pkl"
    best_rmse_path = "models/best_rmse.pkl"
    if not os.path.exists(best_params_path) or not os.path.exists(best_rmse_path):
        st.error("Best parameters or RMSE not found. Run hyperparameter tuning offline first.")
        return

    with open(best_params_path, "rb") as f:
        best_params = pickle.load(f)
    with open(best_rmse_path, "rb") as f:
        best_rmse = pickle.load(f)

    st.write("Best Hyperparameters:")
    st.json(best_params)
    st.write(f"Best Cross-Validation RMSE (Log-Transformed): {best_rmse:.4f}")

    st.write("Hyperparameters optimized and model saved with best parameters!")

if __name__ == "__main__":
    app()