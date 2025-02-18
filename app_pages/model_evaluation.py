import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model and test data
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load test data
data = pd.read_csv("data/processed_train.csv")
target = "SalePrice"

# Drop NaN values before processing
data.dropna(subset=[target], inplace=True)
X = data.drop(columns=[target])
y = data[target]

# Scale test features
X_scaled = scaler.transform(X)

# Get predictions
y_pred = model.predict(X_scaled)

# Calculate evaluation metrics
mae = mean_absolute_error(y, y_pred)
mse = mean_squared_error(y, y_pred)
r2 = r2_score(y, y_pred)

# Save evaluation metrics
metrics = {"mae": mae, "mse": mse, "r2": r2}
with open("models/evaluation_metrics.pkl", "wb") as f:
    pickle.dump(metrics, f)

# Streamlit UI
def app():
    st.title("Model Evaluation")
    st.write("### Evaluation Metrics for the Trained Model")
    st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
    st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
    st.write(f"**R-squared Score:** {r2:.4f}")
    
    st.write("### Predictions vs. Actual")
    comparison_df = pd.DataFrame({"Actual Price": y, "Predicted Price": y_pred})
    st.write(comparison_df.sample(10))  # Show random 10 samples

if __name__ == "__main__":
    app()