import streamlit as st
import pickle
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load trained model and scaler
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load test data
data = pd.read_csv("data/processed_train.csv")
target = "SalePrice"
X = data.drop(columns=[target])
y = np.log(data[target])  # Apply log transformation to match training

# Scale test features
X_scaled = scaler.transform(X)

# Get predictions
y_pred = model.predict(X_scaled)

# Convert predictions back to normal scale
y_pred = np.exp(y_pred)
y_actual = np.exp(y)  # Convert actual values back

# Calculate evaluation metrics
mae = mean_absolute_error(y_actual, y_pred)
mse = mean_squared_error(y_actual, y_pred)
r2 = r2_score(y_actual, y_pred)

def app():
    """Streamlit UI for model evaluation."""
    st.title("Model Evaluation")
    st.write("### Model Performance Metrics")

    # Displaying the metrics with appropriate formatting
    st.metric(label="Mean Absolute Error (MAE)", value=f"{mae:,.2f}")
    st.metric(label="Mean Squared Error (MSE)", value=f"{mse:,.2f}")
    st.metric(label="R-squared Score (R²)", value=f"{r2:.4f}")

    # Dataframe for actual vs predicted values
    st.write("### Predictions vs. Actual")
    comparison_df = pd.DataFrame({"Actual Price": y_actual, "Predicted Price": y_pred})
    st.dataframe(comparison_df.sample(10))  # Display a sample of 10

    # Display a warning if the model is performing poorly
    if r2 < 0:
        st.warning("The model performance is low. Consider improving feature selection or transformations.")

