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

# Check for invalid or zero values before log transformation
y = y[y > 0]  # Remove zero or negative SalePrice
X = X.loc[y.index]  # Ensure X and y have matching indices

# Apply log transformation safely
y_log = np.log(y)

# Scale test features
X_scaled = scaler.transform(X)

# Get predictions
y_pred_log = model.predict(X_scaled)
y_pred = np.exp(y_pred_log)  # Convert log price back to normal scale

y_actual = np.exp(y_log)  # Convert actual values back

# Ensure no NaN values before evaluation
if np.isnan(y_actual).sum() > 0 or np.isnan(y_pred).sum() > 0:
    st.error("Error: NaN values detected in evaluation. Check data processing.")
else:
    # Calculate evaluation metrics
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)

    # Streamlit UI
    def app():
        st.title("Model Evaluation")
        st.write("### Evaluation metrics for the trained model.")

        st.write(f"**Mean Absolute Error (MAE):** {mae:,.2f}")
        st.write(f"**Mean Squared Error (MSE):** {mse:,.2f}")
        st.write(f"**R-squared Score:** {r2:.4f}")

        st.write("### Predictions vs. Actual")
        comparison_df = pd.DataFrame({"Actual Price": y_actual, "Predicted Price": y_pred})
        st.write(comparison_df.sample(10))  # Show random 10 samples

