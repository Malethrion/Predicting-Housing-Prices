import streamlit as st
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load model and necessary data
with open("../models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("../models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

with open("../models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load test data
test_data = pd.read_csv("../data/processed_train.csv")

# Define target variable
target = "SalePrice"

# Ensure no NaN values in the dataset
test_data = test_data.dropna()

# Separate features and target
X_test = test_data.drop(columns=[target])
y_actual = test_data[target]

# Ensure feature alignment
X_test = X_test[feature_names]

# Apply scaler transformation if necessary
X_test_scaled = scaler.transform(X_test)

# Make predictions
y_pred = model.predict(X_test_scaled)

# Ensure no NaN values exist before computing metrics
if np.isnan(y_actual).sum() == 0 and np.isnan(y_pred).sum() == 0:
    mae = mean_absolute_error(y_actual, y_pred)
    mse = mean_squared_error(y_actual, y_pred)
    r2 = r2_score(y_actual, y_pred)
else:
    mae, mse, r2 = "Error: NaN values detected in evaluation", "Error", "Error"

# Streamlit UI
st.title("Model Evaluation")
st.write("Evaluation metrics for the trained model.")

st.metric("Mean Absolute Error (MAE)", mae)
st.metric("Mean Squared Error (MSE)", mse)
st.metric("R-squared Score", r2)

