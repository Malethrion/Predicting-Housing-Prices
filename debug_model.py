import pickle
import pandas as pd
import numpy as np

# Load model, scaler, and feature names
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Load processed data
data_path = "data/processed_train.csv"
data = pd.read_csv(data_path)

# Ensure features match model input
X = data[feature_names]
y_log = np.log1p(data["SalePrice"])  # Log-transformed SalePrice for training

# Scale features
X_scaled = scaler.transform(X)

# Predict log-transformed price
log_predictions = model.predict(X_scaled)

# Reverse the log transformation
predictions = np.expm1(log_predictions)

# Debugging Output
print(f"🔍 Debug: Log Predictions: {log_predictions[:5]}")
print(f"🔄 Debug: Final Predictions: {predictions[:5]}")
