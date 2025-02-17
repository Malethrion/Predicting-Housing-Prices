import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load feature names
with open("models/feature_names.pkl", "rb") as f:
    feature_names = pickle.load(f)

# Load scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Create a test input (update values if needed)
test_input = {
    "GrLivArea": 2000,
    "OverallQual": 7,
    "GarageCars": 2,
    "YearBuilt": 2005,
    "TotalBsmtSF": 1200,
}

# Convert test input to DataFrame
test_features_df = pd.DataFrame([test_input])

# Ensure all required columns are present
missing_cols = {col: 0 for col in feature_names if col not in test_features_df}
missing_df = pd.DataFrame([missing_cols])

# Combine user input with missing columns
test_features_df = pd.concat([test_features_df, missing_df], axis=1)

# Ensure correct column order
test_features_df = test_features_df[feature_names]

# Debug: Print transformed input features before scaling
print("🔍 Transformed Test Input Features (Before Scaling):")
print(test_features_df)

# Apply scaling
test_features_scaled = scaler.transform(test_features_df)

# Debug: Print transformed input features after scaling
print("🔍 Transformed Test Input Features (After Scaling):")
print(test_features_scaled)

# Make a prediction using the trained model
rf_prediction = model.predict(test_features_scaled)

# Debug: Print raw log-transformed prediction
print(f"📊 RF Log1p Prediction: {rf_prediction}")

# Convert log1p back to normal scale
final_rf_price = np.expm1(rf_prediction)

# Debugging: Check if predicted price is reasonable
print(f"🏡 Final Predicted House Price (RF): {final_rf_price}")

# Add a warning if the prediction is out of range
if final_rf_price[0] < 50_000 or final_rf_price[0] > 1_000_000:
    print("⚠️ WARNING: Prediction is out of expected range! Check your data.")
