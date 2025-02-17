import pickle
import numpy as np
import pandas as pd

# Load the trained model
with open("models/trained_model.pkl", "rb") as f:
    model = pickle.load(f)

# Create a test input with realistic values
test_features = pd.DataFrame([{
    "GrLivArea": 2000,
    "OverallQual": 7,
    "GarageCars": 2,
    "YearBuilt": 2005,
    "TotalBsmtSF": 1200,
}])

# Make a prediction
test_prediction = model.predict(test_features)

# Print raw log predictions
print(f"🔍 Raw Log Prediction: {test_prediction}")

# Convert from log scale back to normal price
final_price = np.exp(test_prediction)
print(f"🏡 Final Predicted House Price: {final_price}")
