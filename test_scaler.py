import pickle
import numpy as np

# Load the scaler
with open("models/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# Test with a sample input
sample_input = np.array([[1500, 5, 2, 2000, 1000]])  # Adjust feature order

try:
    scaled_data = scaler.transform(sample_input)
    print("✅ Scaler is working!")
    print(scaled_data)
except Exception as e:
    print(f"❌ Scaler error: {e}")
