import pandas as pd
import numpy as np
import os
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler

# Load the processed dataset
data = pd.read_csv("../data/processed_train.csv")

# Define target variable
target = "SalePrice"

# Debugging: Check for negative or zero values before transformation
print(f"Total rows before filtering: {len(data)}")
print(f"Rows with SalePrice <= 0: {len(data[data[target] <= 0])}")
print(f"Rows with NaN in SalePrice: {data[target].isnull().sum()}")

# Remove rows with zero or negative SalePrice
data = data[data[target] > 0]

# Apply log transformation
data[target] = np.log(data[target])

# Drop any remaining NaN values from the dataset
data = data.dropna()

# Debugging: Check again for any NaN values after transformation
print(f"Total rows after filtering: {len(data)}")
print(f"Missing values in SalePrice after dropna: {data[target].isnull().sum()}")

# Separate features (X) and target (y)
X = data.drop(columns=[target])
y = data[target]

# Verify there are no NaN values before splitting
print(f"Missing values in X: {X.isnull().sum().sum()}")
print(f"Missing values in y: {y.isnull().sum()}")

# Split dataset into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Final check before training
print(f"Missing values in y_train: {y_train.isnull().sum()}")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Ensure models directory exists
os.makedirs("../models", exist_ok=True)

# Save the trained model
with open("../models/trained_model.pkl", "wb") as f:
    pickle.dump(model, f)

# Save feature names
with open("../models/feature_names.pkl", "wb") as f:
    pickle.dump(X_train.columns.tolist(), f)

# Save the scaler
with open("../models/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Trained model, feature names, and scaler saved successfully!")
