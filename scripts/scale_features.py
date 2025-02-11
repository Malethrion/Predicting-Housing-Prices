import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load encoded dataset
df = pd.read_csv("../data/encoded_train.csv")

# Identify numerical columns
numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Scale features
scaler = StandardScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Save scaled dataset
df.to_csv("../data/scaled_train.csv", index=False)

print("Numerical features scaled and dataset saved.")
