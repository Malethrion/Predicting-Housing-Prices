import pandas as pd

# Load dataset
df = pd.read_csv("../data/final_cleaned_train.csv")

# Show first few rows
print("\nFirst 5 Rows:")
print(df.head())

# Show general dataset info
print("\nDataset Information:")
print(df.info())

# Check for missing values
print("\nMissing Values:")
print(df.isnull().sum().sum(), " missing values found.")
