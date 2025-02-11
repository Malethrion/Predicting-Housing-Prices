import pandas as pd

# Load dataset
df = pd.read_csv("../data/train.csv")

# Basic information
print("Dataset Overview:")
print(df.info())  # Shows column data types and missing values

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())  # Shows numerical data distribution

# Check missing values
missing_values = df.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])  # Shows only columns with missing values
