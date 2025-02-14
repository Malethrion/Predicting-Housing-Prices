import pandas as pd
from sklearn.impute import SimpleImputer

# Load dataset
df = pd.read_csv("../data/train.csv")

# Identify missing valuesfrom sklearn.impute import SimpleImputer
import pandas as pd

# Load data
data = pd.read_csv("../ata/train.csv")

# Impute missing values for numerical columns
numerical_columns = ['1stFlrSF', 'GarageArea', 'YearBuilt']  # Example columns
imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = imputer.fit_transform(data[numerical_columns])

# Save the cleaned data
data.to_csv("data/final_cleaned_train.csv", index=False)

missing_values = df.isnull().sum()
missing_columns = missing_values[missing_values > 0].index

# Separate numerical and categorical columns
numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
categorical_cols = df.select_dtypes(include=["object"]).columns

# Ensure all numerical columns contain only numeric data
for col in numerical_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")  # Convert invalid values to NaN

# Handle missing numerical values (replace with median)
num_imputer = SimpleImputer(strategy="median")
df[numerical_cols] = num_imputer.fit_transform(df[numerical_cols])

# Handle missing categorical values (replace with most frequent)
cat_imputer = SimpleImputer(strategy="most_frequent")
df[categorical_cols] = cat_imputer.fit_transform(df[categorical_cols])

# Save cleaned dataset
df.to_csv("../data/cleaned_train.csv", index=False)

print("Missing values handled and dataset saved successfully.")
