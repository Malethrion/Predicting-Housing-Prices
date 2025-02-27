import pandas as pd
import os

def clean_data():
    """Perform data cleaning and save the cleaned dataset."""
    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"File not found: {data_path}. Please ensure the dataset is available in the 'data' directory.")

    # Load dataset
    data = pd.read_csv(data_path)

    # Identify numerical and categorical columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Handle missing values for numerical columns with median
    for col in numeric_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].median())

    # Handle missing values for categorical columns with mode
    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            data[col] = data[col].fillna(data[col].mode()[0])

    # Ensure SalePrice is positive (remove zero or negative values)
    if "SalePrice" in data.columns:
        data = data[data["SalePrice"] > 0]

    # Save cleaned dataset
    cleaned_data_path = "data/final_cleaned_train.csv"
    os.makedirs("data", exist_ok=True)  # Create 'data' directory if it doesn’t exist
    data.to_csv(cleaned_data_path, index=False)

    print("Data Cleaning Completed")
    print(f"Saved as `{cleaned_data_path}`.")
    print("Summary statistics after cleaning:")
    print(data.describe())

if __name__ == "__main__":
    clean_data()