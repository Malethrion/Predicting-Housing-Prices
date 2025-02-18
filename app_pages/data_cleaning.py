import streamlit as st
import pandas as pd
import os

def app():
    """Perform data cleaning and save the cleaned dataset."""
    st.title("Data Cleaning")

    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please make sure the dataset is available.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Identify numerical and categorical columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Handle missing values
    for col in numeric_columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].median(), inplace=True)

    for col in categorical_columns:
        if data[col].isnull().sum() > 0:
            data[col].fillna(data[col].mode()[0], inplace=True)

    # Ensure SalePrice has no zero or negative values
    if "SalePrice" in data.columns:
        data = data[data["SalePrice"] > 0]

    # Save cleaned dataset
    cleaned_data_path = "data/final_cleaned_train.csv"
    os.makedirs("data", exist_ok=True)  # Ensure 'data' directory exists
    data.to_csv(cleaned_data_path, index=False)

    # Streamlit messages
    st.write("### Data Cleaning Completed")
    st.write(f"Saved as `{cleaned_data_path}`.")
    st.write("Summary statistics after cleaning:")
    st.write(data.describe())

if __name__ == "__main__":
    app()
