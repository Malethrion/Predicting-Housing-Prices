import streamlit as st
import pandas as pd
import os

def app():
    """Perform data cleaning and save the cleaned dataset."""
    st.title("Data Cleaning")

    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please make sure the dataset is available.")
        print(f"ERROR: File not found: {data_path}")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Identify numerical and categorical columns
    numeric_columns = data.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = data.select_dtypes(include=['object']).columns

    # Fill missing values separately
    data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())

    if not categorical_columns.empty:
        data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])

    # Save cleaned dataset
    cleaned_data_path = "data/final_cleaned_train.csv"
    os.makedirs("data", exist_ok=True)  # Ensure 'data' directory exists
    data.to_csv(cleaned_data_path, index=False)

    # Streamlit messages
    st.write("### Data Cleaning Completed")
    st.write(f"Saved as `{cleaned_data_path}`.")
    st.write("Summary statistics after cleaning:")
    st.write(data.describe())

    # Console output (for debugging outside Streamlit)
    print("Data Cleaning Completed")
    print(f"Saved cleaned data to: {cleaned_data_path}")
    print(data.describe())

# Add this to make it runnable as a script
if __name__ == "__main__":
    app()
