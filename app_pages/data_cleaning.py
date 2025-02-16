import streamlit as st
import pandas as pd
import os

def app():
    st.title("Data Cleaning")

    data_path = "data/train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please make sure the dataset is available.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Save cleaned dataset
    cleaned_data_path = "data/final_cleaned_train.csv"
    data.to_csv(cleaned_data_path, index=False)

    st.write("### Data Cleaning Completed")
    st.write(f"Saved as `{cleaned_data_path}`.")
    st.write("Summary statistics after cleaning:")
    st.write(data.describe())

