import streamlit as st
import pandas as pd

def app():
    st.title("Data Cleaning")

    # Load dataset
    data = pd.read_csv("../data/train.csv")

    # Handle missing values
    numeric_cols = data.select_dtypes(include=['number']).columns
    categorical_cols = data.select_dtypes(include=['object']).columns

    data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())
    for col in categorical_cols:
        data[col] = data[col].fillna(data[col].mode()[0])

    # Save cleaned data
    data.to_csv("../data/final_cleaned_train.csv", index=False)

    st.write("Data Cleaning Complete and Saved.")
