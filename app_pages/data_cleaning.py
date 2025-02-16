import streamlit as st
import pandas as pd

def app():
    st.title("Data Cleaning")

    # Load dataset
    data = pd.read_csv("../data/train.csv")

    # Handle missing values
    data.fillna(data.mean(), inplace=True)

    # Save cleaned dataset
    data.to_csv("../data/final_cleaned_train.csv", index=False)

    st.write("Data Cleaning Completed. Saved as `final_cleaned_train.csv`.")
