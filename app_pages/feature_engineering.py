import streamlit as st
import pandas as pd

def app():
    st.title("Feature Engineering")

    # Load cleaned data
    data = pd.read_csv("../data/final_cleaned_train.csv")

    # Example transformation: Creating new feature
    data["TotalSF"] = data["TotalBsmtSF"] + data["1stFlrSF"] + data["2ndFlrSF"]

    # Save processed data
    data.to_csv("../data/processed_train.csv", index=False)

    st.write("Feature Engineering Complete and Saved.")
