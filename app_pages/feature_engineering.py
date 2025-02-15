import streamlit as st
import pandas as pd
import numpy as np

st.title("Feature Engineering")

# Load the cleaned dataset
try:
    data = pd.read_csv("data/final_cleaned_train.csv")
    st.write("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("The cleaned dataset was not found. Please run the data cleaning step first.")
    st.stop()

# Feature Engineering Process
st.subheader("Feature Engineering Steps")

# Creating new features
st.write("Creating new features...")

# Example: Creating a new feature for Total Area (adding basement and living area)
if "TotalBsmtSF" in data.columns and "GrLivArea" in data.columns:
    data["TotalArea"] = data["TotalBsmtSF"] + data["GrLivArea"]
    st.write("✅ Added `TotalArea` feature")

# Example: Log transformation of SalePrice to normalize distribution
if "SalePrice" in data.columns:
    data["Log_SalePrice"] = np.log1p(data["SalePrice"])
    st.write("✅ Applied log transformation to `SalePrice`")

# Example: Encoding categorical variables
categorical_cols = data.select_dtypes(include=['object']).columns
data = pd.get_dummies(data, columns=categorical_cols, drop_first=True)
st.write("✅ Encoded categorical features using one-hot encoding")

# Save the transformed dataset
data.to_csv("data/processed_train.csv", index=False)
st.success("Feature Engineering completed! Processed dataset saved.")

# Display a sample of the processed data
st.subheader("Sample of Processed Data")
st.dataframe(data.head())
