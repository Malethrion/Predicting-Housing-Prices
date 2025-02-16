import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def app():
    st.title("Correlation Study")

    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run the data cleaning process first.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Select only numeric columns for correlation
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()

    # Plot heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    st.pyplot()

