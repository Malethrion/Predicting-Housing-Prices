import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

def app():
    st.title("Correlation Study")

    # Check if dataset exists
    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run the data cleaning process first.")
        return
    
    # Load dataset
    data = pd.read_csv(data_path)

    # Compute correlation matrix
    correlation_matrix = data.corr()

    # Plot heatmap
    st.write("### Correlation Heatmap")
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    st.pyplot()

