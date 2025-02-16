import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title("Correlation Study")
    st.write("This section analyzes correlations between housing features and sale price.")

    # Load dataset
    data = pd.read_csv("data/train.csv")

    # Compute correlation matrix
    correlation_matrix = data.corr()

    # Display heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
    st.pyplot()
