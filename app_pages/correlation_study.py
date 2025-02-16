import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def app():
    st.title("Correlation Study")
    
    # Load dataset
    data = pd.read_csv("../data/final_cleaned_train.csv")

    # Compute correlation matrix
    correlation_matrix = data.corr()

    # Display heatmap
    st.markdown("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(correlation_matrix, annot=False, cmap="coolwarm")
    st.pyplot()
