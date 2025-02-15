import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def page_correlation_study():
    st.title("📊 Correlation Study")
    
    df = pd.read_csv("data/final_cleaned_train.csv")
    
    st.write("### Sample Data")
    st.write(df.head())

    st.write("### Correlation Heatmap")
    plt.figure(figsize=(10, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm")
    st.pyplot()

page_correlation_study()
