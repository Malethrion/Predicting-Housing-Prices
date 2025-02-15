import streamlit as st
import pandas as pd

def page_data_cleaning():
    st.title("🧼 Data Cleaning")
    
    df = pd.read_csv("data/final_cleaned_train.csv")
    
    st.write("### Cleaned Data Sample")
    st.write(df.head())

page_data_cleaning()
