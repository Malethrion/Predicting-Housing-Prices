import streamlit as st

def app():
    st.title("Home Page")
    st.write("Welcome to the Predicting Housing Prices App!")
    st.markdown("""
    ## Overview
    This app allows users to explore house price data, analyze trends, and predict house prices using a machine learning model.
    """)

    st.markdown("### Business Requirements")
    st.write("1. Understand the correlation between house features and price.")
    st.write("2. Predict house prices based on user inputs.")
