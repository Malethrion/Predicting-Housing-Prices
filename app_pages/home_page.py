import streamlit as st

def page_home():
    st.title("🏠 Housing Price Prediction")
    st.write("Welcome to the Housing Price Prediction Dashboard.")
    st.markdown("### Project Overview")
    st.write("This project predicts housing prices using machine learning models based on various house features.")

    st.markdown("### Business Requirements")
    st.write("1. Understand the correlation between house features and price.")
    st.write("2. Predict house prices based on user inputs.")

page_home()
