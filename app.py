import streamlit as st
import os
import sys

# Ensure the app_pages directory is in the Python path
sys.path.append(os.path.dirname(__file__))

from app_pages import home_page, prediction_page, feature_importance, model_evaluation

def main():
    st.set_page_config(page_title="Predicting Housing Prices", page_icon="🏡")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict Price", "EDA", "Feature Importance", "Model Evaluation"])
    
    if page == "Home":
        home_page.app()
    elif page == "Predict Price":
        prediction_page.app()
    elif page == "Feature Importance":
        feature_importance.app()
    elif page == "Model Evaluation":
        model_evaluation.app()

if __name__ == "__main__":
    main()
