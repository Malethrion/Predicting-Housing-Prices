import streamlit as st
import home_page
import correlation_study
import data_cleaning
import feature_engineering
import model_evaluation
import prediction_page

# Setup page configuration
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

PAGES = {
    "Home": home_page,
    "Correlation Study": correlation_study,
    "Data Cleaning": data_cleaning,
    "Feature Engineering": feature_engineering,
    "Model Evaluation": model_evaluation,
    "Prediction Page": prediction_page
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))
PAGES[selection].run()
