import streamlit as st

# Set page configuration (must be the first Streamlit command)
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Import app pages
from app_pages import (
    home_page,
    correlation_study,
    data_cleaning,
    feature_engineering,
    model_training,
    model_evaluation,
    hyperparameter_tuning,
    feature_importance,
    final_model,
    deployment,
    prediction_page
)

# Define page titles for navigation
PAGES = {
    "Home": home_page,
    "Correlation Study": correlation_study,
    "Data Cleaning": data_cleaning,
    "Feature Engineering": feature_engineering,
    "Model Training": model_training,
    "Model Evaluation": model_evaluation,
    "Hyperparameter Tuning": hyperparameter_tuning,
    "Feature Importance": feature_importance,
    "Final Model": final_model,
    "Deployment": deployment,
    "Prediction": prediction_page
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Load the selected page
page = PAGES[selection]
page.app()  # Call the selected page's app function

# Footer
st.sidebar.markdown("---")
st.sidebar.text("Predicting Housing Prices")
st.sidebar.text("Developed with Streamlit")
