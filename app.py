import streamlit as st
from app_pages import (
    home_page, correlation_study, data_cleaning, 
    feature_engineering, model_training, model_evaluation,
    hyperparameter_tuning, feature_importance, final_model, deployment, prediction_page
)

# Set page configuration
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Sidebar navigation
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
    "Prediction": prediction_page,
}

st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choose a page", list(PAGES.keys()))

# Load selected page
page = PAGES[selected_page]
page.app()  # Call the app function of the selected module
