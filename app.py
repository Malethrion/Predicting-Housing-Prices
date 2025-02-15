import streamlit as st
from app_pages import home_page, correlation_study, data_cleaning, feature_engineering, model_training, model_evaluation, hyperparameter_tuning, feature_importance, final_model, deployment, prediction_page

# Set up page configuration
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Define pages in the app
PAGES = {
    "01 - Home": home_page,
    "02 - Correlation Study": correlation_study,
    "03 - Data Cleaning": data_cleaning,
    "04 - Feature Engineering": feature_engineering,
    "05 - Model Training": model_training,
    "06 - Model Evaluation": model_evaluation,
    "07 - Hyperparameter Tuning": hyperparameter_tuning,
    "08 - Feature Importance": feature_importance,
    "09 - Final Model": final_model,
    "10 - Deployment": deployment,
    "11 - Prediction": prediction_page
}

# Sidebar navigation
st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

# Display selected page
PAGES[selection].run()
