import streamlit as st
import importlib

# Set the page configuration
st.set_page_config(page_title="Housing Price Prediction", layout="wide")

# Define available pages
PAGES = {
    "Home": "app_pages.home_page",
    "Correlation Study": "app_pages.correlation_study",
    "Data Cleaning": "app_pages.data_cleaning",
    "Feature Engineering": "app_pages.feature_engineering",
    "Model Training": "app_pages.model_training",
    "Model Evaluation": "app_pages.model_evaluation",
    "Hyperparameter Tuning": "app_pages.hyperparameter_tuning",
    "Feature Importance": "app_pages.feature_importance",
    "Final Model": "app_pages.final_model",
    "Deployment": "app_pages.deployment",
    "Prediction": "app_pages.prediction_page",
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choose a page", list(PAGES.keys()))

# Dynamically import and load the selected page
try:
    module = importlib.import_module(PAGES[selected_page])  # Import module dynamically
    module.app()  # Call the app function of the selected module
except ModuleNotFoundError as e:
    st.error(f"Error: {selected_page} module not found. Please check your project structure.")
except AttributeError:
    st.error(f"Error: {selected_page} module is missing an `app()` function.")
