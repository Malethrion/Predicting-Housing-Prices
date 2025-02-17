import streamlit as st
import importlib

# Set the page configuration
st.set_page_config(page_title="Predicting Housing Prices", layout="wide")

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

# Update query params correctly
if "page" not in st.query_params or st.query_params["page"] != selected_page:
    st.query_params.update({"page": selected_page})

# Dynamically import and load the selected page
try:
    module = importlib.import_module(PAGES[selected_page])  # Import module dynamically
    if hasattr(module, "app"):
        module.app()  # Call the app function of the selected module
    else:
        st.error(f"Error: `{selected_page}` module is missing an `app()` function.")
except ModuleNotFoundError:
    st.error(f"Error: `{selected_page}` module not found. Please check your project structure.")

