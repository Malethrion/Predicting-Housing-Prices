import streamlit as st
import importlib

# Set the page configuration
st.set_page_config(page_title="Predicting Housing Prices", layout="wide")

# Define available pages
PAGES = {
    "Home": "app_pages.home_page",
    "Data Cleaning": "app_pages.data_cleaning",
    "Feature Engineering": "app_pages.feature_engineering",
    "Model Training": "app_pages.model_training",
    "Hyperparameter Tuning": "app_pages.hyperparameter_tuning",
    "Feature Importance": "app_pages.feature_importance",
    "Correlation Study": "app_pages.correlation_study",
    "Price Visualization": "app_pages.price_visualization",
    "Prediction": "app_pages.prediction_page",
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choose a page", list(PAGES.keys()))

# ✅ FIX 1: Update query params correctly (fix disappearing pages)
if "page" not in st.query_params or st.query_params["page"] != selected_page:
    st.query_params.update({"page": selected_page})

# ✅ FIX 2: Always load the selected page (force refresh)
try:
    module = importlib.import_module(PAGES[selected_page])  # Import module dynamically
    if hasattr(module, "app"):
        module.app()  # Call the app function of the selected module
    else:
        st.error(f"Error: `{selected_page}` module is missing an `app()` function.")
except ModuleNotFoundError:
    st.error(f"Error: `{selected_page}` module not found. Please check your project structure.")