import streamlit as st
import importlib

# Set the page configuration
st.set_page_config(page_title="Predicting Housing Prices", layout="wide")

# Define user-facing pages only (remove Data Cleaning, Feature Engineering, Model Training)
PAGES = {
    "Home": "app_pages.home_page",
    "Correlation Study": "app_pages.correlation_study",
    "House Price Prediction": "app_pages.prediction_page",  # Updated for consistency
    "Feature Importance": "app_pages.feature_importance",
    "Hyperparameter Tuning": "app_pages.hyperparameter_tuning",
}

# Sidebar navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.selectbox("Choose a page", list(PAGES.keys()))

# Update query parameters to prevent pages from disappearing
if "page" not in st.query_params or st.query_params["page"] != selected_page:
    st.query_params.update({"page": selected_page})

# Load the selected page dynamically
try:
    module = importlib.import_module(PAGES[selected_page])  # Import the module dynamically
    if hasattr(module, "app"):
        module.app()  # Call the app function of the selected module
    else:
        st.error(f"Error: `{selected_page}` module is missing an `app()` function.")
except ModuleNotFoundError:
    st.error(f"Error: `{selected_page}` module not found. Please check your project structure.")