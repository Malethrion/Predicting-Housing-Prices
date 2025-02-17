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

# Avoid reloading the same module multiple times
if "current_page" not in st.session_state or st.session_state.current_page != selected_page:
    st.session_state.current_page = selected_page  # Store the selected page in session state

    # Dynamically import and load the selected page
    try:
        module = importlib.import_module(PAGES[selected_page])  # Import module dynamically
        if hasattr(module, "app"):
            st.experimental_set_query_params(page=selected_page)  # Avoid duplicate calls
            module.app()  # Call the app function of the selected module
        else:
            st.error(f"Error: `{selected_page}` module is missing an `app()` function.")
    except ModuleNotFoundError:
        st.error(f"Error: `{selected_page}` module not found. Please check your project structure.")
