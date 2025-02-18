import streamlit as st
import os
import sys

# Ensure the app_pages directory is in the Python path
sys.path.append(os.path.dirname(__file__))

from app_pages import prediction_page, home_page

dimport streamlit as st
import os
import sys

# Ensure the app_pages directory is in the Python path
sys.path.append(os.path.dirname(__file__))

from app_pages import home_page, prediction_page, eda_page, feature_importance_page, model_evaluation_page

import streamlit as st
import os
import sys

# Ensure the app_pages directory is in the Python path
sys.path.append(os.path.dirname(__file__))

from app_pages import home_page, prediction_page, eda_page, feature_importance_page, model_evaluation_page

def main():
    st.set_page_config(page_title="House Price Prediction", page_icon="🏡")
    
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Go to", ["Home", "Predict Price", "EDA", "Feature Importance", "Model Evaluation"])
    
    if page == "Home":
        home_page.app()
    elif page == "Predict Price":
        prediction_page.app()
    elif page == "EDA":
        eda_page.app()
    elif page == "Feature Importance":
        feature_importance_page.app()
    elif page == "Model Evaluation":
        model_evaluation_page.app()

if __name__ == "__main__":
    main()
