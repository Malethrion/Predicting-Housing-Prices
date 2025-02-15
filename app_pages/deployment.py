import streamlit as st
from app_pages import home_page, correlation_study, data_cleaning, feature_engineering, \
                      model_training, model_evaluation, hyperparameter_tuning, feature_importance, \
                      final_model, prediction_page  # Ensure prediction_page is now available

st.title("🚀 Deployment Steps")

st.write("""
Deploy the final model and test its performance in production.
""")

st.subheader("Deployment Steps")
st.markdown("""
1. Save the trained model  
2. Load the model in a web application  
3. Serve predictions via API  
""")

if st.button("Deploy Model"):
    st.success("✅ Model successfully deployed! You can now use the prediction page.")
