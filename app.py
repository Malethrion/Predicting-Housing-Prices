import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Streamlit page title
st.title('House Price Prediction Dashboard')
st.markdown('### Enter house details below to predict its price')

# Input fields for user
sqft = st.number_input('Square Footage (sqft)', min_value=0)
bedrooms = st.number_input('Number of Bedrooms', min_value=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=1)

# Button to trigger prediction
if st.button('Predict Price'):
    # Convert inputs into a suitable format for the model
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    # Display the predicted price
    st.markdown(f"### Predicted House Price: ${predicted_price[0]:,.2f}")

    # Feature Importance Plot (if applicable)
    if hasattr(model, 'coef_'):
        feature_names = ['Square Footage', 'Bedrooms', 'Bathrooms']  # Modify as per your model
        importance = model.coef_
        
        fig, ax = plt.subplots()
        ax.barh(feature_names, importance)
        ax.set_xlabel('Importance')
        ax.set_title('Feature Importance')
        st.pyplot(fig)
    
    # Model Performance Visualization (Actual vs Predicted Prices)
    # Example (replace with actual model predictions)
    actual_prices = [100000, 200000, 300000]  # Replace with your actual data
    predicted_prices = [110000, 190000, 280000]  # Replace with predicted data
    
    fig = px.scatter(x=actual_prices, y=predicted_prices, labels={'x': 'Actual Prices', 'y': 'Predicted Prices'}, title="Actual vs Predicted Prices")
    st.plotly_chart(fig)
    
    # Interpretation of Feature Importance
    st.markdown("""
    ### Feature Importance
    The plot above shows the relative importance of each feature in predicting house prices. A higher bar indicates a more significant contribution to the predicted price.
    """)
    
    # Prediction Explanation
    st.markdown("""
    ### Prediction Explanation
    The model predicts the price of the house based on the features you provided. It uses historical data to estimate how factors like square footage, number of bedrooms, and bathrooms affect the price.
    """)

