import streamlit as st
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

model = pickle.load(open('model.pkl', 'rb'))

st.title('House Price Prediction Dashboard')
st.markdown('### Enter house details below to predict its price')

# Input fields for user
sqft = st.number_input('Square Footage (sqft)', min_value=0)
bedrooms = st.number_input('Number of Bedrooms', min_value=1)
bathrooms = st.number_input('Number of Bathrooms', min_value=1)

if st.button('Predict Price'):
    # Code for prediction will go here
    pass

if st.button('Predict Price'):
    # Convert inputs into a suitable format for the model
    input_data = np.array([[sqft, bedrooms, bathrooms]])
    
    # Make prediction
    predicted_price = model.predict(input_data)
    
    # Display the predicted price
    st.markdown(f"### Predicted House Price: ${predicted_price[0]:,.2f}")
