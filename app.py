import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler

# Load the trained model and the preprocessing pipeline
model = joblib.load('model.pkl')
scaler = joblib.load('scale_features.pkl')  # Assuming you have a scaler for input features

# Streamlit UI
st.title("Predicting Housing Price App")

# User inputs for house features
sqft = st.number_input("Enter square footage of the house:", min_value=500, max_value=10000, step=10)
year_built = st.number_input("Enter year the house was built:", min_value=1900, max_value=2025, step=1)
garage_area = st.number_input("Enter garage area (sqft):", min_value=0, max_value=2000, step=10)
kitchen_quality = st.selectbox("Select kitchen quality:", ['Excellent', 'Good', 'Average', 'Fair', 'Poor'])

# Add more feature inputs as needed, e.g., 'YearRemodAdd', '2ndFlrSF', etc.

# Gather inputs into a DataFrame
user_data = pd.DataFrame({
    '1stFlrSF': [sqft], 
    'YearBuilt': [year_built],
    'GarageArea': [garage_area],
    'KitchenQual': [kitchen_quality],
    # Add other features similarly
})

# Feature scaling and transformation
user_data_scaled = scaler.transform(user_data)

# Predict house price
prediction = model.predict(user_data_scaled)

# Display prediction result
st.write(f"Predicted House Price: ${prediction[0]:,.2f}")
