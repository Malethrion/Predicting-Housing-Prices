import streamlit as st
import pickle
import pandas as pd

st.title("🏡 House Price Prediction")

# Load the trained model
@st.cache_data
def load_model():
    with open("./models/final_model.pkl", "rb") as file:
        model = pickle.load(file)
    return model

model = load_model()

# User input fields
st.subheader("Enter house details to predict price")

# Example input fields (customize as per your dataset)
overall_qual = st.number_input("Overall Quality", min_value=1, max_value=10, value=5)
gr_liv_area = st.number_input("Ground Living Area (sqft)", min_value=500, max_value=5000, value=1500)
garage_cars = st.number_input("Garage Cars", min_value=0, max_value=5, value=2)
total_bsmt_sf = st.number_input("Total Basement Area (sqft)", min_value=0, max_value=3000, value=800)

# Make prediction
if st.button("Predict Price"):
    input_data = pd.DataFrame([[overall_qual, gr_liv_area, garage_cars, total_bsmt_sf]],
                              columns=["OverallQual", "GrLivArea", "GarageCars", "TotalBsmtSF"])
    
    predicted_price = model.predict(input_data)
    st.success(f"🏠 Estimated House Price: ${predicted_price[0]:,.2f}")
