import streamlit as st
import pandas as pd
import pickle

def app():
    st.title("Prediction Page")

    # Load trained model
    with open("models/trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Get feature names
    data = pd.read_csv("data/processed_train.csv")
    features = data.drop(columns=["SalePrice"]).columns.tolist()

    # User input
    user_input = {}
    for feature in features[:10]:  # Limiting to first 10 features for simplicity
        user_input[feature] = st.number_input(f"{feature}", value=0)

    # Predict button
    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        predicted_price = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${predicted_price:,.2f}")
