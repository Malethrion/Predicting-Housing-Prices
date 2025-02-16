import streamlit as st
import pandas as pd
import pickle

def app():
    st.title("Prediction Page")

    # Load trained model
    with open("models/trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load feature names
    with open("models/feature_names.pkl", "rb") as f:
        trained_features = pickle.load(f)

    # User input
    user_input = {feature: st.number_input(f"{feature}", value=0) for feature in trained_features}

    # Predict button
    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])

        # Ensure input matches trained model feature order
        input_df = input_df.reindex(columns=trained_features, fill_value=0)

        predicted_price = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${predicted_price:,.2f}")
