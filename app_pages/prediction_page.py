import streamlit as st
import pandas as pd
import pickle

def app():
    st.title("Prediction Page")

    # Load trained model
    with open("models/trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load processed data to get feature names
    data = pd.read_csv("data/processed_train.csv")
    features = data.drop(columns=["SalePrice"]).columns.tolist()

    # Select key features for user input
    selected_features = ["GrLivArea", "OverallQual", "GarageCars", "YearBuilt", "TotalBsmtSF"]

    st.write("### Enter House Features")

    user_input = {}
    for feature in selected_features:
        user_input[feature] = st.number_input(f"{feature}", value=float(data[feature].median()))

    # Predict button
    if st.button("Predict Price"):
        input_df = pd.DataFrame([user_input])
        predicted_price = model.predict(input_df)[0]
        st.success(f"Predicted House Price: ${predicted_price:,.2f}")

