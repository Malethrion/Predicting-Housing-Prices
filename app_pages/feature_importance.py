import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import pickle

def app():
    st.title("Feature Importance")

    # Load model
    with open("../models/trained_model.pkl", "rb") as file:
        model = pickle.load(file)

    # Load dataset
    data = pd.read_csv("../data/processed_train.csv")
    feature_names = data.drop(columns=["SalePrice"]).columns

    # Plot importance
    plt.figure(figsize=(10, 5))
    plt.barh(feature_names, model.feature_importances_)
    st.pyplot()
