import streamlit as st
import pickle

def app():
    st.title("Final Model Deployment")

    # Load final model
    with open("../models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    st.write("Final model loaded successfully.")
