import streamlit as st
import os
import pickle

def app():
    st.title("Deployment Steps")

    st.write("Deploy the final trained model and test its performance in production.")

    if st.button("Deploy Model"):
        os.makedirs("../models", exist_ok=True)

        # Load trained model
        with open("../models/trained_model.pkl", "rb") as file:
            model = pickle.load(file)

        # Save again as a deployment model
        with open("../models/deployed_model.pkl", "wb") as file:
            pickle.dump(model, file)

        st.success("Model successfully deployed!")
