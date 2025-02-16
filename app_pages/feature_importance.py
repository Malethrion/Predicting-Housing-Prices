import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def app():
    st.title("Feature Importance")

    # Load model
    with open("models/trained_model.pkl", "rb") as f:
        model = pickle.load(f)

    # Get feature importance
    feature_importance = model.feature_importances_

    # Plot feature importance
    plt.figure(figsize=(10, 5))
    plt.barh(range(len(feature_importance)), feature_importance)
    plt.xlabel("Importance")
    plt.ylabel("Feature Index")
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(range(len(feature_importance)), feature_importance)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Importance")
    st.pyplot(fig)


