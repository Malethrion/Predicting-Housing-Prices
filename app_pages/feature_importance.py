import streamlit as st
import pandas as pd
import pickle
import matplotlib.pyplot as plt

def app():
    st.title("Feature Importance")

    # Load model and feature names
    try:
        with open("models/optimized_model.pkl", "rb") as f:  # Changed from trained_model.pkl to optimized_model.pkl
            model = pickle.load(f)
        with open("models/feature_names.pkl", "rb") as f:
            feature_names = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading model or feature names: {e}")
        return

    # Ensure model has feature importance attribute
    if not hasattr(model, "feature_importances_"):
        st.error("The trained model does not support feature importance calculation.")
        return

    # Extract feature importance
    feature_importance = model.feature_importances_

    # Create DataFrame
    fi_df = pd.DataFrame({"Feature": feature_names, "Importance": feature_importance})
    fi_df = fi_df.sort_values(by="Importance", ascending=False)

    # Plot feature importance
    st.write("### Feature Importance Ranking")
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(fi_df["Feature"].head(20), fi_df["Importance"].head(20))
    ax.set_xlabel("Importance")
    ax.set_ylabel("Feature")
    ax.set_title("Top 20 Important Features")
    st.pyplot(fig)

if __name__ == "__main__":
    app()