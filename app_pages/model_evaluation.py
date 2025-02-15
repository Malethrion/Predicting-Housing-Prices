import streamlit as st

st.title("Model Evaluation")

st.write("Evaluate the trained models and compare their performance.")

# Example: Display available evaluation metrics
st.subheader("Evaluation Metrics")
st.write("- Mean Absolute Error (MAE)")
st.write("- Root Mean Squared Error (RMSE)")
st.write("- R² Score")

# Example: Add a placeholder for results (replace with real evaluation)
if st.button("Run Evaluation"):
    st.write("Evaluating model... (Placeholder for actual evaluation results)")
