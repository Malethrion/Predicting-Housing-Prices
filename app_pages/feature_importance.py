import streamlit as st

def app():
st.title("Feature Importance")

st.write("Analyze which features contribute the most to the prediction.")

# Example: Display feature importance methods
st.subheader("Methods Used")
st.write("- SHAP Values")
st.write("- Permutation Importance")
st.write("- Coefficients from Linear Models")

# Example: Placeholder for feature importance results
if st.button("Compute Feature Importance"):
    st.write("Computing feature importance... (Placeholder for actual analysis)")
