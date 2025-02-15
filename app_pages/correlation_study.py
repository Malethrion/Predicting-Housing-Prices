import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

st.title("📊 Correlation Study")

# Load the dataset
data = pd.read_csv("./data/train.csv")

# Display sample data
st.subheader("Sample Data")
st.write(data.head())

# Handle categorical variables before correlation computation
st.subheader("Correlation Heatmap")

# Convert categorical variables to numerical using label encoding
categorical_cols = data.select_dtypes(include=['object']).columns
data_encoded = data.copy()

for col in categorical_cols:
    data_encoded[col] = data_encoded[col].astype('category').cat.codes

# Compute correlation
corr_matrix = data_encoded.corr()

# Plot heatmap
fig, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=False, cmap="coolwarm", ax=ax)
st.pyplot(fig)
