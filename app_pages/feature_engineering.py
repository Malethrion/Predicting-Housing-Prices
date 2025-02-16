import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def app():
    st.title("Feature Engineering")

    # Load cleaned data
    data = pd.read_csv("data/final_cleaned_train.csv")

    # Encode categorical features
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = label_encoder.fit_transform(data[col])

    # Scale numerical features
    scaler = StandardScaler()
    data[data.select_dtypes(include=["number"]).columns] = scaler.fit_transform(data.select_dtypes(include=["number"]))

    data.to_csv("data/processed_train.csv", index=False)
    st.write("Feature engineering completed. Saved to 'processed_train.csv'.")
