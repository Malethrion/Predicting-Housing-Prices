import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

def app():
    st.title("Feature Engineering")

    # Load dataset
    data = pd.read_csv("../data/final_cleaned_train.csv")

    # Encode categorical variables
    for col in data.select_dtypes(include=['object']).columns:
        data[col] = LabelEncoder().fit_transform(data[col])

    # Scale numerical features
    scaler = StandardScaler()
    data[data.select_dtypes(include=['float64', 'int64']).columns] = scaler.fit_transform(
        data.select_dtypes(include=['float64', 'int64']))

    # Save processed dataset
    data.to_csv("../data/processed_train.csv", index=False)

    st.write("Feature Engineering Completed. Saved as `processed_train.csv`.")
