import streamlit as st
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

def app():
    st.title("Feature Engineering")

    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        st.error(f"File not found: {data_path}. Please run the data cleaning step first.")
        return

    # Load dataset
    data = pd.read_csv(data_path)

    # Identify numerical and categorical columns
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = data.select_dtypes(include=['object']).columns.tolist()

    # Define transformations
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    # Apply transformations
    transformed_data = preprocessor.fit_transform(data)

    # Convert to DataFrame
    feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    processed_data = pd.DataFrame(transformed_data, columns=feature_names)

    # Save processed data
    processed_data_path = "data/processed_train.csv"
    processed_data.to_csv(processed_data_path, index=False)

    st.write("### Feature Engineering Completed")
    st.write(f"Processed data saved as `{processed_data_path}`.")
    st.write("Preview of transformed features:")
    st.write(processed_data.head())

