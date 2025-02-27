import pandas as pd
import os
import numpy as np
import pickle
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer


def engineer_features():
    """Perform feature engineering, including scaling and encoding, and save processed data."""
    data_path = "data/final_cleaned_train.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"File not found: {data_path}. Please run data cleaning first."
        )

    print(f"Data file found: {data_path}")

    # Load dataset
    data = pd.read_csv(data_path)
    print("Dataset Preview (First 5 rows):")
    print(data.head())

    # Identify features and target
    target = "SalePrice"
    if target not in data.columns:
        raise KeyError(
            f"`{target}` column is missing. Ensure `final_cleaned_train.csv` "
            "includes target values."
        )

    y = np.log1p(data[target])  # Log-transform SalePrice for better model performance
    X = data.drop(columns=[target])

    numerical_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_features = X.select_dtypes(include=['object']).columns.tolist()

    print(f"Numeric Features: {len(numerical_features)} | Categorical Features: {len(categorical_features)}")

    # Define and fit transformations using ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', MinMaxScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
             categorical_features)
        ]
    )

    # Fit and transform the data
    try:
        transformed_data = preprocessor.fit_transform(X)
    except Exception as e:
        raise Exception(f"Error in data transformation: {e}")

    # Extract feature names
    try:
        encoded_feature_names = preprocessor.named_transformers_['cat'].get_feature_names_out(
            categorical_features
        )
        feature_names = list(numerical_features) + list(encoded_feature_names)
    except Exception as e:
        raise Exception(f"Error extracting feature names: {e}")

    print("Feature transformation successful!")

    # Convert to DataFrame and add log-transformed SalePrice
    processed_data = pd.DataFrame(transformed_data, columns=feature_names)
    processed_data["SalePrice"] = y  # Restore log-transformed target for training

    # Save processed data
    processed_data_path = "data/processed_train.csv"
    os.makedirs("data", exist_ok=True)
    processed_data.to_csv(processed_data_path, index=False)

    print("Feature Engineering Completed!")
    print(f"Processed data saved at `{processed_data_path}`")
    print("Transformed Data Preview:")
    print(processed_data.head())

    # Save the preprocessor and feature names
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)

    with open(f"{model_dir}/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    with open(f"{model_dir}/feature_names.pkl", "wb") as f:
        pickle.dump(feature_names, f)

    print("Preprocessor and Feature Names Saved!")


if __name__ == "__main__":
    engineer_features()
