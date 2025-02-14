from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import pandas as pd
import joblib

# Load your raw data
data = pd.read_csv("../data/train.csv")

# Split data into features and target
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Define the numerical and categorical features
numerical_features = ['1stFlrSF', 'GarageArea', 'YearBuilt']
categorical_features = ['KitchenQual', 'OverallQual']

# Numerical transformer
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Categorical transformer
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

# Combine transformers into one column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

# Final pipeline to include preprocessing
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor)
])

# Fit the pipeline on the data
X_processed = pipeline.fit_transform(X)

# Save the pipeline as a .pkl file
joblib.dump(pipeline, 'data_cleaning_pipeline.pkl')

# pd.DataFrame(X_processed).to_csv("data/processed_data.csv", index=False)
