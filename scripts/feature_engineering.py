import pandas as pd

# Load cleaned data
data = pd.read_csv("data/final_cleaned_train.csv")

# Create new feature 'PricePerSqft'
data['PricePerSqft'] = data['SalePrice'] / data['GrLivArea']

# Save the updated data with the new feature
data.to_csv("data/feature_engineered_train.csv", index=False)
