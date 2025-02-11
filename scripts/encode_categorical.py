import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load cleaned dataset
df = pd.read_csv("../data/cleaned_train.csv")

# Identify categorical columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Apply One-Hot Encoding
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Save encoded dataset
df.to_csv("../data/encoded_train.csv", index=False)

print("Categorical variables encoded and dataset saved.")
