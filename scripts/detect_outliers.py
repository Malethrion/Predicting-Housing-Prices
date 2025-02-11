import pandas as pd

def remove_outliers(df, threshold=1.5):
    # Select only numerical columns
    num_cols = df.select_dtypes(include=["int64", "float64"]).columns
    df_numeric = df[num_cols]  # Work only on numerical columns

    # Calculate IQR
    Q1 = df_numeric.quantile(0.25)
    Q3 = df_numeric.quantile(0.75)
    IQR = Q3 - Q1

    # Apply filtering on numeric columns only
    mask = ~((df_numeric < (Q1 - threshold * IQR)) | (df_numeric > (Q3 + threshold * IQR))).any(axis=1)
    
    # Return filtered DataFrame (keep categorical data intact)
    return df[mask]

# Load dataset
df = pd.read_csv("../data/scaled_train.csv")

# Remove outliers
df_cleaned = remove_outliers(df)

# Save cleaned dataset
df_cleaned.to_csv("../data/final_cleaned_train.csv", index=False)

print("Outliers removed and dataset saved.")
