import pandas as pd

# Load processed training data
data = pd.read_csv("data/processed_train.csv")

# Check for zero or negative SalePrice values
invalid_prices = data[data["SalePrice"] <= 0]
print("❌ Invalid SalePrice rows:")
print(invalid_prices)
