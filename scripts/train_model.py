import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import pickle

# Load the cleaned dataset
data = pd.read_csv('..data/final_cleaned_train.csv')

# Features and target (assuming 'SalePrice' is the target column)
X = data.drop(columns=['SalePrice'])
y = data['SalePrice']

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the model (RandomForestRegressor)
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train_scaled, y_train)

# Save the trained model to a .pkl file
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("Model has been trained and saved to 'model.pkl'.")
