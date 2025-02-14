import joblib
import pandas as pd
from flask import Flask, request, render_template

app = Flask(__name__)

# Load the pre-trained model and data cleaning pipeline
model = joblib.load('scripts/model.pkl')  # Ensure this is the correct path to your model
data_cleaning_pipeline = joblib.load('scripts/data_cleaning_pipeline.pkl')  # Ensure this is the correct path to your pipeline

# Function to clean incoming data
def clean_input_data(input_data):
    # Assuming the input is a dictionary with all required features
    df = pd.DataFrame([input_data])
    
    # Apply the data cleaning pipeline to the input data
    cleaned_data = data_cleaning_pipeline.transform(df)  # This should match the pipeline steps
    return cleaned_data

# Function to make predictions
def make_prediction(cleaned_data):
    prediction = model.predict(cleaned_data)
    return prediction[0]

@app.route('/')
def home():
    return render_template('index.html')  # Replace with the appropriate HTML file for your UI

@app.route('/predict', methods=['POST'])
def predict():
    # Retrieve data from form inputs (update based on your form structure)
    input_data = {
        '1stFlrSF': float(request.form['1stFlrSF']),
        '2ndFlrSF': float(request.form['2ndFlrSF']),
        'BedroomAbvGr': int(request.form['BedroomAbvGr']),
        'BsmtExposure': request.form['BsmtExposure'],
        'BsmtFinType1': request.form['BsmtFinType1'],
        'BsmtFinSF1': float(request.form['BsmtFinSF1']),
        'BsmtUnfSF': float(request.form['BsmtUnfSF']),
        'TotalBsmtSF': float(request.form['TotalBsmtSF']),
        'GarageArea': float(request.form['GarageArea']),
        'GarageFinish': request.form['GarageFinish'],
        'GarageYrBlt': int(request.form['GarageYrBlt']),
        'GrLivArea': float(request.form['GrLivArea']),
        'KitchenQual': request.form['KitchenQual'],
        'LotArea': float(request.form['LotArea']),
        'LotFrontage': float(request.form['LotFrontage']),
        'MasVnrArea': float(request.form['MasVnrArea']),
        'EnclosedPorch': float(request.form['EnclosedPorch']),
        'OpenPorchSF': float(request.form['OpenPorchSF']),
        'OverallCond': int(request.form['OverallCond']),
        'OverallQual': int(request.form['OverallQual']),
        'WoodDeckSF': float(request.form['WoodDeckSF']),
        'YearBuilt': int(request.form['YearBuilt']),
        'YearRemodAdd': int(request.form['YearRemodAdd']),
    }

    # Clean the incoming data
    cleaned_data = clean_input_data(input_data)

    # Make prediction using the cleaned data
    predicted_price = make_prediction(cleaned_data)
    
    # Return prediction result to the user
    return render_template('result.html', prediction=predicted_price)

if __name__ == '__main__':
    app.run(debug=True)

