import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the trained model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

# Set up the Streamlit app
st.title('Housing Price Prediction')

# Create user inputs for the model
lot_area = st.number_input('Lot Area', min_value=0)
overall_qual = st.selectbox('Overall Quality', [1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
year_built = st.number_input('Year Built', min_value=1800, max_value=2025)

# More inputs can be added based on the columns used in the model

# Once the user submits, make the prediction
if st.button('Predict'):
    input_data = pd.DataFrame({
        'LotArea': [lot_area],
        'OverallQual': [overall_qual],
        'YearBuilt': [year_built],
        # Add more features based on the ones in your model
    })
    
    # Prepare the data using the same preprocessing steps as before
    numeric_columns = input_data.select_dtypes(include=['number']).columns
    categorical_columns = input_data.select_dtypes(include=['object']).columns
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_columns),  # Scaling numerical features
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)  # OneHotEncoding categorical features
        ])
    
    # Make predictions
    prediction = model.predict(input_data)
    
    # Display the prediction
    st.write(f"Predicted Sale Price: ${prediction[0]:,.2f}")

