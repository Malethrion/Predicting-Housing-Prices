import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

def app():
    """Provide a user interface to predict house prices based on input features."""
    st.title("Predicting House Prices")

    # Load trained model, preprocessor, and feature names
    model_path = "models/optimized_model.pkl"  # Use optimized model for better predictions
    preprocessor_path = "models/preprocessor.pkl"
    feature_names_path = "models/feature_names.pkl"

    if not os.path.exists(model_path):
        st.error(f"Optimized model file not found at `{model_path}`. Run hyperparameter tuning and model training first.")
        return
    if not os.path.exists(preprocessor_path):
        st.error(f"Preprocessor file not found at `{preprocessor_path}`. Run feature engineering first.")
        return
    if not os.path.exists(feature_names_path):
        st.error(f"Feature names file not found at `{feature_names_path}`. Run feature engineering first.")
        return

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        with open(feature_names_path, "rb") as f:
            expected_features = pickle.load(f)
    except Exception as e:
        st.error(f"Error loading files: {e}")
        return

    st.write("Optimized Model, Preprocessor, and Feature Names Loaded!")

    # User inputs (numerical and categorical features)
    st.subheader("Enter House Features:")
    GrLivArea = st.number_input("GrLivArea (Above Ground Living Area in SqFt)", min_value=500, max_value=10000, value=3000)
    OverallQual = st.slider("OverallQual (Overall Quality 1-10)", 1, 10, 9)
    GarageCars = st.slider("GarageCars (Number of Garages)", 0, 5, 3)
    YearBuilt = st.number_input("YearBuilt (Year House Built)", min_value=1800, max_value=2025, value=2020)
    TotalBsmtSF = st.number_input("TotalBsmtSF (Total Basement Size in SqFt)", min_value=0, max_value=10000, value=2000)

    # Hardcode common categorical values for high-value homes (based on Kaggle dataset defaults)
    MSZoning = "RL"
    Street = "Pave"
    Alley = "No"
    LotShape = "Reg"
    LandContour = "Lvl"
    Utilities = "AllPub"
    LotConfig = "Inside"
    LandSlope = "Gtl"
    Neighborhood = "NoRidge"  # High-value neighborhood
    Condition1 = "Norm"
    Condition2 = "Norm"
    BldgType = "1Fam"
    HouseStyle = "2Story"  # Larger house style
    RoofStyle = "Gable"
    RoofMatl = "CompShg"
    Exterior1st = "VinylSd"
    Exterior2nd = "VinylSd"
    MasVnrType = "BrkFace"  # Brick face for higher value
    ExterQual = "Ex"  # Excellent quality
    ExterCond = "TA"
    Foundation = "PConc"
    BsmtQual = "Ex"  # Excellent basement quality
    BsmtCond = "TA"
    BsmtExposure = "Gd"  # Good exposure for higher value
    BsmtFinType1 = "GLQ"  # Good Living Quarters
    BsmtFinType2 = "Unf"
    Heating = "GasA"
    HeatingQC = "Ex"
    CentralAir = "Y"
    Electrical = "SBrkr"
    KitchenQual = "Ex"  # Excellent kitchen quality
    Functional = "Typ"
    FireplaceQu = "Ex"  # Excellent fireplace quality
    GarageType = "Attchd"
    GarageFinish = "Fin"  # Finished garage
    GarageQual = "TA"
    GarageCond = "TA"
    PavedDrive = "Y"
    PoolQC = "Ex"  # Excellent pool quality for higher value
    Fence = "GdPrv"  # Good privacy fence for higher value
    MiscFeature = "Shed"  # Shed for higher value
    SaleType = "New"  # New construction for higher value
    SaleCondition = "Partial"  # Partial sale for higher value

    # Define numerical and categorical features
    numerical_features = ['Id', 'MSSubClass', 'LotFrontage', 'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 
                         'YearRemodAdd', 'MasVnrArea', 'BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF', 'TotalBsmtSF', 
                         '1stFlrSF', '2ndFlrSF', 'LowQualFinSF', 'GrLivArea', 'BsmtFullBath', 'BsmtHalfBath', 
                         'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr', 'TotRmsAbvGrd', 'Fireplaces', 
                         'GarageYrBlt', 'GarageCars', 'GarageArea', 'WoodDeckSF', 'OpenPorchSF', 'EnclosedPorch', 
                         '3SsnPorch', 'ScreenPorch', 'PoolArea', 'MiscVal', 'MoSold', 'YrSold']
    categorical_features = ['MSZoning', 'Street', 'Alley', 'LotShape', 'LandContour', 'Utilities', 'LotConfig', 
                           'LandSlope', 'Neighborhood', 'Condition1', 'Condition2', 'BldgType', 'HouseStyle', 
                           'RoofStyle', 'RoofMatl', 'Exterior1st', 'Exterior2nd', 'MasVnrType', 'ExterQual', 
                           'ExterCond', 'Foundation', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 
                           'BsmtFinType2', 'Heating', 'HeatingQC', 'CentralAir', 'Electrical', 'KitchenQual', 
                           'Functional', 'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 
                           'PavedDrive', 'PoolQC', 'Fence', 'MiscFeature', 'SaleType', 'SaleCondition']

    # Create user_data with default float64 for numerical and object for categorical
    user_data = pd.DataFrame(columns=numerical_features + categorical_features)
    user_data[numerical_features] = user_data[numerical_features].astype('float64')
    user_data[categorical_features] = user_data[categorical_features].astype('object')

    # Set numerical features (default to median or realistic values for higher predictions)
    user_data.loc[0, 'GrLivArea'] = GrLivArea
    user_data.loc[0, 'OverallQual'] = OverallQual
    user_data.loc[0, 'GarageCars'] = GarageCars
    user_data.loc[0, 'YearBuilt'] = YearBuilt
    user_data.loc[0, 'TotalBsmtSF'] = TotalBsmtSF

    # Set realistic defaults for other numerical features to support higher prices
    user_data.loc[0, 'LotFrontage'] = 100  # Larger lot frontage for higher-value homes
    user_data.loc[0, 'LotArea'] = 15000  # Larger lot area for higher-value homes
    user_data.loc[0, 'OverallCond'] = 7  # Good condition
    user_data.loc[0, 'YearRemodAdd'] = YearBuilt  # Assume remodeled same year as built
    user_data.loc[0, 'MasVnrArea'] = 400  # Larger masonry veneer area
    user_data.loc[0, 'BsmtFinSF1'] = 1500  # Larger finished basement area
    user_data.loc[0, 'BsmtUnfSF'] = 500  # Unfinished basement area
    user_data.loc[0, '1stFlrSF'] = 2000  # Larger first floor size
    user_data.loc[0, '2ndFlrSF'] = 1500  # Larger second floor size (for 2Story)
    user_data.loc[0, 'BsmtFullBath'] = 1  # Full basement bathroom
    user_data.loc[0, 'FullBath'] = 3  # More full bathrooms for higher value
    user_data.loc[0, 'HalfBath'] = 1  # Half bathrooms
    user_data.loc[0, 'BedroomAbvGr'] = 5  # More bedrooms
    user_data.loc[0, 'KitchenAbvGr'] = 1  # Kitchens
    user_data.loc[0, 'TotRmsAbvGrd'] = 10  # More total rooms
    user_data.loc[0, 'Fireplaces'] = 2  # More fireplaces
    user_data.loc[0, 'GarageArea'] = 800  # Larger garage area
    user_data.loc[0, 'WoodDeckSF'] = 300  # Larger wood deck
    user_data.loc[0, 'OpenPorchSF'] = 150  # Larger open porch
    user_data.loc[0, 'PoolArea'] = 500  # Pool area for higher value
    user_data.loc[0, 'MiscVal'] = 1000  # Miscellaneous value (e.g., shed)
    user_data.loc[0, 'MoSold'] = 6  # Mid-year sale
    user_data.loc[0, 'YrSold'] = 2023  # Recent year

    # Set categorical features with hardcoded values for higher-value homes
    user_data.loc[0, 'MSZoning'] = MSZoning
    user_data.loc[0, 'Street'] = Street
    user_data.loc[0, 'Alley'] = Alley
    user_data.loc[0, 'LotShape'] = LotShape
    user_data.loc[0, 'LandContour'] = LandContour
    user_data.loc[0, 'Utilities'] = Utilities
    user_data.loc[0, 'LotConfig'] = LotConfig
    user_data.loc[0, 'LandSlope'] = LandSlope
    user_data.loc[0, 'Neighborhood'] = Neighborhood
    user_data.loc[0, 'Condition1'] = Condition1
    user_data.loc[0, 'Condition2'] = Condition2
    user_data.loc[0, 'BldgType'] = BldgType
    user_data.loc[0, 'HouseStyle'] = HouseStyle
    user_data.loc[0, 'RoofStyle'] = RoofStyle
    user_data.loc[0, 'RoofMatl'] = RoofMatl
    user_data.loc[0, 'Exterior1st'] = Exterior1st
    user_data.loc[0, 'Exterior2nd'] = Exterior2nd
    user_data.loc[0, 'MasVnrType'] = MasVnrType
    user_data.loc[0, 'ExterQual'] = ExterQual
    user_data.loc[0, 'ExterCond'] = ExterCond
    user_data.loc[0, 'Foundation'] = Foundation
    user_data.loc[0, 'BsmtQual'] = BsmtQual
    user_data.loc[0, 'BsmtCond'] = BsmtCond
    user_data.loc[0, 'BsmtExposure'] = BsmtExposure
    user_data.loc[0, 'BsmtFinType1'] = BsmtFinType1
    user_data.loc[0, 'BsmtFinType2'] = BsmtFinType2
    user_data.loc[0, 'Heating'] = Heating
    user_data.loc[0, 'HeatingQC'] = HeatingQC
    user_data.loc[0, 'CentralAir'] = CentralAir
    user_data.loc[0, 'Electrical'] = Electrical
    user_data.loc[0, 'KitchenQual'] = KitchenQual
    user_data.loc[0, 'Functional'] = Functional
    user_data.loc[0, 'FireplaceQu'] = FireplaceQu
    user_data.loc[0, 'GarageType'] = GarageType
    user_data.loc[0, 'GarageFinish'] = GarageFinish
    user_data.loc[0, 'GarageQual'] = GarageQual
    user_data.loc[0, 'GarageCond'] = GarageCond
    user_data.loc[0, 'PavedDrive'] = PavedDrive
    user_data.loc[0, 'PoolQC'] = PoolQC
    user_data.loc[0, 'Fence'] = Fence
    user_data.loc[0, 'MiscFeature'] = MiscFeature
    user_data.loc[0, 'SaleType'] = SaleType
    user_data.loc[0, 'SaleCondition'] = SaleCondition

    # Show User Input Summary in an optional expander
    with st.expander("View User Input Summary", expanded=False):
        user_input_summary = {
            "GrLivArea": GrLivArea,
            "OverallQual": OverallQual,
            "GarageCars": GarageCars,
            "YearBuilt": YearBuilt,
            "TotalBsmtSF": TotalBsmtSF,
            "MSZoning": MSZoning,
            "Neighborhood": Neighborhood,
            "BldgType": BldgType,
            "HouseStyle": HouseStyle
        }
        st.write("User Input Summary:")
        st.json(user_input_summary)

    # Transform user input using the preprocessor
    try:
        transformed_data = preprocessor.transform(user_data)
    except Exception as e:
        st.error(f"Preprocessing Error: {e}")
        st.write("Expected Features by Preprocessor:", preprocessor.get_feature_names_out().tolist())
        return

    # Ensure feature order matches training (DataFrame for debugging, optional)
    transformed_df = pd.DataFrame(transformed_data, columns=expected_features)

    # Debug: Print transformed data shape (optional, hidden from users)
    st.write("Transformed Input Data Shape:", transformed_df.shape)

    # Predict
    log_price = model.predict(transformed_data)
    predicted_price = np.expm1(log_price[0])  # Convert log price back to normal

    st.subheader("Predicted House Price:")
    st.success(f"Predicted Price: ${predicted_price:,.2f}")

    # Debug: Print raw prediction for verification (optional, hidden from users)
    st.write(f"Raw Log Price Prediction: {log_price[0]}")

if __name__ == "__main__":
    app()