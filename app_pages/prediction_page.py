import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os

def app():
    st.title("🏡 House Price Prediction")

    # Load trained model, preprocessor, and feature names
    model_path = "models/trained_model.pkl"
    preprocessor_path = "models/preprocessor.pkl"
    feature_names_path = "models/feature_names.pkl"

    if not os.path.exists(model_path):
        st.error(f"❌ Model file not found at `{model_path}`. Train the model first.")
        return
    if not os.path.exists(preprocessor_path):
        st.error(f"❌ Preprocessor file not found at `{preprocessor_path}`. Run feature engineering first.")
        return
    if not os.path.exists(feature_names_path):
        st.error(f"❌ Feature names file not found at `{feature_names_path}`. Run feature engineering first.")
        return

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        with open(preprocessor_path, "rb") as f:
            preprocessor = pickle.load(f)
        with open(feature_names_path, "rb") as f:
            expected_features = pickle.load(f)
    except Exception as e:
        st.error(f"❌ Error loading files: {e}")
        return

    st.success("✅ Model, Preprocessor, and Feature Names Loaded!")

    # User inputs (numerical and categorical features)
    st.subheader("Enter House Features:")
    GrLivArea = st.number_input("GrLivArea (Above Ground Living Area in SqFt)", min_value=500, max_value=5000, value=1500)
    OverallQual = st.slider("OverallQual (Overall Quality 1-10)", 1, 10, 5)
    GarageCars = st.slider("GarageCars (Number of Garages)", 0, 4, 2)
    YearBuilt = st.number_input("YearBuilt (Year House Built)", min_value=1800, max_value=2025, value=2000)
    TotalBsmtSF = st.number_input("TotalBsmtSF (Total Basement Size in SqFt)", min_value=0, max_value=5000, value=1000)

    # Hardcode common categorical values (based on Kaggle dataset defaults)
    MSZoning = "RL"  # Most common zoning
    Street = "Pave"  # Most common street type
    Alley = "No"    # Assuming "No" for missing alleys
    LotShape = "Reg"  # Most common lot shape
    LandContour = "Lvl"  # Most common contour
    Utilities = "AllPub"  # Most common utility
    LotConfig = "Inside"  # Most common configuration
    LandSlope = "Gtl"  # Most common slope
    Neighborhood = "NAmes"  # Common neighborhood
    Condition1 = "Norm"  # Most common condition
    Condition2 = "Norm"  # Most common condition
    BldgType = "1Fam"  # Most common building type
    HouseStyle = "1Story"  # Most common house style
    RoofStyle = "Gable"  # Most common roof style
    RoofMatl = "CompShg"  # Most common roof material
    Exterior1st = "VinylSd"  # Most common exterior
    Exterior2nd = "VinylSd"  # Most common exterior
    MasVnrType = "None"  # Most common masonry veneer
    ExterQual = "TA"  # Most common exterior quality
    ExterCond = "TA"  # Most common exterior condition
    Foundation = "PConc"  # Most common foundation
    BsmtQual = "TA"  # Most common basement quality
    BsmtCond = "TA"  # Most common basement condition
    BsmtExposure = "No"  # Most common basement exposure
    BsmtFinType1 = "Unf"  # Most common basement finish type
    BsmtFinType2 = "Unf"  # Most common second basement finish
    Heating = "GasA"  # Most common heating type
    HeatingQC = "Ex"  # Most common heating quality
    CentralAir = "Y"  # Most common central air
    Electrical = "SBrkr"  # Most common electrical
    KitchenQual = "TA"  # Most common kitchen quality
    Functional = "Typ"  # Most common functionality
    FireplaceQu = "No"  # Assuming "No" for no fireplace
    GarageType = "Attchd"  # Most common garage type
    GarageFinish = "Unf"  # Most common garage finish
    GarageQual = "TA"  # Most common garage quality
    GarageCond = "TA"  # Most common garage condition
    PavedDrive = "Y"  # Most common paved drive
    PoolQC = "No"  # Assuming "No" for no pool
    Fence = "No"  # Assuming "No" for no fence
    MiscFeature = "No"  # Assuming "No" for no miscellaneous feature
    SaleType = "WD"  # Most common sale type
    SaleCondition = "Normal"  # Most common sale condition

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

    # Set numerical features (default to 0 or median if not provided)
    user_data.loc[0, 'GrLivArea'] = GrLivArea
    user_data.loc[0, 'OverallQual'] = OverallQual
    user_data.loc[0, 'GarageCars'] = GarageCars
    user_data.loc[0, 'YearBuilt'] = YearBuilt
    user_data.loc[0, 'TotalBsmtSF'] = TotalBsmtSF

    for feature in numerical_features:
        if feature not in ['GrLivArea', 'OverallQual', 'GarageCars', 'YearBuilt', 'TotalBsmtSF']:
            user_data.loc[0, feature] = 0

    # Set categorical features with hardcoded values (no dtype warning since 'object' is used)
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

    # Show only user-provided inputs and key categorical defaults
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
    st.write("🔍 **User Input Summary:**")
    st.json(user_input_summary)

    # Transform user input using the preprocessor
    try:
        transformed_data = preprocessor.transform(user_data)
    except Exception as e:
        st.error(f"❌ Preprocessing Error: {e}")
        st.write("🔍 **Expected Features by Preprocessor:**", preprocessor.get_feature_names_out().tolist())
        return

    # Ensure feature order matches training (DataFrame for debugging, optional)
    transformed_df = pd.DataFrame(transformed_data, columns=expected_features)

    # Debug: Print transformed data shape (optional, hidden from users)
    st.write("🔍 **Transformed Input Data Shape:**", transformed_df.shape)

    # Predict
    log_price = model.predict(transformed_data)
    predicted_price = np.expm1(log_price[0])  # Convert log price back to normal

    st.subheader("💰 Predicted House Price:")
    st.success(f"Predicted Price: **${predicted_price:,.2f}**")

    # Debug: Print raw prediction for verification (optional, hidden from users)
    st.write(f"🔍 **Raw Log Price Prediction:** {log_price[0]}")

if __name__ == "__main__":
    app()