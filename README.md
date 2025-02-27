# Predicting Housing Prices - A Predictive Model for Estimating Real Estate Prices

[Predicting Housing Prices](https://predicting-housing-prices.onrender.com/) is a machine learning (ML) project that uses a publicly available dataset to predict house prices based on various features. The goal is to build a predictive model to help real estate businesses estimate property values more accurately. The project leverages regression analysis and advanced machine learning techniques, implemented with Python and deployed as an interactive Streamlit web application.

## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis and Validation](#hypothesis-and-validation)
- [Mapping Business Requirements to Data Visualizations and ML Tasks](#mapping-business-requirements-to-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Known Issues](#known-issues)
- [Installation and Setup](#installation-and-setup)
- [Deployment](#deployment)
- [Forking and Cloning](#forking-and-cloning)
- [Installing Requirements](#installing-requirements)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)

## Dataset Content

The dataset is sourced from [Kaggle’s "House Prices - Advanced Regression Techniques"](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Each row represents a property listing, with features such as:

- Property size (square feet)
- Number of bedrooms and bathrooms
- Property location and neighborhood
- Various physical attributes (e.g., garage, pool)
- Historical sale price of properties

| Attribute       | Description                                       | Units                       |
|-----------------|---------------------------------------------------|-----------------------------|
| `OverallQual`   | Overall material and finish quality               | Rating from 1 to 10         |
| `GrLivArea`     | Above-grade (ground) living area in square feet   | Square feet                 |
| `TotalBsmtSF`   | Total basement area in square feet                | Square feet                 |
| `GarageCars`    | Size of garage in car capacity                    | Number of cars              |
| `PoolArea`      | Pool area in square feet                          | Square feet                 |
| `SalePrice`     | The sale price of the house                       | USD                         |

[Back to top](#table-of-contents)

## Business Requirements

- Real estate businesses need to predict housing prices using historical data and property attributes.
- **Business Requirement 1**: Understand which features most strongly influence house pricing through correlation analysis and feature importance.
- **Business Requirement 2**: Predict the sale price of houses given a set of property features using a regression model.

[Back to top](#table-of-contents)

## Hypothesis and Validation

- **Hypothesis 1**:
  - House size and neighborhood are the most significant factors influencing house prices.
  - **Validation**: Conduct correlation analysis, feature importance studies, and visualize relationships using scatter plots and heatmaps.

- **Hypothesis 2**:
  - Accurate predictions rely on multiple features, including house size, age, and neighborhood.
  - **Validation**: Evaluate model performance using metrics like RMSE, R-squared, and cross-validation, and analyze feature importance.

- **Hypothesis 3**:
  - Homes in popular neighborhoods have higher prices.
  - **Validation**: Analyze price distributions by neighborhood and visualize with histograms or box plots.

[Back to top](#table-of-contents)

## Mapping Business Requirements to Data Visualizations and ML Tasks

- **Business Requirement 1**: Data Visualization and Correlation Analysis
  - Perform correlation studies (Pearson’s, Spearman’s) and Predictive Power Score (PPS) to identify key predictors.
  - Visualize relationships using heatmaps, scatter plots, and bar charts in the Streamlit app’s “Correlation Study” page.

- **Business Requirement 2**: Regression Model Development
  - Build an XGBoost regression model, optimize hyperparameters with Optuna, and validate using train-test splits and cross-validation.
  - Implement prediction functionality in the “House Price Prediction” page, allowing users to input features and receive price estimates.

[Back to top](#table-of-contents)

## ML Business Case

**Regression Model for House Price Prediction**

- **Objective**: Develop a machine learning model to predict housing prices (`SalePrice`) based on features like size, bedrooms, neighborhood, and more.
- **Target Variable**: `SalePrice` (continuous, in USD).
- **Success Metrics**:
  - R-squared ≥ 0.85 for both training and testing sets.
  - RMSE ≤ 0.13 on log-transformed prices (based on current results).
- **Failure Criteria**:
  - R-squared < 0.80.
  - Poor generalization to unseen data (e.g., high RMSE on test set).
- **Data Source**: Kaggle’s house price dataset, containing historical pricing and feature data for ~1,460 properties.

[Back to top](#table-of-contents)

## Epics and User Stories

### Epic - Information Gathering and Data Collection
- **User Story**: As a data analyst, I can import the Kaggle dataset and save it locally for analysis.
- **User Story**: As a data analyst, I can load the saved dataset for exploration and processing.

### Epic - Data Visualization, Cleaning, and Preparation
- **User Story**: As a data scientist, I can visualize the dataset to identify features correlating with house prices (Business Requirement 1).
- **User Story**: As a data analyst, I can clean the dataset, handling missing values and outliers.
- **User Story**: As a data scientist, I can engineer features (e.g., scaling, encoding) for optimal ML model performance.

### Epic - Model Training, Optimization, and Validation
- **User Story**: As a data engineer, I can train an XGBoost regression model using a train-test split.
- **User Story**: As a data scientist, I can optimize model hyperparameters offline with Optuna for best performance.
- **User Story**: As a data scientist, I can evaluate model performance (e.g., RMSE, R-squared) to ensure accurate predictions.

### Epic - Dashboard Planning, Designing, and Development
- **User Story**: As a non-technical user, I can input house features (e.g., size, quality) and receive a predicted price (Business Requirement 2).
- **User Story**: As a non-technical user, I can view project summaries, business requirements, and conclusions.

### Epic - Dashboard Deployment and Release
- **User Story**: As a user, I can interact with the deployed Streamlit app to explore predictions and visualizations in real time.

[Back to top](#table-of-contents)

## Dashboard Design

The Predicting Housing Prices app is structured as an interactive Streamlit web application with multiple pages, categorized as user-facing (visible in the dashboard) and backend (for development and processing). Below is the design for each page:

### User-Facing Pages (Visible in the Dashboard)

#### Page 1: Home Page
- **Section 1 - Overview**:
  - Introduction to the project, dataset, and business requirements, as defined in `home_page.py`.

#### Page 2: Correlation Study
- Display correlation heatmaps, scatter plots, and bar charts to identify key predictors of house prices, as implemented in `correlation_study.py`.
- Allow users to explore relationships between features and `SalePrice`, with options for:
  - Interactive feature selection for scatter plots (e.g., `GrLivArea`, `OverallQual`).
  - Additional visualizations like histograms or box plots of `SalePrice` by feature.
- Note: Users can toggle between full correlation heatmaps and simplified views focusing on significant correlations with `SalePrice`.

#### Page 3: House Price Prediction
- Provide an input form for users to enter property features (e.g., `GrLivArea`, `OverallQual`, `GarageCars`, `YearBuilt`, `TotalBsmtSF`), as implemented in `prediction_page.py`.
- Display the predicted house price in USD and optional feature summaries (e.g., in an expander for advanced users).
- Include error handling for invalid or missing inputs, ensuring a robust user experience.

#### Page 4: Feature Importance
- Show a bar chart of the top 20 features influencing `SalePrice` from the trained XGBoost model, as implemented in `feature_importance.py`.
- Include model performance metrics (e.g., RMSE, R-squared) to evaluate prediction accuracy.
- Offer optional sorting or filtering of features for deeper analysis.

#### Page 5: Hyperparameter Tuning
- Display the optimized hyperparameters and cross-validation RMSE from offline tuning, as implemented in `hyperparameter_tuning.py`.
- Provide a summary of the best parameters and model performance, ensuring users understand the optimization process without requiring interaction.

### Backend Pages (Not Visible in the Dashboard, for Development/Processing)

#### Page 6: Data Cleaning
- Process the raw dataset (`train.csv`) to handle missing values, outliers, and ensure data quality, as implemented in `data_cleaning.py`.
- Save the cleaned data to `data/final_cleaned_train.csv` for further use, not exposed to users.

#### Page 7: Feature Engineering
- Transform the cleaned data with scaling (`MinMaxScaler`) and encoding (`OneHotEncoder`), as implemented in `feature_engineering.py`.
- Save processed data to `data/processed_train.csv` and preprocessing objects to `models/preprocessor.pkl` and `models/feature_names.pkl`, not exposed to users.

#### Page 8: Model Training
- Train the XGBoost regression model using optimized hyperparameters and evaluate performance, as implemented in `model_training.py`.
- Save the trained model to `models/optimized_model.pkl`, not exposed to users.

[Back to top](#table-of-contents)

## Technologies Used

### Languages
- [Python](https://www.python.org/)

### Python Packages
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis.
- [NumPy](https://numpy.org/) - Numerical computations.
- [Matplotlib](https://matplotlib.org/) - Static visualizations.
- [Seaborn](https://seaborn.pydata.org/) - Statistical data visualization.
- [Scikit-learn](https://scikit-learn.org/) - Preprocessing and model validation.
- [XGBoost](https://xgboost.readthedocs.io/en/stable/) - Gradient boosting for regression.
- [Streamlit](https://streamlit.io/) - Interactive web app development.
- [Optuna](https://optuna.org/) - Hyperparameter optimization.

### Other Technologies
- [Git](https://git-scm.com/) - Version control.
- [Render](https://render.com/) - Cloud deployment platform.

[Back to top](#table-of-contents)

## Testing

### Manual Testing

| Feature                  | Action                          | Expected Result                                      | Actual Result                                      |
|--------------------------|---------------------------------|-----------------------------------------------------|---------------------------------------------------|
| Home Page                | View project summary            | Page loads with introduction and business requirements | Functions as intended                              |
| Correlation Study        | View correlation and price visualization     | Displays simplified heatmap: only significant correlations with `SalePrice`, full correlation heatmap for all numeric features, price distribution, interactive feature vs. `SalePrice` scatter plots and top 10 correlated features with `SalePrice` | Functions as intended                              |
| House Price Prediction   | Input features and predict      | Predicted price displayed in USD                     | Functions as intended                              |
| Feature Importance       | View feature importance chart   | Bar chart shows top 20 features by importance        | Functions as intended                              
| Hyperparameter Tuning    | View tuning results             | Optimized parameters and RMSE displayed instantly    | Functions as intended   

### Screenshots and Testing                           |
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Home.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Navigation%20bar.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Correlations%20with%20SalePrice.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Full%20Correlation%20Heatmap.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/SalePrice%20Distribution.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Feature%20vs.%20SAlePrice%20Scatter%20Plots.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Select%20a%20feature%20(Feature%20vs.%20SAlePrice%20Scatter%20Plots).png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Top%2010%20Features%20Correlated%20with%20SalePrice.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Predicting%20House%20Prices%20sample.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Feature%20Importance%20Ranking.png?raw=true)
![Predicting Housing Prices](https://github.com/Malethrion/Predicting-Housing-Prices/blob/main/static/Hyperparameter%20Tuning.png?raw=true)

[Back to top](#table-of-contents)

## Known Issues

No known issues at this time.

[Back to top](#table-of-contents)

## Installation and Setup

To prepare the app for use, run the following backend processes in order to generate necessary data and model files before launching the Streamlit app:

1. **Data Cleaning**:
   ```bash
   python script/data_cleaning.py

2. **Feature Engineering**:
   ```bash
   python script/feature_engineering.py

3. **Hyperparameter Tuning (Offline)**:
   ```bash
   python script/tune_hyperparameters.py

4. **Model Training**:
   ```bash
   python script/model_training.py

After completing these steps, run the Streamlit app: streamlit run app.py

Ensure all .csv and .pkl files are generated successfully. If errors occur, verify data/train.csv exists and requirements.txt is installed. Take screenshots of terminal output or file generation (e.g., outputs/backend_execution.png) for your submission.

[Back to top](#table-of-contents)

## Deployment

The project is deployed to Render. Follow these steps to deploy:

1. **Prerequisites**:
- Install Git and Python 3.8+ on your machine.
- Sign up for a Render account and install the Render CLI (render).

2. **Clone the Repository**:
- Clone this repository using:
   ```bash
   git clone https://github.com/Malethrion/Predicting-Housing-Prices.git

- Navigate to the project directory:
   ```bash
   cd Predicting-Housing-Prices

3. **Set Up Environment**:  
- Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   .\.venv\Scripts\activate  # Windows

- Install dependencies:
   ```bash
   pip install -r requirements.txt

4. **Pre-Generate Backend Files**:
- Run the backend processes locally as described in “Installation and Setup” to generate data/*.csv and models/*.pkl files:
   ```bash
   python app_pages/data_cleaning.py
   python app_pages/feature_engineering.py
   python tune_hyperparameters.py
   python app_pages/model_training.py

- Commit these files to your Git repository or ensure they’re included in deployment (temporarily remove models/*.pkl and data/*.csv from .gitignore if needed):
   ```bash
   # .gitignore (temporary update for deployment)
   venv/
   __pycache__/
   *.ipynb_checkpoints/

5. **Deploy to Render**:
- Log in to Render via the CLI:
 - Go to Render Account Settings, reveal your API key, and run:
   ```bash
   render login
 - Paste your API key when prompted.

- Create a new Render app:
 - Run render apps create and select “Web Service.”
 - Choose your GitHub repository and branch (e.g., main).
 - Set the runtime to Python 3.8+ (Render-22 stack).
 - Configure the build command: pip install -r requirements.txt.
 - Set the start command: streamlit run app.py --server.port $PORT.

- Deploy the app by pushing changes to GitHub:
   ```bash
   git push origin main

- Monitor deployment logs in the Render dashboard. Once complete, access your app at the provided URL.

6. **Troubleshooting**:
- If the slug size is too large, add large files (e.g., .ipynb, large datasets) to .slugignore:
   ```text
   # .slugignore
   jupyter_notebooks/*.ipynb
   data/*.csv

- Take a screenshot of your Render dashboard or deployment logs (e.g., outputs/render_deployment.png) for your submission.

[Back to top](#table-of-contents)

## Forking and Cloning

### Forking

1. Navigate to the [GitHub repository](https://github.com/Malethrion/Predicting-Housing-Prices).

2. Click the Fork button in the top-right corner to create a copy in your GitHub account.

### Cloning

1. Click the Code button on the repository page and copy the HTTPS URL.

2. In your terminal, clone the repository:
   ```bash
   git clone https://github.com/Malethrion/Predicting-Housing-Prices.git

3. Navigate to the project directory:
   ```bash
   cd Predicting-Housing-Prices

[Back to top](#table-of-contents)

## Installing Requirements

1. Run the following command to install the required packages:
   ```bash
   pip install -r requirements.txt

2. Ensure your requirements.txt includes only the necessary dependencies:
   ```bash
   numpy==1.26.4
   pandas==2.2.3
   matplotlib==3.8.0
   seaborn==0.13.2
   scikit-learn==1.6.1
   streamlit==1.42.2
   optuna==4.2.1
   xgboost==2.1.4

[Back to top](#table-of-contents)

## Credits

### Data Sources
- [Kaggle’s "House Prices - Advanced Regression Techniques" dataset](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data)

[Back to top](#table-of-contents)

## Acknowledgements
- Thanks to Mo Shami for guidance and feedback throughout the project.
- Appreciation to Code Institute for providing the learning platform and resources.

[Back to top](#table-of-contents)

