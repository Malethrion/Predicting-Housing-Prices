# Housing Price Predictor - A Predictive Model for Estimating Real Estate Prices

[Housing Price Predictor](https://your-deployed-link.com/) is a machine-learning (ML) project that uses a publicly available dataset to predict the prices of houses based on various features. The goal is to build a predictive model that can help real estate businesses estimate the value of properties more accurately. The project leverages regression analysis and advanced machine learning techniques for this task.

## Table of Contents

- [Dataset Content](#dataset-content)
- [Business Requirements](#business-requirements)
- [Hypothesis](#hypothesis-and-how-to-validate)
- [Mapping Business Requirements to Data Visualization and ML Tasks](#the-rationale-to-map-the-business-requirements-to-the-data-visualizations-and-ml-tasks)
- [ML Business Case](#ml-business-case)
- [Epics and User Stories](#epics-and-user-stories)
- [Dashboard Design](#dashboard-design)
- [Technologies Used](#technologies-used)
- [Testing](#testing)
- [Unfixed Bugs](#unfixed-bugs)
- [Deployment](#deployment)
- [Credits](#credits)
- [Acknowledgements](#acknowledgements)

## Dataset Content

The dataset is sourced from [Kaggle](https://www.kaggle.com/c/house-prices-advanced-regression-techniques/data). Each row represents a property listing, with features such as:

- The size of the property (square feet)
- Number of bedrooms and bathrooms
- Property location and neighborhood
- Various physical attributes (e.g., garage, pool, etc.)
- Historical sale price of properties

| Attribute       | Information                                       | Units                       |
|-----------------|---------------------------------------------------|-----------------------------|
| `OverallQual`   | Overall material and finish quality               | Rating from 1 to 10         |
| `GrLivArea`     | Above grade (ground) living area in square feet  | square feet                 |
| `TotalBsmtSF`   | Total basement area in square feet                | square feet                 |
| `GarageCars`    | Size of garage in car capacity                    | Number of cars              |
| `PoolArea`      | Pool area in square feet                          | square feet                 |
| `SalePrice`     | The sale price of the house                       | USD                         |

[Back to top](#table-of-contents)

## Business Requirements

* Real estate businesses need to predict housing prices based on historical data and various house attributes.
* **Business Requirement 1**: Real estate businesses want to understand which features most strongly influence house pricing.
* **Business Requirement 2**: Predict the sale price of houses given a set of property features.

[Back to top](#table-of-contents)

## Hypothesis and how to validate?

* **Hypothesis 1**:
    - We hypothesize that house size and neighborhood will be the most significant factors influencing house prices.
    - **Validation**: Perform correlation analysis and feature importance study.

* **Hypothesis 2**:
    - A successful prediction will rely on multiple features such as house size, age, and neighborhood.
    - **Validation**: Evaluate feature importance and model performance using various metrics.

* **Hypothesis 3**:
    - Homes in popular neighborhoods will have higher prices.
    - **Validation**: Visualize and analyze price distribution by neighborhood.

[Back to top](#table-of-contents)

## The rationale to map the business requirements to the Data Visualizations and ML tasks

* **Business Requirement 1**: Data Visualization and Correlation study
    - We need to perform a correlation study to identify the most significant factors impacting house prices.
    - Pearson’s and Spearman’s correlation tests will be applied to understand relationships.
    - Predictive Power Score (PPS) will help assess relationships between features, considering both categorical and numerical data.

* **Business Requirement 2**: Regression Model
    - The goal is to predict house prices using regression analysis.
    - We will build a regression model and apply hyperparameter optimization to maximize its performance.
    - This task will be completed during the **Model Training, Optimization, and Validation** Epic.

[Back to top](#table-of-contents)

## ML Business Case

**Regression Model**

* We want to develop a machine learning model to predict housing prices based on features such as house size, number of bedrooms, neighborhood, and more.
* The target variable is `SalePrice`, a continuous variable.
* Success metrics:
    - R-squared of at least 0.85 for both train and test sets.
* The model will be considered a failure if:
    - The R-squared is below 0.80.
    - The model does not generalize well to unseen data.
* The training data for the model comes from Kaggle's dataset, which contains historical housing prices and features.

[Back to top](#table-of-contents)

## Epics and User Stories

### Epic - Information Gathering and Data Collection
* **User Story**: As a data analyst, I can import the dataset from Kaggle and save it locally for analysis.
* **User Story**: As a data analyst, I can load the saved dataset for further exploration and analysis.

### Epic - Data Visualization, Cleaning, and Preparation
* **User Story**: As a data scientist, I can visualize the dataset and identify which features most strongly correlate with house prices (**Business Requirement 1**).
* **User Story**: As a data analyst, I can clean the dataset and handle any missing or outlier values.
* **User Story**: As a data scientist, I can carry out feature engineering to transform the data for optimal use in the ML model.

### Epic - Model Training, Optimization, and Validation
* **User Story**: As a data engineer, I can train the regression model using a train-test split.
* **User Story**: As a data scientist, I can optimize hyperparameters for the best model performance.
* **User Story**: As a data scientist, I can evaluate model performance to determine if it successfully predicts house prices.

### Epic - Dashboard Planning, Designing, and Development
* **User Story**: As a non-technical user, I can input features (e.g., square footage, number of bedrooms) and get a predicted house price.
* **User Story**: As a non-technical user, I can view a summary of the project's business case and conclusions.

### Epic - Dashboard Deployment and Release
* **User Story**: As a user, I can interact with the dashboard and see live predictions and analysis.

[Back to top](#table-of-contents)


## Live Link:
https://predicting-housing-prices.onrender.com

## Cloud IDE Reminders

To log into the Heroku toolbelt CLI:

1. Log in to your Heroku account and go to _Account Settings_ in the menu under your avatar.
2. Scroll down to the _API Key_ and click _Reveal_
3. Copy the key
4. In the terminal, run `heroku_config`
5. Paste in your API key when asked

You can now use the `heroku` CLI program - try running `heroku apps` to confirm it works. This API key is unique and private to you so do not share it. If you accidentally make it public then you can create a new one with _Regenerate API Key_.

* Set the runtime.txt Python version to a [Heroku-22](https://devcenter.heroku.com/articles/python-support#supported-runtimes) stack currently supported version.
* The project was deployed to Heroku using the following steps.

1. Log in to Heroku and create an App
2. At the Deploy tab, select GitHub as the deployment method.
3. Select your repository name and click Search. Once it is found, click Connect.
4. Select the branch you want to deploy, then click Deploy Branch.
5. The deployment process should happen smoothly if all deployment files are fully functional. Click now the button Open App on the top of the page to access your App.
6. If the slug size is too large then add large files not required for the app to the .slugignore file.
