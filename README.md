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
