import optuna
import xgboost as xgb
import pandas as pd
import numpy np
from sklearn.model_selection import cross_val_score

# Load dataset
data = pd.read_csv("data/processed_train.csv")
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "random_state": 42,
        "enable_categorical": False
    }
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -np.mean(scores)

# Reduce verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)

# Optimize
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=100)

# Save best parameters and RMSE
import os
os.makedirs("models", exist_ok=True)
with open("models/best_params.pkl", "wb") as f:
    pickle.dump(study.best_params, f)
with open("models/best_rmse.pkl", "wb") as f:  # Save best RMSE for display
    pickle.dump(-study.best_value, f)

print(f"Best parameters: {study.best_params}")
print(f"Best RMSE: {-study.best_value:.4f}")