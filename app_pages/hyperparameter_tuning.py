import optuna
import xgboost as xgb
from sklearn.model_selection import cross_val_score
import numpy as np

# Load dataset
data = pd.read_csv("data/processed_train.csv")
X = data.drop(columns=["SalePrice"])
y = data["SalePrice"]

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 1000, step=100),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0, 5),
    }
    
    model = xgb.XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=5, scoring="neg_root_mean_squared_error")
    return -np.mean(scores)  # Minimize RMSE

# Run optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=50)

# Get best parameters
best_params = study.best_params
print("Best Parameters:", best_params)

# Train final model with best parameters
final_model = xgb.XGBRegressor(**best_params)
final_model.fit(X, y)

# Save model
import pickle
with open("models/trained_model.pkl", "wb") as f:
    pickle.dump(final_model, f)
