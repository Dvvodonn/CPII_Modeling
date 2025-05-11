#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from CPII_RealEstate.models.random_forest import RandomForest

# where we’ll stash (and later reload) our tuned RF hyperparameters
PARAMS_FILE_RF = "CPII_RealEstate/training/best_params/rf_best_params.pkl"
# where we save/load the final Random Forest model
MODEL_PATH      = "CPII_RealEstate/outputs/random_forest_model.pkl"


def tune_and_evaluate():
    """
    Run a grid search over n_estimators, max_depth, and min_sample_split,
    printing MAE for each combination and persisting the best parameters.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training(
        "CPII_RealEstate/data/house_data.csv"
    )

    # Define parameter grid
    param_grid = {
        'n_estimators':     [50, 150, 200],
        'max_depth':        [None, 14, 16],
        'min_sample_split': [2, 5, 10]
    }

    best_mae = float('inf')
    best_params = {}

    # Grid search
    for n, d, m in itertools.product(
        param_grid['n_estimators'],
        param_grid['max_depth'],
        param_grid['min_sample_split']
    ):
        forest = RandomForest(
            n_estimators=n,
            max_depth=d,
            min_sample_split=m
        )
        forest.fit_with_timing(X_train, y_train)
        preds = forest.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Params: n_estimators={n}, max_depth={d}, min_sample_split={m} -> MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_params = {
                "n_estimators":     n,
                "max_depth":        d,
                "min_sample_split": m
            }

    print(
        f"\nBest params: n_estimators={best_params['n_estimators']}, "
        f"max_depth={best_params['max_depth']}, "
        f"min_sample_split={best_params['min_sample_split']} "
        f"with MAE: {best_mae:.2f}"
    )

    # Persist tuned RF hyperparameters for later
    os.makedirs(os.path.dirname(PARAMS_FILE_RF), exist_ok=True)
    with open(PARAMS_FILE_RF, "wb") as f:
        pickle.dump(best_params, f)
    print(f"✔ Saved tuned RF hyperparameters → {PARAMS_FILE_RF}")


def train_and_evaluate(retrain=False):
    """
    Train (or load) a Random Forest model and evaluate its performance.
    """
    # Prepare data
    X_train, X_test, y_train, y_test = preprocess_for_training(
        "CPII_RealEstate/data/house_data.csv"
    )

    # Load existing model?
    if not retrain and os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "rb") as f:
            forest = pickle.load(f)
        print("Loaded Random Forest model from file.")
    else:
        # Load tuned hyperparameters if available, else use defaults
        if os.path.exists(PARAMS_FILE_RF):
            with open(PARAMS_FILE_RF, "rb") as f:
                params = pickle.load(f)
            print(f"✔ Loaded tuned RF hyperparameters: {params}")
        else:
            params = {
                "n_estimators":     200,
                "max_depth":        None,
                "min_sample_split": 10
            }
            print(f"⚠ No tuned RF params found; using defaults: {params}")

        # Train with those params
        forest = RandomForest(**params)
        forest.fit_with_timing(X_train, y_train)

        # Persist the trained model
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        with open(MODEL_PATH, "wb") as f:
            pickle.dump(forest, f)
        print(f"✔ Trained and saved Random Forest model → {MODEL_PATH}")

    # Evaluate
    preds = forest.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


if __name__ == "__main__":
    retrain_flag = "--retrain" in sys.argv
    if "--tune" in sys.argv:
        tune_and_evaluate()
    else:
        train_and_evaluate(retrain=retrain_flag)
