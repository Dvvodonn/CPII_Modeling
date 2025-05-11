#!/usr/bin/env python3

import sys
import os
import pickle
import itertools
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from CPII_RealEstate.models.gradientboost import GradientBoostingRegressor

# where we’ll stash (and later reload) tuned GBM hyperparameters
PARAMS_FILE_GBM = "CPII_RealEstate/training/best_params/gbm_best_params.pkl"
# where we save/load the final trained model
MODEL_PATH_GBM  = "CPII_RealEstate/outputs/gradient_boosting_model.pkl"


def tune_and_evaluate():
    """
    Run a grid search over hyperparameters for GradientBoostingRegressor,
    print MAE for each combo, and persist the best parameters.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training(
        "CPII_RealEstate/data/house_data.csv"
    )

    # Parameter grid
    param_grid = {
        'n_estimators':      [150, 200, 300],
        'learning_rate':     [0.01, 0.05, 0.1],
        'max_depth':         [3, 5, 7],
        'min_sample_split':  [2, 10, 15]
    }

    best_mae = float('inf')
    best_params = {}

    # Grid search
    for ne, lr, md, mss in itertools.product(
        param_grid['n_estimators'],
        param_grid['learning_rate'],
        param_grid['max_depth'],
        param_grid['min_sample_split']
    ):
        gbm = GradientBoostingRegressor(
            n_estimators=ne,
            learning_rate=lr,
            max_depth=md,
            min_sample_split=mss
        )
        gbm.fit_with_timing(X_train, y_train)
        preds = gbm.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Params: n_estimators={ne}, learning_rate={lr}, "
              f"max_depth={md}, min_sample_split={mss} -> MAE: {mae:.2f}")

        if mae < best_mae:
            best_mae = mae
            best_params = {
                "n_estimators":     ne,
                "learning_rate":    lr,
                "max_depth":        md,
                "min_sample_split": mss
            }

    print(
        f"\nBest params: n_estimators={best_params['n_estimators']}, "
        f"learning_rate={best_params['learning_rate']}, "
        f"max_depth={best_params['max_depth']}, "
        f"min_sample_split={best_params['min_sample_split']} "
        f"with MAE: {best_mae:.2f}"
    )

    # Persist tuned GBM hyperparameters
    os.makedirs(os.path.dirname(PARAMS_FILE_GBM), exist_ok=True)
    with open(PARAMS_FILE_GBM, "wb") as f:
        pickle.dump(best_params, f)
    print(f"✔ Saved tuned GBM hyperparameters → {PARAMS_FILE_GBM}")


def train_and_evaluate(retrain=False):
    """
    Train (or load) a Gradient Boosting model and evaluate performance.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training(
        "CPII_RealEstate/data/house_data.csv"
    )

    # Load existing model?
    if not retrain and os.path.exists(MODEL_PATH_GBM):
        with open(MODEL_PATH_GBM, "rb") as f:
            gbm = pickle.load(f)
        print("Loaded Gradient Boosting model from file.")
    else:
        # Load tuned hyperparameters if available, else use defaults
        if os.path.exists(PARAMS_FILE_GBM):
            with open(PARAMS_FILE_GBM, "rb") as f:
                params = pickle.load(f)
            print(f"✔ Loaded tuned GBM hyperparameters: {params}")
        else:
            params = {
                "n_estimators":     50,
                "learning_rate":    0.05,
                "max_depth":        5,
                "min_sample_split": 10
            }
            print(f"⚠ No tuned GBM params found; using defaults: {params}")

        # Train with those parameters
        gbm = GradientBoostingRegressor(**params)
        gbm.fit_with_timing(X_train, y_train)

        # Persist the trained model
        os.makedirs(os.path.dirname(MODEL_PATH_GBM), exist_ok=True)
        with open(MODEL_PATH_GBM, "wb") as f:
            pickle.dump(gbm, f)
        print(f"✔ Trained and saved Gradient Boosting model → {MODEL_PATH_GBM}")

    # Evaluate
    preds = gbm.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2  = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")


if __name__ == "__main__":
    retrain_flag = "--retrain" in sys.argv
    if "--tune" in sys.argv:
        tune_and_evaluate()
    else:
        train_and_evaluate(retrain=retrain_flag)
