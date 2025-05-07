#!/usr/bin/env python3

import sys
import os
import pickle
import itertools
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from CPII_RealEstate.models.gradientboost import GradientBoostingRegressor

def tune_and_evaluate():
    """
    Run a grid search over hyperparameters for GradientBoostingRegressor.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")
    # Parameter grid
    param_grid = {
        'n_estimators':      [50, 100, 150, 200],
        'learning_rate':     [0.01, 0.05, 0.1],
        'max_depth':         [3, 5, 7],
        'min_sample_split':  [2, 5, 10]
    }
    best_mae = float('inf')
    best_params = None

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
        preds = gbm.predict(X_test)  # use features, not y_test
        mae = mean_absolute_error(y_test, preds)
        print(f"Params: n_estimators={ne}, learning_rate={lr}, max_depth={md}, min_sample_split={mss} -> MAE: {mae:.2f}")
        if mae < best_mae:
            best_mae = mae
            best_params = (ne, lr, md, mss)

    print(f"\nBest params: n_estimators={best_params[0]}, learning_rate={best_params[1]}, "
          f"max_depth={best_params[2]}, min_sample_split={best_params[3]} with MAE: {best_mae:.2f}")


def train_and_evaluate(retrain=False):
    """
    Train or load a Gradient Boosting model and evaluate its performance.
    """
    # Load and preprocess data from the installed package
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")

    # Model persistence path within the package outputs folder
    model_path = "CPII_RealEstate/outputs/gradient_boosting_model.pkl"

    # Load existing model if not retraining
    if not retrain and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            gbm = pickle.load(f)
        print("Loaded Gradient Boosting model from file.")
    else:
        # Instantiate with tuned parameters
        gbm = GradientBoostingRegressor(
            n_estimators=50,
            learning_rate=0.05,
            max_depth=5,
            min_sample_split=10
        )
        # Train with timing
        gbm.fit_with_timing(X_train, y_train)
        # Ensure the outputs directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save the trained model
        with open(model_path, "wb") as f:
            pickle.dump(gbm, f)
        print("Trained and saved Gradient Boosting model.")

    # Make predictions on the test set
    preds = gbm.predict(X_test)

    # Evaluate performance
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