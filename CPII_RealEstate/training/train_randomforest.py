#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
import itertools
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from CPII_RealEstate.models.random_forest import RandomForest

def tune_and_evaluate():
    """
    Run a grid search over n_estimators, max_depth, and min_sample_split,
    printing MAE for each combination and the best parameters.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")
    # Define parameter grid
    param_grid = {
        'n_estimators': [150,200],
        'max_depth': [14,16],
        'min_sample_split': [10]
    }
    best_mae = float('inf')
    best_params = None
    # Iterate over grid
    for n, d, m in itertools.product(param_grid['n_estimators'],
                                      param_grid['max_depth'],
                                      param_grid['min_sample_split']):
        forest = RandomForest(n_estimators=n, max_depth=d, min_sample_split=m)
        forest.fit_with_timing(X_train, y_train)
        preds = forest.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Params: n_estimators={n}, max_depth={d}, min_sample_split={m} -> MAE: {mae:.2f}")
        if mae < best_mae:
            best_mae = mae
            best_params = (n, d, m)
    print(f"\nBest params: n_estimators={best_params[0]}, max_depth={best_params[1]}, "
          f"min_sample_split={best_params[2]} with MAE: {best_mae:.2f}")

def train_and_evaluate(retrain=False):
    """
    Train or load a Random Forest model and evaluate its performance.
    """
    # Prepare data
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")
    
    # Model persistence path
    model_path = "CPII_RealEstate/outputs/random_forest_model.pkl"
    
    # Load or train
    if not retrain and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            forest = pickle.load(f)
        print("Loaded Random Forest model from file.")
    else:
        forest = RandomForest(n_estimators=125, max_depth=10, min_sample_split=10)
        forest.fit_with_timing(X_train, y_train)
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        with open(model_path, "wb") as f:
            pickle.dump(forest, f)
        print("Trained and saved Random Forest model.")
    
    # Predict and evaluate
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