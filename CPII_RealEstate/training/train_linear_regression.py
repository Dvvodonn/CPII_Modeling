

#!/usr/bin/env python3

import sys
import os
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from CPII_RealEstate.models.lin_reg import LinearRegression

def train_and_evaluate(retrain=False):
    """
    Train or load a Linear Regression model and evaluate its performance.
    """
    # Load and preprocess data from the installed package
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")

    # Model persistence path within the package outputs folder
    model_path = "CPII_RealEstate/outputs/linear_regression_model.pkl"

    # Load existing model if not retraining
    if not retrain and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            lr = pickle.load(f)
        print("Loaded Linear Regression model from file.")
    else:
        # Instantiate and train the model with timing
        lr = LinearRegression()
        lr.fit_with_timing(X_train, y_train)
        # Ensure the outputs directory exists
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # Save the trained model
        with open(model_path, "wb") as f:
            pickle.dump(lr, f)
        print("Trained and saved Linear Regression model.")

    # Make predictions on the test set
    preds = lr.predict(X_test)

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
    train_and_evaluate(retrain=retrain_flag)