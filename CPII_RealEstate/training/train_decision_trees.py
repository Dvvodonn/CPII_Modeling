import numpy as np
import sys
import os
import pickle
import itertools

# Storing params
PARAMS_FILE = "CPII_RealEstate/training/best_params/dt_best_params.pkl"


# Ensure parent directory is in path for local imports
from CPII_RealEstate.models.decision_tree import DecisionTree
from CPII_RealEstate.utils.data_preprocessing import preprocess_for_training
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def tune_and_evaluate():
    """
    Perform a simple grid search over max_depth and min_sample_split
    for the DecisionTree, printing MAE for each combination.
    """
    # Load data
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")
    # Define parameter grid
    param_grid = {
        'max_depth': [10, 20, 30],
        'min_sample_split': [10,20,30]
    }
    best_mae = float('inf')
    best_params = {}
    # Grid search
    for depth, mss in itertools.product(param_grid['max_depth'], param_grid['min_sample_split']):
        tree = DecisionTree(max_depth=depth, min_sample_split=mss)
        tree.fit(X_train, y_train)
        preds = tree.predict(X_test)
        mae = mean_absolute_error(y_test, preds)
        print(f"Params: max_depth={depth}, min_sample_split={mss} -> MAE: {mae:.2f}")
        if mae < best_mae:
            best_mae = mae
            best_params = {"max_depth": depth, "min_sample_split": mss}
    print(f"\nBest params: max_depth={best_params['max_depth']}, "
            f"min_sample_split={best_params['min_sample_split']} "
            f"with MAE: {best_mae:.2f}")
    # persist tuned hyperparameters for later
    os.makedirs(os.path.dirname(PARAMS_FILE), exist_ok=True)
    with open(PARAMS_FILE, "wb") as f:
       pickle.dump(best_params, f)
    print(f" Saved tuned hyperparameters → {PARAMS_FILE}")
    
def train_decision_tree(retrain=False):
    X_train, X_test, y_train, y_test = preprocess_for_training("CPII_RealEstate/data/house_data.csv")
    if not retrain and os.path.exists("CPII_RealEstate/outputs/tree_model.pkl"):
        with open("CPII_RealEstate/outputs/tree_model.pkl", "rb") as f:
            tree = pickle.load(f)
        print("Loaded model from file.")
    else:
        # load tuned hyperparameters if available, else use defaults
        if os.path.exists(PARAMS_FILE):
            with open(PARAMS_FILE, "rb") as f:
                params = pickle.load(f)
            print(f"Loaded tuned hyperparameters: {params}")
        else:
            params = {"max_depth": 20, "min_sample_split": 4}
            print(f"No tuned params found; using defaults: {params}")
        tree = DecisionTree(**params)
        tree.fit_with_timing(X_train, y_train)
        with open("CPII_RealEstate/outputs/tree_model.pkl", "wb") as f:
            pickle.dump(tree, f)

    preds = tree.predict(X_test)
    print("Sample Predictions:", preds)

    """
    try:
        tree.root.print_tree()
    except Exception as e:
        print(f"Error printing tree: {e}")
    """

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
        train_decision_tree(retrain=retrain_flag)
