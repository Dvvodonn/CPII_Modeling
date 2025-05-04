import numpy as np
import sys
import os
import pickle

# Ensure parent directory is in path for local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.data_preprocessing import preprocess_for_training
from models.decision_tree import DecisionTree
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_decision_tree(retrain=False):
    X_train, X_test, y_train, y_test = preprocess_for_training("data/house_data.csv")

    if not retrain and os.path.exists("outputs/tree_model.pkl"):
        with open("outputs/tree_model.pkl", "rb") as f:
            tree = pickle.load(f)
        print("Loaded model from file.")
    else:
        tree = DecisionTree(max_depth=8, min_sample_split=10)
        tree.fit_with_timing(X_train, y_train)
        with open("outputs/tree_model.pkl", "wb") as f:
            pickle.dump(tree, f)

    preds = tree.predict(X_test)
    print("Sample Predictions:", preds)

    try:
        tree.root.print_tree()
    except Exception as e:
        print(f"Error printing tree: {e}")

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

if __name__ == "__main__":
    test_decision_tree(retrain="--retrain" in sys.argv)