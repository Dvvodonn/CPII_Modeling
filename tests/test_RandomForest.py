

import sys
import os
import pickle
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from models.random_forest import RandomForest
from utils.data_preprocessing import preprocess_for_training

def test_random_forest(retrain=False):
    """
    Tests Random Forests.  --retrain to retrain model, 
    otherwise previous model will be loaded
    """
    X_train, X_test, y_train, y_test = preprocess_for_training("data/house_data.csv")
    model_path = "outputs/random_forest_model.pkl"

    if not retrain and os.path.exists(model_path):
        with open(model_path, "rb") as f:
            forest = pickle.load(f)
        print("Loaded Random Forest model from file.")
    else:
        forest = RandomForest(n_estimators=10, max_depth=8, min_sample_split=10)
        forest.fit_with_timing(X_train, y_train)
        with open(model_path, "wb") as f:
            pickle.dump(forest, f)
        print("Trained and saved Random Forest model.")

    preds = forest.predict(X_test)

    mae = mean_absolute_error(y_test, preds)
    mse = mean_squared_error(y_test, preds)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, preds)

    print("Random Forest Evaluation:")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R² Score: {r2:.2f}")

if __name__ == "__main__":
    test_random_forest(retrain="--retrain" in sys.argv)