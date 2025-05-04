import os
import argparse
import pandas as pd
import numpy as np
from lin_reg import LinearRegression
from gradientboost import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Argument parsing for verbose flag
def get_args():
    parser = argparse.ArgumentParser(description="Compare Linear vs Gradient Boosting models")
    parser.add_argument(
        "--verbose", type=bool, default=True,
        help="Enable detailed logging (True/False)"
    )
    return parser.parse_args()

args = get_args()
VERBOSE = args.verbose

# Logging helper
def log(msg):
    if VERBOSE:
        print(msg)

# 1. Load dataset
log("Loading data from house_data.csv...")
df = pd.read_csv('house_data.csv')
log(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
log("Columns: " + ", ".join(df.columns))

# 2. Basic inspection
if VERBOSE:
    log("\nFirst 5 rows:")
    print(df.head())

# 3. Preprocess: drop non-numeric fields
if 'date' in df.columns:
    log("Dropping 'date' for numeric conversion...")
    df['date'] = pd.to_datetime(df['date'], format='%Y%m%dT%H%M%S')
    df['year']  = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df.drop('date', axis=1, inplace=True)

# 4. Prepare features and target
X = df.drop('price', axis=1).to_numpy(dtype=float)
y = df['price'].to_numpy(dtype=float)
log(f"Feature matrix X shape: {X.shape}\nTarget vector y shape: {y.shape}\n")

# 5. Split into train/test
Xtr, Xt, ytr, yt = train_test_split(X, y, test_size=0.2, random_state=42)
log(f"Train set: X={Xtr.shape}, y={ytr.shape}")
log(f"Test set:  X={Xt.shape}, y={yt.shape}\n")

# 6. Linear Regression Model
log("Training Linear Regression...")
lr = LinearRegression()
lr.fit(Xtr, ytr)
lr_pred = lr.predict(Xt)
log("Linear Regression done.\n")

# 7. Gradient Boosting Model
log("Training Gradient Boosting...")
gb = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=3)
gb.fit(Xtr, ytr)
gb_pred = gb.predict(Xt)
log("Gradient Boosting done.\n")

# 8. Evaluation
print("Model      MSE       R2")
for name, pred in [("LinearReg", lr_pred), ("GradBoost", gb_pred)]:
    mse = mean_squared_error(yt, pred)
    r2 = r2_score(yt, pred)
    print(f"{name:10s} {mse:8.2f} {r2:8.3f}")

# 9. Sample predictions
if VERBOSE:
    print("\nSample preds vs actual:")
    for i in range(min(5, len(yt))):
        print(f"{i}: Actual={yt[i]:.2f}, LR={lr_pred[i]:.2f}, GB={gb_pred[i]:.2f}")
