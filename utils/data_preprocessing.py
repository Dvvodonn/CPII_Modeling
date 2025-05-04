import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

def load_data(filepath: str):
    return pd.read_csv(filepath)

def clean_data(df: pd.DataFrame):
    df = df.copy()
    df.drop(columns=['waterfront','sqft_lot15','sqft_living15','long','lat','zipcode','sqft_basement','sqft_above','date','id'], errors='ignore')
    df = df.dropna()
    return df

def split_features_targets(df: pd.DataFrame):
    X = df.drop(columns=['price']).to_numpy()
    y = df['price'].to_numpy()
    return X,y

def preprocess_for_training(filepath: str):
    df = load_data(filepath)
    df = clean_data(df)
    X, y = split_features_targets(df)
    return train_test_split(X, y, test_size=0.2, random_state=42)


