import numpy as np
import pytest
from CPII_RealEstate.models.gradientboost import GradientBoostingRegressor
from sklearn.metrics import r2_score

@pytest.fixture
def simple_gb_model():
    # y = 3x + 10 + noise
    rng = np.random.RandomState(42)
    X = np.linspace(0, 10, 100).reshape(-1, 1)
    y = 3 * X.flatten() + 10 + rng.normal(0, 1, size=100)
    model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=2)
    model.fit_with_timing(X, y)
    return model, X, y

def test_import():
    # ensure the class is importable
    assert GradientBoostingRegressor is not None

def test_prediction_shape(simple_gb_model):
    model, X, y = simple_gb_model
    preds = model.predict(X)
    # predictions should match input shape
    assert preds.shape == y.shape

def test_r2_score(simple_gb_model):
    model, X, y = simple_gb_model
    preds = model.predict(X)
    score = r2_score(y, preds)
    # on low-noise linear data, expect high R^2
    assert score > 0.8
