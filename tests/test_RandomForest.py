import numpy as np
import pytest

from CPII_RealEstate.models.random_forest import RandomForest

@pytest.fixture
def simple_rf_model():
    # create synthetic data: y = 2.5*x0 - 1.0*x1 + noise
    rng = np.random.RandomState(0)
    X = rng.rand(30, 4)
    y = X[:, 0] * 2.5 + X[:, 1] * (-1.0) + rng.normal(scale=0.1, size=30)
    model = RandomForest(n_estimators=10, max_depth=5)
    model.fit_with_timing(X, y)
    return model, X, y

def test_import():
    # ensure the class is importable
    assert RandomForest is not None

def test_n_estimators_attribute(simple_rf_model):
    model, X, y = simple_rf_model
    # check that the n_estimators parameter was stored
    assert hasattr(model, 'n_estimators')
    assert model.n_estimators == 10

def test_predict_output_type_and_shape(simple_rf_model):
    model, X, y = simple_rf_model
    preds = model.predict(X)
    # predictions should be a numpy array of the same length as y
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape

def test_score_between_0_and_1(simple_rf_model):
    model, X, y = simple_rf_model
    score = model.score(X, y)
    # R^2 score should be between 0 and 1 on training data
    assert 0.0 <= score <= 1.0

def test_performance_mae(simple_rf_model):
    model, X, y = simple_rf_model
    preds = model.predict(X)
    mae = np.mean(np.abs(preds - y))
    # with low noise, mean absolute error should be small
    assert mae < 0.5
