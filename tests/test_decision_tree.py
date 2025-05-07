

import numpy as np
import pytest
from CPII_RealEstate.models.decision_tree import DecisionTree
from sklearn.metrics import r2_score

@pytest.fixture
def simple_dt_model():
    # y = 4x + 3 + small noise
    rng = np.random.RandomState(1)
    X = rng.rand(50, 1)
    y = 4 * X.flatten() + 3 + rng.normal(scale=0.05, size=50)
    model = DecisionTree(max_depth=3)
    model.fit_with_timing(X, y)
    return model, X, y

def test_import():
    # ensure the class is importable
    assert DecisionTree is not None

def test_max_depth_attribute(simple_dt_model):
    model, X, y = simple_dt_model
    # check that the max_depth parameter was stored
    assert hasattr(model, 'max_depth')
    assert model.max_depth == 3

def test_predict_output_type_and_shape(simple_dt_model):
    model, X, y = simple_dt_model
    preds = model.predict(X)
    # predictions should be a numpy array of the same length as y
    assert isinstance(preds, np.ndarray)
    assert preds.shape == y.shape

def test_score_consistency(simple_dt_model):
    model, X, y = simple_dt_model
    preds = model.predict(X)
    score_method = model.score(X, y)
    # compare against sklearn's r2_score
    expected = r2_score(y, preds)
    assert pytest.approx(expected, rel=1e-4) == score_method