import numpy as np
import pytest
from CPII_RealEstate.models.lin_reg import LinearRegression

@pytest.fixture
def simple_linear_model():
    # Simple linear data: y = 2x + 5
    X = np.array([[1], [2], [3], [4]])
    y = np.array([7, 9, 11, 13])  # 2*x + 5
    model = LinearRegression()
    model.fit_with_timing(X, y)
    return model, X, y

def test_coefficients(simple_linear_model):
    model, X, y = simple_linear_model
    # Check that learned coefficients are close to [bias=5, slope=2]
    np.testing.assert_almost_equal(model.coefficients, [5.0, 2.0], decimal=1)

def test_prediction(simple_linear_model):
    model, X, y = simple_linear_model
    pred = model.predict(np.array([[5]]))
    # Prediction should be approximately 15.0
    assert pytest.approx(15.0, rel=1e-1) == pred[0]

def test_r2_score(simple_linear_model):
    model, X, y = simple_linear_model
    r2 = model.score(X, y)
    # R-squared should be approximately 1.0
    assert pytest.approx(1.0, rel=1e-4) == r2
