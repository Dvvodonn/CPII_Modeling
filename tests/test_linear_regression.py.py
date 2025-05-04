import unittest
import numpy as np
from models.lin_reg import LinearRegression

class TestLinearRegression(unittest.TestCase):

    def setUp(self):
        # Simple linear data: y = 2x + 5
        self.X = np.array([[1], [2], [3], [4]])
        self.y = np.array([7, 9, 11, 13])  # 2*x + 5
        self.model = LinearRegression()
        self.model.fit(self.X, self.y)

    def test_coefficients(self):
        # Check that learned coefficients are close to [bias=5, slope=2]
        np.testing.assert_almost_equal(self.model.coefficients, [5.0, 2.0], decimal=1)

    def test_prediction(self):
        pred = self.model.predict(np.array([[5]]))
        self.assertAlmostEqual(pred[0], 15.0, places=1)

    def test_r2_score(self):
        r2 = self.model.score(self.X, self.y)
        self.assertAlmostEqual(r2, 1.0, places=4)

if __name__ == '__main__':
    unittest.main()
