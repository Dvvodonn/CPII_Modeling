import unittest
import numpy as np
from models.gradientboost import GradientBoostingRegressor

class TestGradientBoosting(unittest.TestCase):

    def setUp(self):
        # y = 3x + 10 + noise
        np.random.seed(42)
        self.X = np.linspace(0, 10, 100).reshape(-1, 1)
        self.y = 3 * self.X.flatten() + 10 + np.random.normal(0, 1, 100)
        self.model = GradientBoostingRegressor(n_estimators=10, learning_rate=0.1, max_depth=2)
        self.model.fit(self.X, self.y)

    def test_prediction_shape(self):
        preds = self.model.predict(self.X)
        self.assertEqual(preds.shape, self.y.shape)

    def test_r2_score(self):
        from sklearn.metrics import r2_score
        preds = self.model.predict(self.X)
        score = r2_score(self.y, preds)
        self.assertGreater(score, 0.9)

if __name__ == '__main__':
    unittest.main()
