from .decision_tree import DecisionTree
import numpy as np
from .Baseclass import Model
class GradientBoostingRegressor(Model):
    def __init__(self, n_estimators=10, learning_rate=0.1, max_depth=3, min_sample_split=2):
        super().__init__(name="Gradient Boosting")
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.trees = []
        self.initial_prediction = None

    def fit(self, X, y):
        # Start with the mean as the initial prediction
        self.initial_prediction = y.mean()
        residuals = y - self.initial_prediction

        for _ in range(self.n_estimators):
            tree = DecisionTree(min_sample_split=self.min_sample_split, max_depth=self.max_depth)
            tree.fit(X, residuals)
            predictions = tree.predict(X)
            residuals -= self.learning_rate * predictions
            self.trees.append(tree)

    def predict(self, X):
        y_pred = np.full(X.shape[0], self.initial_prediction)
        for tree in self.trees:
            y_pred += self.learning_rate * tree.predict(X)
        return y_pred
