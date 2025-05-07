import numpy as np
from .decision_tree import DecisionTree
from .Baseclass import Model

class RandomForest(Model):
    def __init__(self, n_estimators=100, max_depth=None, min_sample_split=2):
        super().__init__(name="RandomForest")
        """
        Initialize the random forest with given parameters.
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_sample_split = min_sample_split
        self.trees = []

    def _bootstrap_sample(self,X,y):
        """
        Sample from data set randomly with replacement
        """
        n_samples = X.shape[0]
        indecies = np.random.choice(n_samples, size=n_samples, replace=True)
        return X[indecies], y[indecies]
    
    def fit(self,X,y):
        """
        Train trees based of bootstrapped data
        """
        self.trees = []
        for _ in range(self.n_estimators):
            X_sample,y_sample = self._bootstrap_sample(X,y)
            tree = DecisionTree(min_sample_split=self.min_sample_split, max_depth=self.max_depth)
            tree.fit(X_sample,y_sample)
            self.trees.append(tree)

    def predict(self,X):
        tree_preds = np.array([tree.predict(X) for tree in self.trees])
        return np.mean(tree_preds,axis=0)