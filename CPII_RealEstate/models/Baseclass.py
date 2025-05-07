
from abc import ABC, abstractmethod
import time

class Model(ABC):
    def __init__(self, name="UnnamedModel"):
        self.name = name

    @abstractmethod
    def fit(self, X, y):
        pass

    @abstractmethod
    def predict(self, X):
        pass

    def fit_with_timing(self, X, y):
        start = time.time()
        self.fit(X, y)
        print(f"{self.name} training time: {time.time() - start:.2f} seconds")

    def score(self, X, y):
        """
        Compute coefficient of determination R²:
          R² = 1 - (SS_res / SS_tot)
        where SS_res = Σ(y_i - ŷ_i)² and SS_tot = Σ(y_i - ȳ)².
        """
        preds = self.predict(X)
        ss_res = ((y - preds) ** 2).sum()
        ss_tot = ((y - y.mean()) ** 2).sum()
        return 1 - ss_res / ss_tot if ss_tot != 0 else 0.0