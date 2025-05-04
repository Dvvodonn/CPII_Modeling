

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