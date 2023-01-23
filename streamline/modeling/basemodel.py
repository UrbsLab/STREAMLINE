from abc import ABC


class BaseModel(ABC):
    def __init__(self):
        self.name = None
        self.model = None

    def fit(self, X, y):
        pass

    def predict(self, X):
        pass
