class BaseLeaner:

    def __init__(self):
        pass

    def fit(self, X, y):
        raise NotImplementedError

    def predict(self, X) -> float:
        raise NotImplementedError
