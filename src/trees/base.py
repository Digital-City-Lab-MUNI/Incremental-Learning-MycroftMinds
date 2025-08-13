class BaseLeaner:
    """
    Base class for regression tree learners.
    """

    def __init__(self):
        """
        Initialize the base learner.
        """
        pass

    def fit(self, X, y):
        """
        Fit the model to the data.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        y : array-like
            Target vector.
        """
        raise NotImplementedError

    def predict(self, X) -> float:
        """
        Predict target values for X.

        Parameters
        ----------
        X : array-like
            Feature matrix or single sample.

        Returns
        -------
        float
            Predicted value(s).
        """
        raise NotImplementedError
