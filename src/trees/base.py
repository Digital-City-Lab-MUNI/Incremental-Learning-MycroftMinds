"""Abstract base class for regression tree learners.

This module defines the minimal learner interface used across tree
implementations in this project. Concrete learners should implement ``fit``
and ``predict`` while inheriting common behavior via this base.
"""

import numpy as np


class BaseLearner:
    """Base class for regression tree learners."""

    def __init__(self) -> None:
        """Initialize the base learner."""
        pass

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
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

    def predict(self, X: np.ndarray) -> float:
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
