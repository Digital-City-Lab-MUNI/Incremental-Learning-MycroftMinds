import numpy as np
import pandas as pd


class PreviousValueRegressor:
    """
    Baseline regressor that always predicts the previous observed value.
    Simple incremental learning model that remembers only the last seen value.
    """
    def __init__(self):
        self.last_value = 0.0  # Default value when no data has been seen

    def learn_one(self, x, y):
        """River compatibility - learn from one sample"""
        self.last_value = y
        return self

    def predict_one(self, x):
        """River compatibility - predict for one sample"""
        return self.last_value
