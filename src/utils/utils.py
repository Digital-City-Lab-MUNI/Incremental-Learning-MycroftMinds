import pandas as pd
import numpy as np
import random
from typing import Tuple

def get_unique_random_numbers(N, M):
    if N > M:
        raise ValueError("N must be less than or equal to M")
    return random.sample(range(M), N)

def ensure_2d_array(X: np.ndarray) -> np.ndarray:
    """
    Ensure that the input is a 2D numpy array. If the input is 1D, it reshapes it to 2D.
    """
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X

def aggregate_values(x1: pd.Series, x2: pd.Series, freq: str, aggregation: str = 'mean') -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate values of two series based on a given frequency.
    Args:
        x1 (pd.Series): First series to aggregate.
        x2 (pd.Series): Second series to aggregate.
        freq (str): Frequency string for resampling (e.g., '5T' for 5 minutes).
        aggregation (str): Name of the aggregation function to apply. Must be supported by `pandas` (e.g., 'mean', 'sum'). Default is 'mean'.
    Returns:
        Tuple[pd.Series, pd.Series]: Aggregated series.
    """
    # Ensure the series are datetime indexed
    if not isinstance(x1.index, pd.DatetimeIndex):
        x1.index = pd.to_datetime(x1.index)
    if not isinstance(x2.index, pd.DatetimeIndex):
        x2.index = pd.to_datetime(x2.index)
    
    # Resample and aggregate the series
    return x1.resample(freq).agg(aggregation), x2.resample(freq).agg(aggregation)


def get_device_data(data_split: Tuple[np.ndarray, np.ndarray],
                    device_id) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Extract data for a specific device from a data split. Turns the y values into a pandas Series with the first column of X as the index."""
    
    if isinstance(data_split, tuple) and len(data_split) == 2:
        X, y = data_split
        mask = X[:, 1] == device_id
        y_series = pd.Series(y[mask], index=X[mask, 0])
        return X[mask], y_series
    
    return None, None