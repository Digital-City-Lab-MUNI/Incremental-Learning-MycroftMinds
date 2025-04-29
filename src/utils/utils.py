import pandas as pd
import numpy as np
from typing import Tuple

def aggregate_values(x1: pd.Series, x2: pd.Series, freq: str) -> Tuple[pd.Series, pd.Series]:
    """
    Aggregate values of two series based on a given frequency.
    Args:
        x1 (pd.Series): First series to aggregate.
        x2 (pd.Series): Second series to aggregate.
        freq (str): Frequency string for resampling (e.g., '5T' for 5 minutes).
    Returns:
        Tuple[pd.Series, pd.Series]: Aggregated series.
    """
    # Ensure the series are datetime indexed
    if not isinstance(x1.index, pd.DatetimeIndex):
        x1.index = pd.to_datetime(x1.index)
    if not isinstance(x2.index, pd.DatetimeIndex):
        x2.index = pd.to_datetime(x2.index)
    # Resample and aggregate the series
    
    return x1.resample(freq).sum(), x2.resample(freq).sum()

def get_device_data(data_split: Tuple[np.ndarray, np.ndarray],
                    device_id) -> Tuple[np.ndarray, np.ndarray] | Tuple[None, None]:
    """Extract data for a specific device from a data split. Turns the y values into a pandas Series with the first column of X as the index."""
    
    if isinstance(data_split, tuple) and len(data_split) == 2:
        X, y = data_split
        mask = X[:, 1] == device_id
        y_series = pd.Series(y[mask], index=X[mask, 0])
        return X[mask], y_series
    
    return None, None