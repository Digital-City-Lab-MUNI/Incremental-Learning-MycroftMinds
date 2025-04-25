import pandas as pd
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
