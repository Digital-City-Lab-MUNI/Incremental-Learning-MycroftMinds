# This file now acts as a wrapper to maintain backward compatibility
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

# Import metrics functions (except aggregate_values which we kept here)
from src.metrics import (
    calculate_metrics,
    all_time_aggregation_metrics
)

# Import visualization functions
from src.visualization import (
    plot_predictions,
    plot_hourly_predictions
)

# Import training workflow functions
from src.training import (
    seq_fit_predict_batch_model,
    seq_fit_predict_incremental_model
)

# Import memory functions for backward compatibility with code accessing utils.print_memory_stats
from src.memory import calculate_memory_usage, print_memory_stats

