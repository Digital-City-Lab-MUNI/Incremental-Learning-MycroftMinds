"""General utility helpers.

This module groups lightweight helper functions used across evaluation and
model training code paths. They deliberately avoid adding heavy dependencies
and keep focused, testable behavior.

Functions
---------
get_unique_random_numbers
    Sample N distinct random integers from range(upper_bound) as numpy array.
ensure_2d_array
    Guarantee an ``np.ndarray`` has 2 dimensions (row vector if originally 1D).
aggregate_values
    Resample and aggregate two aligned time series with identical indices.
get_device_data
    Filter a split ``(X, y)`` array pair for a single device id; converts
    labels to a ``pandas.Series`` indexed by timestamp.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Sequence

# Registry for the most recent numeric dataset load (src.data.io.load_dataset):
# original device labels behind the numeric codes, and the source timezone behind
# the epoch-second time column (so resampling keeps the original local-time
# bucketing, e.g. Europe/Prague days around DST).
_DEVICE_LABELS: list = []
_TIMEZONE = None


def register_device_labels(labels: Sequence) -> None:
    """Register the original device labels for the most recent numeric dataset load."""
    _DEVICE_LABELS[:] = list(labels)


def register_timezone(tz) -> None:
    """Register the source timezone for the most recent numeric dataset load."""
    global _TIMEZONE
    _TIMEZONE = tz


def device_label(code):
    """Map a numeric device code back to its original label (identity if unknown)."""
    try:
        i = int(code)
    except (TypeError, ValueError):
        return code
    if 0 <= i < len(_DEVICE_LABELS):
        return _DEVICE_LABELS[i]
    return code


def as_timestamp(value):
    """Interpret a time-column value as ``pd.Timestamp``.

    Numeric values are epoch seconds (the layout produced by
    ``src.data.io.load_dataset``), converted back to the registered source
    timezone; anything else (e.g. a ``pd.Timestamp`` in a raw object-dtype array)
    is passed through ``pd.Timestamp`` directly.
    """
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if _TIMEZONE is not None:
            return pd.Timestamp(value, unit="s", tz="UTC").tz_convert(_TIMEZONE)
        return pd.Timestamp(value, unit="s")
    return pd.Timestamp(value)


def time_index(col: np.ndarray) -> pd.DatetimeIndex:
    """Build a ``DatetimeIndex`` from a time column (epoch seconds or timestamps)."""
    if col.dtype == object:
        try:
            return pd.DatetimeIndex(col)
        except Exception:
            return pd.Index(col)
    idx = pd.DatetimeIndex(pd.to_datetime(col, unit="s", utc=True))
    if _TIMEZONE is not None:
        idx = idx.tz_convert(_TIMEZONE)
    return idx


def get_unique_random_numbers(N: int, upper_bound: int, seed: int | None = None) -> np.ndarray:
    """Sample ``N`` distinct random integers from ``range(upper_bound)`` with optional random seed."""
    if N > upper_bound:
        raise ValueError("N must be less than or equal to upper_bound")

    rng = np.random.default_rng(seed)  # Creates a local Generator instance
    return rng.choice(upper_bound, size=N, replace=False)


def ensure_2d_array(X: np.ndarray | Sequence[float]) -> np.ndarray:
    """Ensure that input is 2D. Converts the input to a NumPy array and reshapes 1D arrays to ``(1, n_features)``."""
    X = np.asarray(X)
    if X.ndim == 1:
        X = X.reshape(1, -1)
    return X


def aggregate_values(x1: pd.Series, x2: pd.Series, freq: str, aggregation: str = "mean") -> Tuple[pd.Series, pd.Series]:
    """Resample and aggregate two aligned timestamped series.

    Both input series are (lazily) coerced to have a ``DatetimeIndex`` then
    resampled with the provided frequency and aggregation function.

    Parameters
    ----------
    x1 : pandas.Series
        First time series.
    x2 : pandas.Series
        Second time series (same original index alignment expected as ``x1``).
    freq : str
        Pandas offset alias specifying the new sampling frequency (e.g. '5T',
        '15min', 'H').
    aggregation : str, default 'mean'
        Name of an aggregation function understood by ``Series.agg`` (e.g.
        'mean', 'sum', 'max').

    Returns
    -------
    tuple[pandas.Series, pandas.Series]
        Two resampled series in the order ``(x1_resampled, x2_resampled)``.
    """
    # Ensure the series are datetime indexed
    if not isinstance(x1.index, pd.DatetimeIndex):
        x1.index = pd.to_datetime(x1.index)
    if not isinstance(x2.index, pd.DatetimeIndex):
        x2.index = pd.to_datetime(x2.index)

    # Resample and aggregate the series
    return x1.resample(freq).agg(aggregation), x2.resample(freq).agg(aggregation)


def get_device_data(
    data_split: Tuple[np.ndarray, np.ndarray], device_id: int | str
) -> Tuple[np.ndarray, pd.Series] | Tuple[None, None]:
    """Filter a split for a single device.

    Expects ``data_split`` to be a tuple ``(X, y)`` where ``X`` has timestamp in
    column 0 and device id in column 1. Rows matching ``device_id`` are
    retained and ``y`` is converted to a Series indexed by the timestamp.

    Parameters
    ----------
    data_split : tuple[numpy.ndarray, numpy.ndarray]
        Pair ``(X, y)`` with shape ``X[:, 0] -> timestamp`` and ``X[:, 1] ->
        device id``.
    device_id : int or str
        Device identifier to filter for.

    Returns
    -------
    tuple[numpy.ndarray, pandas.Series] | tuple[None, None]
        Filtered feature matrix and target series. If the input structure is
        invalid (not a 2-tuple) returns ``(None, None)``.
    """
    if isinstance(data_split, tuple) and len(data_split) == 2:
        X, y = data_split
        mask = X[:, 1] == device_id
        if X.dtype == object:
            y_series = pd.Series(y[mask], index=X[mask, 0])
        else:
            # numeric layout: column 0 holds epoch seconds -> proper DatetimeIndex
            y_series = pd.Series(y[mask], index=time_index(X[mask, 0]))
        return X[mask], y_series

    return None, None
