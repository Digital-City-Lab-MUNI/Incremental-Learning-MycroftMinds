"""Feature engineering utilities for time series forecasting.

This module provides helpers to:

* Create time-based lag features across arbitrary temporal offsets while
    handling timezone / DST complexities via UTC alignment.
* Generate cyclical encodings (sine / cosine) for common calendar components.
* Filter feature sets so that lagged predictors respect a required forecast
    horizon.

Only documentation and typing adjustments were introduced in this pass; logic
remains unchanged.
"""

import pandas as pd
import numpy as np
import traceback
from typing import List, Dict, Tuple


def add_lag_features(df: pd.DataFrame, columns: List[str], lag_freqs: Dict[str, str]) -> pd.DataFrame:
    """Add lagged variants of selected columns at arbitrary time offsets.

    For each ``(suffix, offset)`` pair in ``lag_freqs`` a set of features is
    produced by aligning each row's UTC timestamp minus the offset to the
    nearest earlier (``direction='backward'``) observation within a tolerance
    equal to the offset duration. This approach avoids issues with daylight
    saving transitions by performing merges on UTC-normalized timestamps.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing a ``time`` column. The ``time`` column will
        be localized/converted to ``Europe/Prague`` (DST-safe) if needed.
    columns : list of str
        Base feature columns to lag.
    lag_freqs : dict
        Mapping ``suffix -> pandas Timedelta string`` (e.g. ``{"1h": "1H"}``).

    Returns
    -------
    pd.DataFrame
        Copy of the original DataFrame with additional lagged columns named
        ``{original}_{suffix}``.

    Notes
    -----
    Rows whose timestamps are ambiguous or non-existent during localization are
    removed (logged) to preserve temporal consistency.
    """
    try:
        df = df.copy()
        df["time"] = pd.to_datetime(df["time"], errors="coerce")

        # Handle timezone localization more carefully
        if df["time"].dt.tz is None:
            # If no timezone, localize to Prague timezone with DST handling
            df["time"] = df["time"].dt.tz_localize("Europe/Prague", ambiguous="NaT", nonexistent="shift_forward")
        elif df["time"].dt.tz.zone != "Europe/Prague":
            # If different timezone, convert to Prague
            df["time"] = df["time"].dt.tz_convert("Europe/Prague")

        # Add UTC time for lag alignment (this avoids DST issues in calculations)
        df["time_utc"] = df["time"].dt.tz_convert("UTC")

        # Drop rows with NaT in either time column
        initial_count = len(df)
        df = df[df["time"].notna() & df["time_utc"].notna()].copy()
        dropped_count = initial_count - len(df)

        if dropped_count > 0:
            print(f"[INFO] Dropped {dropped_count} rows due to DST ambiguity/invalid timestamps")

        df.sort_values("time_utc", inplace=True)
        df.reset_index(drop=True, inplace=True)

        # Add lag features using UTC time for calculations
        for suffix, lag_str in lag_freqs.items():
            try:
                lag_delta = pd.Timedelta(lag_str)
                df[f"time_utc_shift_{suffix}"] = df["time_utc"] - lag_delta

                # Create temp dataframe with UTC time and columns to lag
                temp = df[["time_utc"] + columns].copy()
                rename_dict = {col: f"{col}_{suffix}" for col in columns}
                temp.rename(columns=rename_dict, inplace=True)

                # Perform merge_asof using UTC times
                merged = pd.merge_asof(
                    df.sort_values(f"time_utc_shift_{suffix}"),
                    temp.sort_values("time_utc"),
                    left_on=f"time_utc_shift_{suffix}",
                    right_on="time_utc",
                    direction="backward",
                    tolerance=lag_delta,
                )

                # Add lagged columns to original dataframe
                for col in columns:
                    df[f"{col}_{suffix}"] = merged[f"{col}_{suffix}"].values

                # Clean up temporary column
                df.drop(columns=[f"time_utc_shift_{suffix}"], inplace=True)

            except Exception as e:
                print(f"[WARNING] Failed to add lag features for {suffix} ({lag_str}): {e}")
                continue

        # Clean up UTC time column
        df.drop(columns=["time_utc"], inplace=True)

        return df

    except Exception as e:
        print(f"[ERROR] Failed to add lag features: {e}")
        traceback.print_exc()
        return df


def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Encode cyclic calendar components with sine and cosine pairs.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing a timezone-aware or naive ``time`` column already
        in datetime format.

    Returns
    -------
    pd.DataFrame
        DataFrame copy with added columns: ``hour_sin``, ``hour_cos``,
        ``weekday_sin``, ``weekday_cos``, ``month_sin``, ``month_cos``.
    """
    try:
        df = df.copy()

        # Hour of day (24 hours)
        df["hour_sin"] = np.sin(2 * np.pi * df["time"].dt.hour / 24)
        df["hour_cos"] = np.cos(2 * np.pi * df["time"].dt.hour / 24)

        # Day of week (7 days)
        df["weekday_sin"] = np.sin(2 * np.pi * df["time"].dt.weekday / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["time"].dt.weekday / 7)

        # Month of year (12 months)
        df["month_sin"] = np.sin(2 * np.pi * (df["time"].dt.month - 1) / 12)
        df["month_cos"] = np.cos(2 * np.pi * (df["time"].dt.month - 1) / 12)

        # Uncomment these if additional cyclical features are needed
        # # Add minute-level precision to hour
        # hours_with_minutes = df['time'].dt.hour + df['time'].dt.minute / 60
        # df['hour_minute_sin'] = np.sin(2 * np.pi * hours_with_minutes / 24)
        # df['hour_minute_cos'] = np.cos(2 * np.pi * hours_with_minutes / 24)

        # # Day of month (up to 31 days)
        # day_of_month_max = df['time'].dt.daysinmonth
        # df['day_of_month_sin'] = np.sin(2 * np.pi * (df['time'].dt.day - 1) / day_of_month_max)
        # df['day_of_month_cos'] = np.cos(2 * np.pi * (df['time'].dt.day - 1) / day_of_month_max)

        # # Quarter of year (4 quarters)
        # df['quarter_sin'] = np.sin(2 * np.pi * (df['time'].dt.quarter - 1) / 4)
        # df['quarter_cos'] = np.cos(2 * np.pi * (df['time'].dt.quarter - 1) / 4)

        return df

    except Exception as e:
        print(f"[ERROR] Failed to add cyclical time features: {e}")
        traceback.print_exc()
        return df


def filter_features_for_forecast_horizon(
    feature_names: List[str], forecast_horizon: str
) -> Tuple[List[str], List[str]]:
    """Filter out lag features whose period is shorter than the horizon.

    Parameters
    ----------
    feature_names : list of str
        Candidate feature column names (may contain lag patterns of the form
        ``..._lag_<offset>``).
    forecast_horizon : str
        Horizon as pandas Timedelta string (e.g. ``"24h"``, ``"7d"``).

    Returns
    -------
    (list of str, list of str)
        Tuple ``(valid_features, excluded_features)``. Non-lag features are
        always retained.
    """
    forecast_td = pd.Timedelta(forecast_horizon)
    valid_features = []
    excluded_features = []

    for feature in feature_names:
        # Check if this is a lag feature
        if "_lag_" in feature:
            # Parse the lag period, e.g., "L1_act_power_lag_1h" -> "1h"
            lag_parts = feature.split("_lag_")
            if len(lag_parts) > 1:
                lag_period = lag_parts[1]
                try:
                    lag_td = pd.Timedelta(lag_period)

                    # Only keep features with lag >= forecast horizon
                    if lag_td >= forecast_td:
                        valid_features.append(feature)
                    else:
                        excluded_features.append(feature)
                except:
                    # If we can't parse the lag period, include the feature
                    valid_features.append(feature)
            else:
                valid_features.append(feature)
        else:
            # Non-lag features are always valid
            valid_features.append(feature)

    return valid_features, excluded_features
