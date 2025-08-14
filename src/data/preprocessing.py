"""Data loading and preprocessing utilities for time series datasets.

This module provides helper functions to:

* Read raw CSV batches (``first_batch`` / ``second_batch``) into pandas DataFrames.
* Optionally aggregate time series to a coarser temporal resolution while validating lag specifications.
* Preprocess individual device DataFrames
  (timezone normalization, temporal feature extraction, optional cyclical features).
* Concatenate per-device frames while preserving temporal ordering.
* Filter for complete (non-missing) device time series.
* Generate supervised learning samples via sliding windows.

Only documentation / typing updates were introduced; no functional logic changes are intended.
"""

import os
import pandas as pd
import numpy as np
import traceback
from pandas.tseries.frequencies import to_offset
from typing import List, Optional, Dict, Tuple


def read_second_batch(
    extract_path: str,
    location: str = "industrial",
    site_ids: Optional[List[str]] = None,
) -> List[pd.DataFrame]:
    """Read site CSV files from the ``second_batch`` directory.

    Parameters
    ----------
    extract_path : str
        Base directory containing the ``second_batch`` subfolder (split into
        ``industrial`` or ``commercial``).
    location : str, default="industrial"
        Either ``"industrial"`` or ``"commercial"`` - selects the subfolder.
    site_ids : list of str, optional
        Specific site identifiers to load. If ``None`` all sites in the folder
        are loaded.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per loaded site. Columns are minimally cleaned and a
        ``device_id`` / ``serial_number`` column is injected.

    Raises
    ------
    ValueError
        If the expected location directory does not exist.
    """
    if location == "industrial":
        second_batch_path = os.path.join(extract_path, "industrial")
    else:
        second_batch_path = os.path.join(extract_path, "commercial")
    if not os.path.isdir(second_batch_path):
        raise ValueError(f"Directory not found: {second_batch_path}")

    all_site_data = []

    for filename in os.listdir(second_batch_path):
        if not filename.endswith(".csv"):
            continue
        site_name = os.path.splitext(filename)[0]  # 'site1.csv' -> 'site1'

        if site_ids is not None and site_name not in site_ids:
            continue

        file_path = os.path.join(second_batch_path, filename)
        try:
            df = pd.read_csv(file_path, parse_dates=["time"])
            df["device_id"] = site_name
            df["serial_number"] = site_name
            df = df.sort_values(by=["time"])
            if location == "industrial":
                df.drop(columns={"Q+ total", "min", "max"}, errors="ignore", inplace=True)
            elif location == "commercial":
                df.drop(columns={"seriesId"}, inplace=True)
                df["consumption"] = df["consumption"].diff()
                # df = df.iloc[1:]
            all_site_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {filename} is empty. Skipping.")
        except pd.errors.ParserError:
            print(f"Warning: Could not parse file {filename}. Skipping.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_site_data


def read_first_batch(
    extract_path: str,
    meter_names: Optional[List[str]] = None,
    aggregation: Optional[str] = None,
) -> List[pd.DataFrame]:
    """Read ``first_batch`` meter CSV files and assemble per-meter DataFrames.

    Parameters
    ----------
    extract_path : str
        Directory containing the CSV exports for the first batch.
    meter_names : list of str, optional
        Meter base-name prefixes to include (e.g. ``["df_TRB145-1", ...]``).
        If ``None`` a default list of known meters is used.
    aggregation : str, optional
        (Reserved) Frequency string for optional temporal aggregation. Currently
        unused; present for future extension.

    Returns
    -------
    list of pd.DataFrame
        One concatenated DataFrame per meter name (even if composed from
        multiple monthly files).

    Notes
    -----
    If a meter has no matching files, an empty list passed to ``pd.concat``
    would raise an error. Current implementation assumes at least one file per
    meter; consider adding a guard if this assumption may not hold.
    """
    if meter_names is None:
        meter_names = [
            "df_TRB145-1",
            "df_TRB145-2",
            "df_TRB145-3",
            "df_TRB145-4",
            "df_TRB145-5",
        ]

    all_meter_data = []
    for meter_name in meter_names:
        individual_meter_data = []
        for filename in os.listdir(extract_path):
            if filename.startswith(meter_name) and filename.endswith(".csv"):  # Assuming CSV files
                # print(filename)
                filepath = os.path.join(extract_path, filename)
                try:
                    df = pd.read_csv(filepath)  # Or pd.read_excel(...) for Excel files
                    df["device_id"] = meter_name
                    df["serial_number"] = meter_name
                    df.drop(columns={"result", "table", "_time", "Unnamed: 0"}, inplace=True)
                    df.rename(columns={"timestamp": "time"}, inplace=True)

                    # if aggregation is not None:
                    #     df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                    #     df['time'] = df['time'].dt.tz_convert(None)
                    #     df = df.set_index('time')
                    #     agg_cols = df.select_dtypes(include='number').columns
                    #     df = df.groupby([pd.Grouper(freq=aggregation), 'device_id', 'serial_number']).agg(
                    #         {col: 'mean' for col in agg_cols}
                    #     )
                    #     df = df.reset_index()

                    individual_meter_data.append(df)
                    # meter_data[meter_name] = df
                    # Move to the next meter name after finding the file
                except pd.errors.EmptyDataError:
                    print(f"Warning: File {filename} is empty. Skipping.")
                except pd.errors.ParserError:
                    print(f"Warning: Could not parse file {filename}. Skipping.")
                except Exception as e:
                    print(f"Error reading {filename}: {e}")
        all_meter_data.append(pd.concat(individual_meter_data, ignore_index=True))
    return all_meter_data


def aggregate_and_validate_lags(
    data_concat: pd.DataFrame,
    lag_freqs: Dict[str, int],
    time_agg: Optional[str] = None,
    fill_missing: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """Aggregate data and filter incompatible lag specifications.

    Performs optional temporal aggregation (by mean over numerical columns) at
    the frequency ``time_agg`` and then removes lag entries whose resolution is
    finer than the aggregation level.

    Parameters
    ----------
    data_concat : pd.DataFrame
        Original time series containing at least ``time``, ``device_id`` and
        ``serial_number`` columns.
    lag_freqs : dict
        Mapping of (label -> frequency value). Implementation expects values
        usable by ``pandas.to_offset``; entries whose offset is finer than the
        aggregation level are dropped. (Naming suggests frequency strings but
        current code calls ``to_offset`` on the dict value.)
    time_agg : str, optional
        Pandas offset alias (e.g. ``"1H"``, ``"30min"``). If ``None`` no
        aggregation is applied and lag filtering is skipped.
    fill_missing : bool, default=False
        If ``True`` attempts to fill in missing timestamps per device using a
        weekday + time-slot mean strategy (DST aware for tz-aware data).

    Returns
    -------
    (pd.DataFrame, dict)
        Tuple of the (possibly aggregated and gap-filled) DataFrame and a
        filtered dictionary of lag specifications.

    Raises
    ------
    Exception
        Propagates unexpected aggregation or validation failures after logging.
    """
    if not time_agg:
        return data_concat, lag_freqs

    try:
        print(f"[INFO] Aggregating data by: {time_agg}")
        data_concat["time"] = pd.to_datetime(data_concat["time"])
        data_concat = data_concat.set_index("time")
        agg_cols = data_concat.select_dtypes(include="number").columns
        data_concat = data_concat.groupby([pd.Grouper(freq=time_agg), "device_id", "serial_number"]).agg(
            {col: "mean" for col in agg_cols}
        )
        data_concat = data_concat.reset_index()

        if fill_missing:
            missing_entries = []

            # Ensure 'time' is datetime with tz info preserved
            data_concat["time"] = pd.to_datetime(data_concat["time"])

            for (device_id, serial_number), group in data_concat.groupby(["device_id", "serial_number"]):
                group = group.sort_values("time")

                # Create full range in UTC to avoid DST issues
                start_time = group["time"].min()
                end_time = group["time"].max()

                # If the data has timezone info, work in UTC
                if start_time.tz is not None:
                    # Convert to UTC for range generation
                    start_utc = start_time.tz_convert("UTC")
                    end_utc = end_time.tz_convert("UTC")

                    # Create range in UTC
                    full_range_utc = pd.date_range(start=start_utc, end=end_utc, freq=time_agg)

                    # Convert back to original timezone, handling DST transitions
                    original_tz = start_time.tz
                    full_range = full_range_utc.tz_convert(original_tz)

                    # Handle DST transitions by removing ambiguous/non-existent times
                    existing_times = set(group["time"])

                    # Filter out times that would cause DST issues
                    valid_times = []
                    for t in full_range:
                        try:
                            # Test if this timestamp is valid by trying to localize it
                            if t not in existing_times:
                                # Additional check: ensure the timestamp is not ambiguous
                                # by converting to UTC and back
                                test_utc = t.tz_convert("UTC")
                                test_back = test_utc.tz_convert(original_tz)
                                if test_back == t:
                                    valid_times.append(t)
                        except Exception:
                            # Skip problematic timestamps
                            continue

                    missing_times = valid_times
                else:
                    # No timezone info, proceed as before
                    full_range = pd.date_range(start=start_time, end=end_time, freq=time_agg)
                    existing_times = set(group["time"])
                    missing_times = [t for t in full_range if t not in existing_times]

                for t in missing_times:
                    missing_entries.append(
                        {
                            "time": t,
                            "device_id": device_id,
                            "serial_number": serial_number,
                            **{col: pd.NA for col in agg_cols},
                        }
                    )

            if missing_entries:
                missing_df = pd.DataFrame(missing_entries)
                data_concat = pd.concat([data_concat, missing_df], ignore_index=True)

            # Sort
            data_concat = data_concat.sort_values(["device_id", "serial_number", "time"]).reset_index(drop=True)

            # Filling strategy: average by weekday and exact time slot (as per aggregation level)
            if not data_concat.empty:
                agg_offset = to_offset(time_agg)

                data_concat["weekday"] = data_concat["time"].dt.dayofweek

                # For time slot, use UTC time to avoid DST issues
                if data_concat["time"].dt.tz is not None:
                    utc_times = data_concat["time"].dt.tz_convert("UTC")
                    data_concat["rounded_slot"] = utc_times.dt.floor(time_agg).dt.time
                else:
                    data_concat["rounded_slot"] = (data_concat["time"].dt.floor(time_agg)).dt.time

                group_cols = ["device_id", "weekday", "rounded_slot"]

                fill_values = data_concat.groupby(group_cols)[agg_cols].transform("mean")

                for col in agg_cols:
                    data_concat[col] = data_concat[col].fillna(fill_values[col])

                # Drop helper columns
                data_concat = data_concat.drop(columns=["rounded_slot"])

    except Exception as e:
        print(f"[ERROR] Aggregation failed: {e}")
        raise

    # Validate and filter lag frequencies
    try:
        agg_offset = to_offset(time_agg)
        valid_lag_freqs = {}
        for freq, val in lag_freqs.items():
            try:
                lag_offset = to_offset(val)
                if lag_offset.nanos < agg_offset.nanos:
                    print(f"[WARNING] Skipping lag '{freq}' — smaller than aggregation level '{time_agg}'")
                else:
                    valid_lag_freqs[freq] = val
            except Exception as e:
                print(f"[WARNING] Invalid lag frequency '{freq}': {e}")
        return data_concat, valid_lag_freqs

    except Exception as e:
        print(f"[ERROR] Lag validation failed: {e}")
        raise


def preprocess_df(
    dataframes: List[pd.DataFrame],
    device_list: Optional[List[str]] = None,
    add_cyclical_features: bool = False,
) -> List[pd.DataFrame]:
    """Normalize timezone and enrich each DataFrame with temporal features.

    Processing steps per DataFrame:
    * Optional device filtering.
    * Convert ``time`` to timezone-aware UTC then localize to Europe/Prague
      (falls back to naive on failure).
    * Extract hour / weekday / month (in UTC to avoid DST discontinuities).
    * Optionally append cyclical (sine/cosine) encodings via
      ``add_cyclical_time_features``.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        Input per-device or per-site DataFrames containing at least a ``time``
        column and ``device_id``.
    device_list : list of str, optional
        Subset of device IDs to retain. If ``None`` all devices are kept.
    add_cyclical_features : bool, default=False
        When ``True`` injects additional cyclical time-of-day / day-of-week
        features.

    Returns
    -------
    list of pd.DataFrame
        Preprocessed DataFrames in the same order as input (filtered if
        applicable). Returns an empty list on unrecoverable error.
    """
    try:
        processed = []
        for df in dataframes:
            # Filter by device_id if device_list is provided
            if device_list is not None:
                df = df[df["device_id"].isin(device_list)]

            # Convert 'time' column to datetime and handle errors
            df["time"] = pd.to_datetime(df["time"], errors="coerce", utc=True)

            # Convert to Prague timezone, handling DST transitions
            try:
                df["time"] = df["time"].dt.tz_convert("Europe/Prague")
            except Exception as e:
                print(f"[WARNING] Timezone conversion failed: {e}")
                # Fall back to removing timezone info
                df["time"] = df["time"].dt.tz_convert(None)

            df.reset_index(drop=True, inplace=True)

            if "id" in df.columns:
                df = df.drop(columns=["id"])

            # Use UTC time for time features to avoid DST issues
            if df["time"].dt.tz is not None:
                utc_times = df["time"].dt.tz_convert("UTC")
                df["hour"] = utc_times.dt.hour
                df["weekday"] = utc_times.dt.weekday
                df["month"] = utc_times.dt.month
            else:
                df["hour"] = df["time"].dt.hour
                df["weekday"] = df["time"].dt.weekday
                df["month"] = df["time"].dt.month

            # Optionally add cyclical features if requested
            if add_cyclical_features:
                # Import directly from feature_engineering
                from src.data.feature_engineering import add_cyclical_time_features

                df = add_cyclical_time_features(df)

            processed.append(df)

        return processed

    except Exception as e:
        print(f"[ERROR] Failed to preprocess dataframes: {e}")
        traceback.print_exc()
        return []


def concat_dataframes_by_time(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate multiple preprocessed DataFrames and order the rows by 'time'.

    Assumes that 'time' column is already in ``datetime`` format and sorted from ``preprocess_df``.

    Parameters
    ----------
    dataframes : list of pd.DataFrame
        DataFrames already normalized & time-sorted (output of
        :func:`preprocess_df`).

    Returns
    -------
    pd.DataFrame
        Single combined DataFrame, index reset.
    """
    try:
        # Simply concatenate, data is already sorted by time in preprocess_df
        combined_df = pd.concat(dataframes, ignore_index=True)
        combined_df.reset_index(drop=True, inplace=True)
        return combined_df

    except Exception as e:
        print(f"[ERROR] Failed to concatenate dataframes: {e}")
        traceback.print_exc()
        return pd.DataFrame()


def get_complete_time_series(df: pd.DataFrame) -> List[pd.DataFrame]:
    """Extract only fully observed device time series (without missing values).

    Parameters
    ----------
    df : pd.DataFrame
        Concatenated DataFrame containing multiple devices / serial numbers.

    Returns
    -------
    list of pd.DataFrame
        One DataFrame per ``(device_id, serial_number)`` pair without any
        missing values. Groups with any NA are skipped.
    """
    try:
        complete_series = []
        grouped = df.groupby(["device_id", "serial_number"])
        for (device, serial), group in grouped:
            # Ensure the group is sorted by time.
            group_sorted = group.sort_values("time")

            # Skip groups with any missing data.
            if group_sorted.isnull().any().any():
                print(f"Skipping time series for device_id: {device}, serial_number: {serial} due to missing data.")
                continue
            complete_series.append(group_sorted)

        return complete_series

    except Exception as e:
        print(f"[ERROR] Failed to get complete time series: {e}")
        traceback.print_exc()
        return []


def create_supervised_data_from_series(
    df: pd.DataFrame,
    input_features: List[str],
    target_feature: str,
    window_size: int = 1,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a single time series DataFrame into windowed supervised samples.

    Constructs sliding windows of size ``window_size`` (moving forward with
    stride ``step``) over the ordered rows of ``df``.

    Parameters
    ----------
    df : pd.DataFrame
        Source time series (must be ordered by time externally).
    input_features : list of str
        Column names used as predictor variables inside each window.
    target_feature : str
        Column whose value at the prediction index (the row immediately after
        each window) becomes the target.
    window_size : int, default=1
        Number of consecutive rows included as model input for each sample.
    step : int, default=1
        Offset increment between successive windows.

    Returns
    -------
    (np.ndarray, np.ndarray)
        Tuple ``(X, y)`` where ``X`` has shape ``(n_samples, window_size * n_features)``
        and ``y`` has shape ``(n_samples,)``.
    """
    try:
        X, y = [], []
        for i in range(window_size, len(df), step):
            window_data = df.iloc[i - window_size : i][input_features].values.flatten()
            X.append(window_data)
            y.append(df.iloc[i][target_feature])
        return np.array(X), np.array(y)

    except Exception as e:
        print(f"[ERROR] Failed to create supervised data from series: {e}")
        traceback.print_exc()
        return np.array([]), np.array([])


def clean_data(
    ts_set: List[pd.DataFrame],
    input_features: List[str],
    target_feature: str,
    window_size: int,
    step: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Generate supervised samples for multiple time series and drop NA windows.

    Parameters
    ----------
    ts_set : list of pd.DataFrame
        Collection of complete (ideally) time series DataFrames.
    input_features : list of str
        Predictor columns to extract per window.
    target_feature : str
        Target column for each sample.
    window_size : int
        Sliding window length.
    step : int
        Stride between starting indices of successive windows.

    Returns
    -------
    (list of np.ndarray, list of np.ndarray)
        Lists ``X_list`` and ``y_list`` each containing arrays per input
        time series after removing any samples with missing values.
    """
    try:
        X_list, y_list = [], []
        for ts in ts_set:
            X_ts, y_ts = create_supervised_data_from_series(ts, input_features, target_feature, window_size, step)
            valid = ~pd.DataFrame(X_ts).isna().any(axis=1)
            X_list.append(X_ts[valid])
            y_list.append(y_ts[valid])
        return X_list, y_list

    except Exception as e:
        print(f"[ERROR] Failed to clean data: {e}")
        traceback.print_exc()
        return [], []
