import os
import glob
import pandas as pd
import numpy as np
import argparse
import warnings
import traceback
from typing import List, Tuple, Dict, Optional
from sklearn.model_selection import TimeSeriesSplit
from dateutil.relativedelta import relativedelta

warnings.filterwarnings("ignore")

# method to count months in the dataset timeline
def count_full_months(series_start, series_end):
    current = pd.Timestamp(series_start).replace(day=1)
    end = pd.Timestamp(series_end)
    count = 0

    while current < end:
        next_month = current + relativedelta(months=1)
        if (next_month - current).days <= 28:
            # Skip months with fewer days (shouldn't really happen, but just in case)
            current = next_month
            continue
        if (min(end, next_month) - current).days >= 30:
            count += 1
        current = next_month

    return count

# ---------------------------
# Preprocessing Functions
# ---------------------------
def preprocess_df(dataframes: List[pd.DataFrame], 
                 add_cyclical_features: bool = False) -> List[pd.DataFrame]:
    """
    Preprocess a list of DataFrames by converting 'time' column to datetime, sorting, and extracting time components.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames to preprocess.
        add_cyclical_features (bool): Whether to add sine/cosine cyclical features for time components.
                                     Default is False for backward compatibility.

    Returns:
        list of pd.DataFrame: List of preprocessed DataFrames.
    """
    try:
        processed = []
        for df in dataframes:
            df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            df['time'] = df['time'].dt.tz_convert(None)
            df.sort_values('time', inplace=True)
            df.reset_index(drop=True, inplace=True)

            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            # Always add basic time components for backward compatibility
            df['hour'] = df['time'].dt.hour
            df['minute'] = df['time'].dt.minute
            df['second'] = df['time'].dt.second
            df['day'] = df['time'].dt.day
            df['month'] = df['time'].dt.month
            
            # Optionally add cyclical features if requested
            if add_cyclical_features:
                df = add_cyclical_time_features(df)
                
            processed.append(df)

        return processed
    
    except Exception as e:
        print(f"[ERROR] Failed to preprocess dataframes: {e}")
        traceback.print_exc()
        return []

def concat_dataframes_by_time(dataframes: List[pd.DataFrame]) -> pd.DataFrame:
    """
    Concatenate a list of DataFrames into a single DataFrame and order the rows by 'time'.
    Assumes that 'time' column is already in datetime format and sorted from preprocess_df.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames to concatenate.

    Returns:
        pd.DataFrame: Concatenated DataFrame.
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
    """
    Group the concatenated DataFrame by device_id and serial_number,
    then return a list of groups (DataFrames) that have no missing values.

    Args:
        df (pd.DataFrame): DataFrame to group and filter.
    
    Returns:
        list of pd.DataFrame: List of DataFrames, each representing a complete time series for a device.
    """
    try:
        complete_series = []
        grouped = df.groupby(['device_id', 'serial_number'])
        for (device, serial), group in grouped:
            # Ensure the group is sorted by time.
            group_sorted = group.sort_values('time')

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

# ---------------------------
# Feature Engineering
# ---------------------------
def add_lag_features(df, columns, lag_freqs):

    try:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.sort_values('time', inplace=True)

        # for col in columns:
        #     df[f'{col}_lag_1'] = df[col].shift(1)

        for suffix, lag_str in lag_freqs.items():
            lag_delta = pd.Timedelta(lag_str)
            df[f'time_shift_{suffix}'] = df['time'] - lag_delta
            temp = df[['time'] + columns].copy()
            rename_dict = {col: f'{col}_{suffix}' for col in columns}
            temp.rename(columns=rename_dict, inplace=True)

            merged = pd.merge_asof(
                df.sort_values(f'time_shift_{suffix}'),
                temp.sort_values('time'),
                left_on=f'time_shift_{suffix}',
                right_on='time',
                direction='backward',
                tolerance=pd.Timedelta(lag_str)
            )

            for col in columns:
                df[f'{col}_{suffix}'] = merged[f'{col}_{suffix}'].values

            df.drop(columns=[f'time_shift_{suffix}'], inplace=True)
            #print(df.columns)

        return df

    except Exception as e:
        print(f"[ERROR] Failed to add lag features: {e}")
        traceback.print_exc()
        return df

def add_cyclical_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add cyclical time features using sine and cosine transformations.
    This creates better features for time-based data by encoding the cyclical nature
    of time components (hour of day, day of week, month of year, etc.)
    
    Args:
        df (pd.DataFrame): DataFrame with a 'time' column in datetime format
        
    Returns:
        pd.DataFrame: DataFrame with added cyclical time features
    """
    try:
        df = df.copy()
        
        # Hour of day (24 hours)
        df['hour_sin'] = np.sin(2 * np.pi * df['time'].dt.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df['time'].dt.hour / 24)
        
        # Day of week (7 days)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['time'].dt.dayofweek / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['time'].dt.dayofweek / 7)
        
        # Month of year (12 months)
        df['month_sin'] = np.sin(2 * np.pi * (df['time'].dt.month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['time'].dt.month - 1) / 12)

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

# ---------------------------
# Create Supervised Data
# ---------------------------
def create_supervised_data_from_series(df: pd.DataFrame, 
                                       input_features: List[str], 
                                       target_feature: str, 
                                       window_size: int = 1, 
                                       step: int = 1) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create supervised data from a time series DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame containing the time series data.
        input_features (list of str): List of feature names to be used as inputs.
        target_feature (str): Name of the target feature.
        window_size (int): Size of the sliding window. Defaults to 1.
        step (int): Step size for the sliding window. Defaults to 1.

    Returns:
        tuple: Tuple containing the input features (X) and target variable (y).
    """
    try:
        X, y = [], []
        for i in range(window_size, len(df), step):
            window_data = df.iloc[i-window_size:i][input_features].values.flatten()
            X.append(window_data)
            y.append(df.iloc[i][target_feature])
        return np.array(X), np.array(y)
    
    except Exception as e:
        print(f"[ERROR] Failed to create supervised data from series: {e}")
        traceback.print_exc()
        return np.array([]), np.array([])


def clean_data(ts_set: List[pd.DataFrame],
               input_features: List[str], 
               target_feature: str, 
               window_size: int, 
               step: int) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Clean the data by creating supervised data from a set of time series.

    Args:
        ts_set (list of pd.DataFrame): List of DataFrames representing time series data.
        input_features (list of str): List of feature names to be used as inputs.
        target_feature (str): Name of the target feature.
        window_size (int): Size of the sliding window.
        step (int): Step size for the sliding window.

    Returns:
        tuple: Tuple containing the input features (X) and target variable (y) for all time series.
    """
    try:
        X_list, y_list = [], []
        for ts in ts_set:
            X_ts, y_ts = create_supervised_data_from_series(ts, input_features, target_feature, window_size, step)
            valid = ~np.isnan(X_ts).any(axis=1)
            X_list.append(X_ts[valid])
            y_list.append(y_ts[valid])
        return X_list, y_list
    
    except Exception as e:
        print(f"[ERROR] Failed to clean data: {e}")
        traceback.print_exc()
        return [], []

def get_train_test_sets(data_dir: str,
                        lag_freqs: Dict,
                        columns_to_add_lag: List[str],
                        target_feature: str = 'total_power_active',
                        walk_forward_sets: bool = False,
                        train_months: int = 3,
                        valid_months: int = 1,
                        window_size: int = 1,
                        step: int = 1,
                        use_cyclical_features: bool = False) -> Tuple[
                                                Tuple[np.ndarray, np.ndarray], 
                                                Tuple[np.ndarray, np.ndarray], 
                                                Tuple[np.ndarray, np.ndarray]
                                                ]:
    
    """
    Create training, validation, and test sets from the data directory.

    The data is preprocessed, and lag features are added based on the specified frequencies.

    Args:
        data_dir (str): Directory containing the CSV files.
        lag_freqs (dict): Dictionary mapping suffixes to lag frequencies.
        columns_to_add_lag (list): List of columns to create lag features for.
        target_feature (str): Target feature for prediction. Defaults to 'total_power_active'.
        walk_forward_sets (bool): Whether to use walk-forward validation. Defaults to False.
        train_months (int): Number of months for the training window (if using walk-forward validation). Defaults to 3.
        valid_months (int): Number of months for the validation window (if using walk-forward validation). Defaults to 1.
        window_size (int): Size of the sliding window for creating supervised data. Defaults to 1.
        step (int): Step size for the sliding window. Defaults to 1.
        use_cyclical_features (bool): Whether to use sine/cosine transformations for time features. Defaults to False.

    Returns:
        tuple: A tuple containing training, validation, and test sets. Each set is a tuple of (X, y), 
               where X is the feature matrix and y is the target vector.
               Returns (None, None), (None, None), (None, None) if an error occurs.
    """
    if not os.path.exists(data_dir):
        print(f"[ERROR] The directory {data_dir} does not exist.")
        return (None, None), (None, None), (None, None)

    try:
        csv_files = glob.glob(os.path.join(data_dir, "smart_meter_data_*.csv"))
        if not csv_files:
            print(f"[ERROR] No matching CSV files found in {data_dir}")
            return (None, None), (None, None), (None, None)
            
        dataframes = [pd.read_csv(file) for file in csv_files]
    
    except Exception as e:
        print(f"[ERROR] Failed to read CSV files: {e}")
        traceback.print_exc()
        return (None, None), (None, None), (None, None)

    try:
        processed_dfs = preprocess_df(dataframes, add_cyclical_features=use_cyclical_features)
        if not processed_dfs:
            print("[ERROR] Preprocessing returned empty result")
            return (None, None), (None, None), (None, None)
            
        data_concat = concat_dataframes_by_time(processed_dfs)
        if data_concat.empty:
            print("[ERROR] Concatenation returned empty DataFrame")
            return (None, None), (None, None), (None, None)
            
        data_concat.drop(columns='values_state', inplace=True, errors='ignore')
        complete_time_series = get_complete_time_series(data_concat)
        if not complete_time_series:
            print("[ERROR] No complete time series found in the data")
            return (None, None), (None, None), (None, None)
        
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        traceback.print_exc()
        return (None, None), (None, None), (None, None)

    if columns_to_add_lag is None:
        columns_to_add_lag = [
            'total_power_active', 'total_power_reactive', 'total_power_apparent',
            'import_energy_active', 'export_energy_active',
            'import_energy_reactive', 'export_energy_reactive',
            'total_energy_active', 'total_energy_reactive'
        ]

    input_features = [
        col for col in data_concat.drop(columns=columns_to_add_lag).columns
        if col not in ['time', 'device_id', 'serial_number']
    ]

    # Append lagged column names
    for col in columns_to_add_lag:
        for freq in lag_freqs.keys():
            lagged_col_name = f"{col}_{freq}"
            input_features.append(lagged_col_name)
    print("input_features: ", input_features)

    window_size = 1
    step = 1
    train_ratio = 0.6
    valid_ratio = 0.2

    train_sets, valid_sets, test_sets = [], [], []

    for idx, ts in enumerate(complete_time_series[:]):
        try:
            ts['time'] = pd.to_datetime(ts['time'])

            train_end_idx = int(len(ts) * train_ratio)
            valid_end_idx = int(len(ts) * (train_ratio+valid_ratio))

            ts_lagged = add_lag_features(ts, columns_to_add_lag, lag_freqs)

            ts_test = ts_lagged.iloc[valid_end_idx:]
            X_test, y_test = clean_data([ts_test], input_features, target_feature, window_size, step)
            test_sets.append((np.vstack(X_test), np.hstack(y_test)))
            
            if not walk_forward_sets:
                ts_train = ts_lagged.iloc[:train_end_idx]
                series_start = ts_train['time'].min()
                series_end = ts_train['time'].max()
                #print(series_start, series_end)

                X_train, y_train = clean_data([ts_train], input_features, target_feature, window_size, step)
                train_sets.append((np.vstack(X_train), np.hstack(y_train)))

                ts_valid = ts_lagged[train_end_idx:valid_end_idx]
                X_valid, y_valid = clean_data([ts_valid], input_features, target_feature, window_size, step)
                valid_sets.append((np.vstack(X_valid), np.hstack(y_valid)))
            else:
                ts_train = ts_lagged.iloc[:(train_end_idx+valid_end_idx)]
                series_start = ts_train['time'].min()
                series_end = ts_train['time'].max()

                # Count how many full months are in the training+validation period
                num_full_months = count_full_months(series_start, series_end)
                #print(f"Number of full months: {num_full_months}")

                X_train, y_train = clean_data([ts_train], input_features, target_feature, window_size, step)
                train_sets.append((np.vstack(X_train), np.hstack(y_train)))

        except Exception as e:
            print(f"[WARNING] Skipped one time series (index {idx}) due to error:")
            traceback.print_exc()
            continue

    # Stack all sets
    try:
        if walk_forward_sets:
            all_train_sets, all_valid_sets = [], []

            X_train_all = np.vstack([x for x, _ in train_sets])
            y_train_all = np.hstack([y for _, y in train_sets])

            X_test_all = np.vstack([x for x, _ in test_sets])
            y_test_all = np.hstack([y for _, y in test_sets])

            n_splits = num_full_months-(train_months+valid_months)+1
            print('number of walk forward splits', n_splits)

            tscv = TimeSeriesSplit(n_splits=n_splits)

            for i, (train_index, valid_index) in enumerate(tscv.split(X_train_all)):
                all_train_sets.append((X_train_all[train_index], y_train_all[train_index]))
                all_valid_sets.append((X_train_all[valid_index], y_train_all[valid_index]))

            print("data range: ", series_start, series_end)
            total_timestamps = pd.date_range(start=series_start, end=series_end, periods=X_train_all.shape[0])

            for i, (train_index, valid_index) in enumerate(tscv.split(X_train_all)):
                # Get start and end datetime for train and valid sets
                train_start_time = total_timestamps[train_index[0]]
                train_end_time = total_timestamps[train_index[-1]]

                valid_start_time = total_timestamps[valid_index[0]]
                valid_end_time = total_timestamps[valid_index[-1]]

                print(f"Split {i+1}:")
                print(f"  Train window: {train_start_time.date()} to {train_end_time.date()}")
                print(f"  Valid window: {valid_start_time.date()} to {valid_end_time.date()}")

            return all_train_sets, all_valid_sets, [X_test_all, y_test_all]
        else:
            X_train_all = np.vstack([x for x, _ in train_sets])
            y_train_all = np.hstack([y for _, y in train_sets])

            X_valid_all = np.vstack([x for x, _ in valid_sets])
            y_valid_all = np.hstack([y for _, y in valid_sets])

            X_test_all = np.vstack([x for x, _ in test_sets])
            y_test_all = np.hstack([y for _, y in test_sets])


            print(f"Total Training samples: {X_train_all.shape[0]}")
            print(f"Total Validation samples: {X_valid_all.shape[0]}")
            print(f"Total Test samples: {X_test_all.shape[0]}")

            return (X_train_all, y_train_all), (X_valid_all, y_valid_all), (X_test_all, y_test_all)
    except Exception as e:
        print(f"[ERROR] Failed to stack all sets: {e}")
        return None, None, None, None, None, None
