import pandas as pd
import numpy as np
import traceback
from typing import List, Optional

def preprocess_df(dataframes: List[pd.DataFrame],
                 device_list: Optional[List[str]] = None,
                 add_cyclical_features: bool = False) -> List[pd.DataFrame]:
    """
    Preprocess a list of DataFrames by converting 'time' column to datetime and extracting time components.

    Args:
        dataframes (list of pd.DataFrame): List of DataFrames to preprocess.
        device_list (list of str): List of device IDs to filter the DataFrames. If None, no filtering is applied.
        add_cyclical_features (bool): Whether to add sine/cosine cyclical features for time components.
                                     Default is False for backward compatibility.

    Returns:
        list of pd.DataFrame: List of preprocessed DataFrames.
    """
    try:
        processed = []
        for df in dataframes:
            # Filter by device_id if device_list is provided
            if device_list is not None:
                df = df[df['device_id'].isin(device_list)]

            # Convert 'time' column to datetime and handle errors
            df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
            df['time'] = df['time'].dt.tz_convert(None)
            df.reset_index(drop=True, inplace=True)

            if 'id' in df.columns:
                df = df.drop(columns=['id'])

            df['hour'] = df['time'].dt.hour
            df['weekday'] = df['time'].dt.weekday
            df['month'] = df['time'].dt.month
            #possibly add combination hour + minute
            #df['minute'] = df['time'].dt.hour + df['time'].dt.minute / 60
            
            # Optionally add cyclical features if requested
            if add_cyclical_features:
                # Import directly from feature_engineering
                from .feature_engineering import add_cyclical_time_features
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

def create_supervised_data_from_series(df: pd.DataFrame, 
                                      input_features: List[str], 
                                      target_feature: str, 
                                      window_size: int = 1, 
                                      step: int = 1) -> tuple:
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
              step: int) -> tuple:
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
            valid = ~pd.DataFrame(X_ts).isna().any(axis=1)
            X_list.append(X_ts[valid])
            y_list.append(y_ts[valid])
        return X_list, y_list
    
    except Exception as e:
        print(f"[ERROR] Failed to clean data: {e}")
        traceback.print_exc()
        return [], []
