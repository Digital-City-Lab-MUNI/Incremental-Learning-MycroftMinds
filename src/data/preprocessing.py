import os
import pandas as pd
import numpy as np
import traceback
from pandas.tseries.frequencies import to_offset
from typing import List, Optional, Optional, Dict, Tuple

def read_second_batch(extract_path, location='industrial', site_ids=None):
    """
    Reads data from the 'second_batch' folder for specified sites.

    Args:
        extract_path (str): Base path containing the 'second_batch' folder.
        site_ids (list or None): List of site identifiers like ['site1', 'site2'].
                                 If None, reads all site files.
        location (str): industrial or commercial.

    Returns:
        List of DataFrames, one per site.
    """
    if location=='industrial':
        second_batch_path = os.path.join(extract_path, 'second_batch/industrial')
    else:
        second_batch_path = os.path.join(extract_path, 'second_batch/commercial')
    if not os.path.isdir(second_batch_path):
        raise ValueError(f"Directory not found: {second_batch_path}")
    
    all_site_data = []
    
    for filename in os.listdir(second_batch_path):
        if not filename.endswith('.csv'):
            continue
        site_name = os.path.splitext(filename)[0]  # 'site1.csv' -> 'site1'

        if site_ids is not None and site_name not in site_ids:
            continue
        
        file_path = os.path.join(second_batch_path, filename)
        try:
            df = pd.read_csv(file_path, parse_dates=['time'])
            df['device_id'] = site_name
            df['serial_number'] = site_name
            df = df.sort_values(by=['time'])
            if location=='industrial':
                df.drop(columns={'Q+ total', 'min', 'max'}, errors='ignore', inplace=True)
            elif location=='commercial':
                df.drop(columns={'seriesId'}, inplace=True)
                df['consumption'] = df['consumption'].diff()
                # df = df.iloc[1:]
            all_site_data.append(df)
        except pd.errors.EmptyDataError:
            print(f"Warning: File {filename} is empty. Skipping.")
        except pd.errors.ParserError:
            print(f"Warning: Could not parse file {filename}. Skipping.")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    return all_site_data

def read_new_meter_data(extract_path, meter_names=None, aggregation=None):
    """
    Reads data from specified meter files into pandas DataFrames.

    Args:
        data_path: The path to the directory containing the meter data files.
        meter_names: A list of meter names to read data for.

    Returns:
        A list with corresponding DataFrames.
        Returns an empty list if no valid meter files are found or an error occurs.
    """
    if meter_names is None:
        meter_names = ['df_TRB145-1', 'df_TRB145-2', 'df_TRB145-3', 'df_TRB145-4', 'df_TRB145-5']


    all_meter_data = []
    for meter_name in meter_names:
        individual_meter_data  = []
        for filename in os.listdir(extract_path):
            if filename.startswith(meter_name) and filename.endswith(".csv"):  # Assuming CSV files
                #print(filename)
                filepath = os.path.join(extract_path, filename)
                try:
                    df = pd.read_csv(filepath)  # Or pd.read_excel(...) for Excel files
                    df['device_id'] = meter_name
                    df['serial_number'] = meter_name
                    df.drop(columns={'result', 'table', '_time', 'Unnamed: 0'}, inplace=True)
                    df.rename(columns={'timestamp':'time'}, inplace=True)

                    # if aggregation is not None:
                    #     df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
                    #     df['time'] = df['time'].dt.tz_convert(None)
                    #     df = df.set_index('time')
                    #     agg_cols = df.select_dtypes(include='number').columns
                    #     df = df.groupby([pd.Grouper(freq=aggregation), 'device_id', 'serial_number']).agg({col: 'mean' for col in agg_cols})
                    #     df = df.reset_index()

                    individual_meter_data.append(df)
                    #meter_data[meter_name] = df
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
    fill_missing = False
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Aggregates the time series data and filters lag frequencies that are incompatible
    with the aggregation level.

    Parameters:
        data_concat (pd.DataFrame): Original time series dataframe with 'time', 'device_id' and 'serial_number' columns.
        lag_freqs (dict): Dictionary mapping lag frequency strings (e.g., '1H', '1D') to the number of lags.
        time_agg (str, optional): Aggregation frequency string (e.g., '1H', '30min'). If None, no aggregation.
        fill_missing (bool): fill missing values

    Returns:
        Tuple of:
        - Aggregated (or original) data_concat
        - Filtered lag_freqs dict where each lag >= aggregation level
    """
    if not time_agg:
        return data_concat, lag_freqs

    try:
        print(f"[INFO] Aggregating data by: {time_agg}")
        data_concat['time'] = pd.to_datetime(data_concat['time'])
        data_concat = data_concat.set_index('time')
        agg_cols = data_concat.select_dtypes(include='number').columns
        data_concat = data_concat.groupby([pd.Grouper(freq=time_agg), 'device_id', 'serial_number']).agg({col: 'mean' for col in agg_cols})
        data_concat = data_concat.reset_index()
        
        if fill_missing:
            missing_entries = []

            # Ensure 'time' is datetime with tz info preserved
            data_concat['time'] = pd.to_datetime(data_concat['time'])

            for (device_id, serial_number), group in data_concat.groupby(['device_id', 'serial_number']):
                group = group.sort_values('time')
                full_range = pd.date_range(start=group['time'].min(), end=group['time'].max(), freq=time_agg)
                
                existing_times = set(group['time'])
                missing_times = [t for t in full_range if t not in existing_times]

                for t in missing_times:
                    missing_entries.append({
                        'time': t,
                        'device_id': device_id,
                        'serial_number': serial_number,
                        **{col: pd.NA for col in agg_cols}
                    })

            if missing_entries:
                missing_df = pd.DataFrame(missing_entries)
                data_concat = pd.concat([data_concat, missing_df], ignore_index=True)

            # Sort
            data_concat = data_concat.sort_values(['device_id', 'serial_number', 'time']).reset_index(drop=True)

            # Filling strategy: average by weekday and exact time slot (as per aggregation level)
            if not data_concat.empty:
                from pandas.tseries.frequencies import to_offset
                agg_offset = to_offset(time_agg)

                data_concat['weekday'] = data_concat['time'].dt.dayofweek
                data_concat['rounded_slot'] = (data_concat['time'].dt.floor(time_agg)).dt.time

                group_cols = ['device_id', 'weekday', 'rounded_slot']

                fill_values = (
                    data_concat.groupby(group_cols)[agg_cols]
                    .transform('mean')
                )

                for col in agg_cols:
                    data_concat[col] = data_concat[col].fillna(fill_values[col])

                # Drop helper columns
                data_concat = data_concat.drop(columns=['rounded_slot'])

            
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
                from src.data.feature_engineering import add_cyclical_time_features
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
