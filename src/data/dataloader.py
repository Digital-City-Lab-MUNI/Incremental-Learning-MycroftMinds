import os
import glob
import pandas as pd
import numpy as np
import warnings
import traceback
from typing import List, Tuple, Dict, Optional, Union, Any
from sklearn.model_selection import TimeSeriesSplit

# Import modules from the new structure
from src.data.io import save_dataset, load_dataset
from src.utils.memory import calculate_memory_usage, optimize_memory_usage
from src.data.preprocessing import preprocess_df, concat_dataframes_by_time, get_complete_time_series, clean_data, aggregate_and_validate_lags, read_new_meter_data, read_second_batch
from src.data.feature_engineering import add_lag_features, add_cyclical_time_features

warnings.filterwarnings("ignore")

# ---------------------------
# Main Dataset Creation Function
# ---------------------------
def get_train_test_sets(data_dir: str,
                        lag_freqs: Dict,
                        columns_to_add_lag: List[str],
                        target_feature: str = 'total_power_active',
                        device_list: Optional[List[str]] = None,
                        walk_forward_sets: bool = True,
                        train_days: int = 28,
                        valid_days: int = 28,
                        window_size: int = 1,
                        step: int = 1,
                        add_cyclical_features: bool = False,
                        optimize_memory: bool = True,
                        save: bool = True,
                        time_agg: Optional[str] = None,
                        intermediate_prefix: str = '5_min_wfv', 
                        new_data='second_batch',
                        location=None,
                        fill_missing=True) -> Tuple[
                                                Any, 
                                                Any, 
                                                Tuple[np.ndarray, np.ndarray]
                                                ]:
    
    """
    Create training, validation, and test sets from the data directory with optimized memory usage.

    The data is preprocessed, lag features are added, and the result is returned in memory-efficient format.

    Args:
        data_dir (str): Directory containing the CSV files.
        lag_freqs (dict): Dictionary mapping suffixes to lag frequencies.
        columns_to_add_lag (list of str): List of columns to create lag features for.
        target_feature (str): Target feature for prediction. Defaults to 'total_power_active'.
        device_list (list of str): List of devices to include. If None, all devices are included. Defaults to None.
        walk_forward_sets (bool): Whether to use walk-forward validation. Defaults to True.
        train_days (int): Number of days for the training window (if using walk-forward validation). Defaults to 28 (4 weeks).
        valid_days (int): Number of days for the validation window (if using walk-forward validation). Defaults to 28 (4 weeks).
        window_size (int): Size of the sliding window for creating supervised data. Defaults to 1.
        step (int): Step size for the sliding window. Defaults to 1.
        add_cyclical_features (bool): Whether to use sine/cosine transformations for time features. Defaults to False.
        optimize_memory (bool): Whether to optimize memory usage by using efficient dtypes. Defaults to True.
        save (bool): Whether to save the results to disk. Defaults to True.
        intermediate_prefix (str): Prefix for intermediate files. Defaults to 'processed'.
        new_data (str): 'first_batch' or 'second_batch' (defaults 'second_batch')
        location (str): to be used with second batch. specify 'industrial' or 'commercial' (only valid for second batch data)
        fill_missing (bool): whether to fill the missing values (defaults True), method: weekly, hourly mean
        
    Returns:
        tuple: A tuple containing training, validation, and test sets. Each set is a tuple of (X, y), 
               where X is the feature matrix and y is the target vector.
    """
    # Check if processed data already exists
    if not os.path.exists(data_dir):
        print(f"[ERROR] The directory {data_dir} does not exist.")
        return (None, None), (None, None), (None, None)

    try:
        if new_data=='first_batch':
            dataframes = read_new_meter_data(data_dir, device_list, aggregation=time_agg)
        elif new_data=='second_batch':
            dataframes = read_second_batch(data_dir, location=location, site_ids=device_list)
        else:
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
        processed_dfs = preprocess_df(dataframes, device_list, add_cyclical_features)
        if not processed_dfs:
            print("[ERROR] Preprocessing returned empty result")
            return (None, None), (None, None), (None, None)
            
        data_concat = concat_dataframes_by_time(processed_dfs)
        if data_concat.empty:
            print("[ERROR] Concatenation returned empty DataFrame")
            return (None, None), (None, None), (None, None)

        # Aggregation (if any)
        data_concat, lag_freqs = aggregate_and_validate_lags(data_concat, lag_freqs, time_agg, fill_missing=fill_missing)
        
        data_concat.drop(columns='values_state', inplace=True, errors='ignore')
        complete_time_series = get_complete_time_series(data_concat)

        if not complete_time_series:
            print("f[ERROR] No complete time series found in the data")
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
        if col not in ['serial_number'] # 'time', 'device_id'
    ]

    # Append lagged column names to input features
    for col in columns_to_add_lag:
        for freq in lag_freqs.keys():
            lagged_col_name = f"{col}_{freq}"
            input_features.append(lagged_col_name)
    print("input_features: ", input_features)
    print()

    train_ratio = 0.6
    valid_ratio = 0.2

    # NEW: Process each site separately to maintain site-specific timelines
    site_processed_data = {}
    
    for idx, ts in enumerate(complete_time_series[:]):
        try:
            ts['time'] = pd.to_datetime(ts['time'])
            
            # Get site identifier
            site_id = ts['device_id'].iloc[0]
            print(f"Processing site: {site_id}")
            
            train_end_idx = int(len(ts) * train_ratio)
            valid_end_idx = int(len(ts) * (train_ratio+valid_ratio))

            ts_lagged = add_lag_features(ts, columns_to_add_lag, lag_freqs)
            
            # Store site-specific data
            site_processed_data[site_id] = {
                'data': ts_lagged,
                'train_end_idx': train_end_idx,
                'valid_end_idx': valid_end_idx
            }
            
        except Exception as e:
            print(f"[WARNING] Skipped one time series (index {idx}) due to error:")
            traceback.print_exc()
            continue

    # Now process train/valid/test splits while keeping sites separate
    train_sets, valid_sets, test_sets = [], [], []
    
    for site_id, site_data in site_processed_data.items():
        ts_lagged = site_data['data']
        train_end_idx = site_data['train_end_idx']
        valid_end_idx = site_data['valid_end_idx']
        
        # Test set
        ts_test = ts_lagged.iloc[valid_end_idx:]
        if len(ts_test) > 0:
            X_test, y_test = clean_data([ts_test], input_features, target_feature, window_size, step)
            if len(X_test) > 0:
                test_sets.append((np.vstack(X_test), np.hstack(y_test)))
        
        if not walk_forward_sets:
            # Non-walk-forward case
            ts_train = ts_lagged.iloc[:train_end_idx]
            if len(ts_train) > 0:
                X_train, y_train = clean_data([ts_train], input_features, target_feature, window_size, step)
                if len(X_train) > 0:
                    train_sets.append((np.vstack(X_train), np.hstack(y_train)))

            ts_valid = ts_lagged.iloc[train_end_idx:valid_end_idx]
            if len(ts_valid) > 0:
                X_valid, y_valid = clean_data([ts_valid], input_features, target_feature, window_size, step)
                if len(X_valid) > 0:
                    valid_sets.append((np.vstack(X_valid), np.hstack(y_valid)))
        else:
            # Walk-forward case - process site data for training
            ts_train = ts_lagged.iloc[:valid_end_idx]
            if len(ts_train) > 0:
                X_train, y_train = clean_data([ts_train], input_features, target_feature, window_size, step)
                if len(X_train) > 0:
                    train_sets.append((np.vstack(X_train), np.hstack(y_train)))

    # Stack all sets
    try:
        if walk_forward_sets:
            all_train_sets, all_valid_sets = [], []

            # Combine all sites' training data
            X_train_all = np.vstack([x for x, _ in train_sets])
            y_train_all = np.hstack([y for _, y in train_sets])

            # Sort by timestamp to maintain chronological order
            sorted_indices = np.argsort(X_train_all[:, 0])
            X_train_all = X_train_all[sorted_indices]
            y_train_all = y_train_all[sorted_indices]

            # Combine all sites' test data
            X_test_all = np.vstack([x for x, _ in test_sets])
            y_test_all = np.hstack([y for _, y in test_sets])
            sorted_indices = np.argsort(X_test_all[:, 0])
            X_test_all = X_test_all[sorted_indices]
            y_test_all = y_test_all[sorted_indices]
            
            # Get the max timestamp in training set
            train_max_time = X_train_all[:, 0].max()

            # Filter test set to only include rows with time > max train time
            mask = X_test_all[:, 0] > train_max_time
            X_test_all = X_test_all[mask]
            y_test_all = y_test_all[mask]

            print("Overall data time range: ", X_train_all[:, 0].min(), X_test_all[:, 0].max())
            print()
            print("Test data time range: ", X_test_all[:, 0].min(), X_test_all[:, 0].max())
            print()
            print("(Train + Valid) data range: ", X_train_all[:, 0].min(), X_train_all[:, 0].max())
            print()

            # Convert timestamp column to Series for easier handling
            timestamps = pd.Series(X_train_all[:, 0])
            unique_hours = sorted(timestamps.dt.floor('H').unique())
            num_hours = len(unique_hours)

            train_hours = train_days * 24
            valid_hours = valid_days * 24
            train_valid = train_hours + valid_hours

            if train_valid > num_hours:
                print("[ERROR] data timeline too small for provided train+valid splits")
                return (None, None), (None, None), (None, None)

            all_train_sets = []
            all_valid_sets = []

            i = 0
            start_idx = 0

            # Walk-forward loop
            while start_idx + train_valid <= num_hours:
                train_range = unique_hours[start_idx : start_idx + train_hours]
                valid_range = unique_hours[start_idx + train_hours : start_idx + train_hours + valid_hours]

                train_mask = timestamps.dt.floor('H').isin(train_range)
                valid_mask = timestamps.dt.floor('H').isin(valid_range)

                X_train_split = X_train_all[train_mask.values]
                y_train_split = y_train_all[train_mask.values]

                X_valid_split = X_train_all[valid_mask.values]
                y_valid_split = y_train_all[valid_mask.values]

                if len(X_valid_split) == 0:
                    break

                all_train_sets.append((X_train_split, y_train_split))
                all_valid_sets.append((X_valid_split, y_valid_split))

                train_start_time = X_train_split[:, 0].min()
                train_end_time = X_train_split[:, 0].max()
                valid_start_time = X_valid_split[:, 0].min()
                valid_end_time = X_valid_split[:, 0].max()

                print(f"Split {i + 1}:")
                print(f"  Train window: {train_start_time} to {train_end_time}")
                print(f"  Valid window: {valid_start_time} to {valid_end_time}")
                print(f"  Train samples: {len(X_train_split)}, Valid samples: {len(X_valid_split)}")
                print()

                # Slide forward by train_hours for next split
                start_idx += train_hours
                i += 1

            # Handle remaining data
            if start_idx + train_hours < num_hours:
                train_range = unique_hours[start_idx : start_idx + train_hours]
                valid_range = unique_hours[start_idx + train_hours :]

                train_mask = timestamps.dt.floor('H').isin(train_range)
                valid_mask = timestamps.dt.floor('H').isin(valid_range)

                X_train_split = X_train_all[train_mask.values]
                y_train_split = y_train_all[train_mask.values]

                X_valid_split = X_train_all[valid_mask.values]
                y_valid_split = y_train_all[valid_mask.values]

                if len(X_valid_split) > 0:
                    all_train_sets.append((X_train_split, y_train_split))
                    all_valid_sets.append((X_valid_split, y_valid_split))

                    train_start_time = X_train_split[:, 0].min()
                    train_end_time = X_train_split[:, 0].max()
                    valid_start_time = X_valid_split[:, 0].min()
                    valid_end_time = X_valid_split[:, 0].max()

                    print(f"Split {i + 1}:")
                    print(f"  Train window: {train_start_time} to {train_end_time}")
                    print(f"  Valid window: {valid_start_time} to {valid_end_time}")
                    print(f"  Train samples: {len(X_train_split)}, Valid samples: {len(X_valid_split)}")
    
        else:
            # Non-walk-forward case
            X_train_all = np.vstack([x for x, _ in train_sets])
            y_train_all = np.hstack([y for _, y in train_sets])

            X_valid_all = np.vstack([x for x, _ in valid_sets])
            y_valid_all = np.hstack([y for _, y in valid_sets])

            X_test_all = np.vstack([x for x, _ in test_sets])
            y_test_all = np.hstack([y for _, y in test_sets])

            print(f"Total Training samples: {X_train_all.shape[0]}")
            print(f"Total Validation samples: {X_valid_all.shape[0]}")
            print(f"Total Test samples: {X_test_all.shape[0]}")

        # Apply memory optimization if requested
        if optimize_memory:
            # Process walk_forward_sets
            if walk_forward_sets and isinstance(all_train_sets, list):
                for i in range(len(all_train_sets)):
                    all_train_sets[i] = optimize_memory_usage(all_train_sets[i])

                for i in range(len(all_valid_sets)):
                    all_valid_sets[i] = optimize_memory_usage(all_valid_sets[i])

                final_test = optimize_memory_usage((X_test_all, y_test_all))

                # Save memory usage info
                train_mem = sum(calculate_memory_usage(ts) for ts in all_train_sets)
                valid_mem = sum(calculate_memory_usage(ts) for ts in all_valid_sets)
                test_mem = calculate_memory_usage(final_test)

                print(f"Memory usage after optimization: Train={train_mem:.2f}MB, Valid={valid_mem:.2f}MB, Test={test_mem:.2f}MB")

                # Save the processed data if requested
                if save:
                    save_dataset(all_train_sets, all_valid_sets, final_test, 
                                data_dir=data_dir, prefix=intermediate_prefix, use_numpy=True)

                return all_train_sets, all_valid_sets, final_test

            else:
                # Non-walk-forward case
                train_opt = optimize_memory_usage((X_train_all, y_train_all))
                valid_opt = optimize_memory_usage((X_valid_all, y_valid_all)) 
                test_opt = optimize_memory_usage((X_test_all, y_test_all))

                # Save memory usage info
                train_mem = calculate_memory_usage(train_opt)
                valid_mem = calculate_memory_usage(valid_opt)
                test_mem = calculate_memory_usage(test_opt)

                print(f"Memory usage after optimization: Train={train_mem:.2f}MB, Valid={valid_mem:.2f}MB, Test={test_mem:.2f}MB")

                # Save the processed data if requested
                if save:
                    save_dataset(train_opt, valid_opt, test_opt, 
                                data_dir=data_dir, prefix=intermediate_prefix, use_numpy=True)

                return train_opt, valid_opt, test_opt

        # Return unoptimized results if optimization is not requested
        if walk_forward_sets:
            return all_train_sets, all_valid_sets, (X_test_all, y_test_all)
        else:
            return (X_train_all, y_train_all), (X_valid_all, y_valid_all), (X_test_all, y_test_all)
        
    except Exception as e:
        print(f"[ERROR] Failed to stack all sets: {e}")
        return None, None, None
