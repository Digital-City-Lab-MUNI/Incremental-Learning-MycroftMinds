import os
import glob
import pandas as pd
import numpy as np
import argparse
import warnings

# ---------------------------
# Preprocessing Functions
# ---------------------------
def preprocess_df(dataframes):
    processed = []
    for df in dataframes:
        df['time'] = pd.to_datetime(df['time'], errors='coerce', utc=True)
        df['time'] = df['time'].dt.tz_convert(None)
        df.sort_values('time', inplace=True)
        df.reset_index(drop=True, inplace=True)

        if 'id' in df.columns:
            df = df.drop(columns=['id'])
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        df['second'] = df['time'].dt.second
        df['day'] = df['time'].dt.day
        df['month'] = df['time'].dt.month
        processed.append(df)
    return processed

def concat_dataframes_by_time(dataframes):
    """
    Concatenate a list of DataFrames into a single DataFrame and order the rows by 'time'.
    """
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df['time'] = pd.to_datetime(combined_df['time'], errors='coerce')
    combined_df.sort_values('time', inplace=True)
    combined_df.reset_index(drop=True, inplace=True)
    return combined_df

def get_complete_time_series(df):
    """
    Group the concatenated DataFrame by device_id and serial_number,
    then return a list of groups (DataFrames) that have no missing values.
    """
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

# ---------------------------
# Feature Engineering
# ---------------------------
def add_lag_features(df, columns, lag_freqs):
    df = df.copy()
    df['time'] = pd.to_datetime(df['time'], errors='coerce')
    df.sort_values('time', inplace=True)

    for col in columns:
        df[f'{col}_lag_1'] = df[col].shift(1)

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

    return df

# ---------------------------
# Create Supervised Data
# ---------------------------
def create_supervised_data_from_series(df, input_features, target_feature, window_size=1, step=1):
    X, y = [], []
    for i in range(window_size, len(df), step):
        window_data = df.iloc[i-window_size:i][input_features].values.flatten()
        X.append(window_data)
        y.append(df.iloc[i][target_feature])
    return np.array(X), np.array(y)

def clean_data(ts_set, input_features, target_feature, window_size, step):
    X_list, y_list = [], []
    for ts in ts_set:
        X_ts, y_ts = create_supervised_data_from_series(ts, input_features, target_feature, window_size, step)
        valid = ~np.isnan(X_ts).any(axis=1)
        X_list.append(X_ts[valid])
        y_list.append(y_ts[valid])
    return X_list, y_list


def get_train_test_sets(data_dir, lag_args, columns_to_add_lag, target_feature = 'total_power_active',
                        walk_forward_sets=False, train_months=3, valid_months=1):
    """
    data_dir: data directory
    lag_args: list of lag values in format ["feature=offset"]
    columns_to_add_lag: features to include in training (if None, include all features other than target feature)
    walk_forward_sets: True to use walk-forward validation
    train_months: number of months for training window (WFV)
    valid_months: number of months for validation window (WFV)

    Returns:
        (X_train, y_train), (X_valid, y_valid), (X_test, y_test)
    """
    import traceback
    warnings.filterwarnings("ignore")

    try:
        csv_files = glob.glob(os.path.join(data_dir, "smart_meter_data_*.csv"))
        dataframes = [pd.read_csv(file) for file in csv_files]
    except Exception as e:
        print(f"[ERROR] Failed to read CSV files: {e}")
        return None, None, None, None, None, None

    try:
        processed_dfs = preprocess_df(dataframes)
        data_concat = concat_dataframes_by_time(processed_dfs)
        data_concat.drop(columns='values_state', inplace=True, errors='ignore')
        complete_time_series = get_complete_time_series(data_concat)
    except Exception as e:
        print(f"[ERROR] Preprocessing failed: {e}")
        return None, None, None, None, None, None

    try:
        lag_freqs = {k: v for k, v in (item.split('=') for item in lag_args)}
    except Exception as e:
        print(f"[ERROR] Invalid lag_args format: {e}")
        return None, None, None, None, None, None

    if columns_to_add_lag is None:
        columns_to_add_lag = [
            'total_power_active', 'total_power_reactive', 'total_power_apparent',
            'import_energy_active', 'export_energy_active',
            'import_energy_reactive', 'export_energy_reactive',
            'total_energy_active', 'total_energy_reactive'
        ]

    input_features = [
        col for col in data_concat.columns
        if col not in ['time', 'device_id', 'serial_number']
    ]

    window_size = 1
    step = 1

    train_sets, valid_sets, test_sets = [], [], []

    for idx, ts in enumerate(complete_time_series):
        try:
            ts = ts.sort_values(by='time').reset_index(drop=True)
            ts['time'] = pd.to_datetime(ts['time'])

            train_end_idx = int(len(ts) * 0.6)
            valid_end_idx = int(len(ts) * 0.8)

            ts_test = ts.iloc[valid_end_idx:]
            ts_test_lagged = add_lag_features(ts_test, columns_to_add_lag, lag_freqs)
            X_test, y_test = clean_data([ts_test_lagged], input_features, target_feature, window_size, step)
            test_sets.append((np.vstack(X_test), np.hstack(y_test)))

            if not walk_forward_sets:
                ts_train = ts.iloc[:train_end_idx]
                ts_valid = ts.iloc[train_end_idx:valid_end_idx]

                ts_train_lagged = add_lag_features(ts_train, columns_to_add_lag, lag_freqs)
                ts_valid_lagged = add_lag_features(ts_valid, columns_to_add_lag, lag_freqs)

                X_train, y_train = clean_data([ts_train_lagged], input_features, target_feature, window_size, step)
                X_valid, y_valid = clean_data([ts_valid_lagged], input_features, target_feature, window_size, step)

                train_sets.append((np.vstack(X_train), np.hstack(y_train)))
                valid_sets.append((np.vstack(X_valid), np.hstack(y_valid)))

            else:
                start_time = ts['time'].min()
                current_start = start_time
                end_time = ts_test['time'].min()

                while current_start + pd.DateOffset(months=train_months + valid_months) <= end_time:
                    train_end = current_start + pd.DateOffset(months=train_months)
                    valid_end = train_end + pd.DateOffset(months=valid_months)

                    ts_train = ts[(ts['time'] >= current_start) & (ts['time'] < train_end)]
                    ts_valid = ts[(ts['time'] >= train_end) & (ts['time'] < valid_end)]

                    ts_train_lagged = add_lag_features(ts_train, columns_to_add_lag, lag_freqs)
                    ts_valid_lagged = add_lag_features(ts_valid, columns_to_add_lag, lag_freqs)

                    X_train, y_train = clean_data([ts_train_lagged], input_features, target_feature, window_size, step)
                    X_valid, y_valid = clean_data([ts_valid_lagged], input_features, target_feature, window_size, step)

                    if X_train and X_valid:
                        train_sets.append((np.vstack(X_train), np.hstack(y_train)))
                        valid_sets.append((np.vstack(X_valid), np.hstack(y_valid)))

                    current_start = current_start + pd.DateOffset(months=valid_months)

        except Exception as e:
            print(f"[WARNING] Skipped one time series (index {idx}) due to error:")
            traceback.print_exc()
            continue

    # Stack all sets
    try:
        X_train_all = np.vstack([x for x, _ in train_sets])
        y_train_all = np.hstack([y for _, y in train_sets])

        X_valid_all = np.vstack([x for x, _ in valid_sets])
        y_valid_all = np.hstack([y for _, y in valid_sets])

        X_test_all = np.vstack([x for x, _ in test_sets])
        y_test_all = np.hstack([y for _, y in test_sets])
    except Exception as e:
        print(f"[ERROR] Failed to stack all sets: {e}")
        return None, None, None, None, None, None

    print(f"Total Training samples: {X_train_all.shape[0]}")
    print(f"Total Validation samples: {X_valid_all.shape[0]}")
    print(f"Total Test samples: {X_test_all.shape[0]}")

    return (X_train_all, y_train_all), (X_valid_all, y_valid_all), (X_test_all, y_test_all)

