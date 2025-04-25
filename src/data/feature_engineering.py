import pandas as pd
import numpy as np
import traceback
from typing import List, Dict

def add_lag_features(df: pd.DataFrame, columns: List[str], lag_freqs: Dict[str, str]) -> pd.DataFrame:
    """
    Add lagged features to a DataFrame based on specified columns and lag frequencies.
    
    Args:
        df (pd.DataFrame): The input DataFrame with a 'time' column
        columns (List[str]): List of columns to create lag features for
        lag_freqs (Dict[str, str]): Dictionary mapping suffix names to lag frequencies
        
    Returns:
        pd.DataFrame: DataFrame with added lag features
    """
    try:
        df = df.copy()
        df['time'] = pd.to_datetime(df['time'], errors='coerce')
        df.sort_values('time', inplace=True)

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
        df['weekday_sin'] = np.sin(2 * np.pi * df['time'].dt.weekday / 7)
        df['weekday_cos'] = np.cos(2 * np.pi * df['time'].dt.weekday / 7)
        
        # Month of year (12 months)
        df['month_sin'] = np.sin(2 * np.pi * (df['time'].dt.month - 1) / 12)
        df['month_cos'] = np.cos(2 * np.pi * (df['time'].dt.month - 1) / 12)

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
