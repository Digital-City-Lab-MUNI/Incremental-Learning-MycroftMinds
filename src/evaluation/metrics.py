from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Any

from src.utils.utils import aggregate_values

def directional_accuracy(y_true, y_pred):
    """
    Calculate the directional accuracy of predictions.
    Directional accuracy is the percentage of times the predicted direction
    matches the true direction.

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
    Returns:
        tuple: A tuple containing the directional accuracy and the first-difference correlation.
    """
    dy_true = np.diff(y_true).reshape(-1, 1)
    dy_pred = np.diff(y_pred).reshape(-1, 1)

    corr, _ = pearsonr(dy_true, dy_pred)  # First-diff correlation

    return np.mean(np.sign(dy_true) == np.sign(dy_pred)), corr[0]

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, device_id: str, period: str, data_freq: str, verbose=True) -> dict:
    """
    Calculate RMSE, MAE, MAPE, directional accuracy, and first-difference correlation for predictions.

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
        device_id (str): Device's ID for which the analysis was done.
        period (str): The period for which the metrics are calculated. Ideally in the format `'YYYY-MM-DD HH:MM:SS' - 'YYYY-MM-DD HH:MM:SS'`.
        data_freq (str): Frequency of original data. Can be '1s', '5min', '15min', 'H', 'D', etc.
        verbose (bool): If True, print the metrics.

    Returns:
        dict: A dictionary containing the metrics for different aggregation levels.
    """
    # Function to display metrics for a given period
    def display_metrics(period_name, metrics_dict):
        if metrics_dict:
            print(f"{period_name.capitalize()} Metrics:")
            print("-" * 50)
            print(f"\tRMSE: {metrics_dict['rmse']:.4f}")
            print(f"\tMAE: {metrics_dict['mae']:.4f}")
            print(f"\tMAPE: {metrics_dict['mape']:.4%}")
            print(f"\tDirectional Accuracy: {metrics_dict['dir_accuracy']:.4%}")
            print(f"\tFirst-Diff Correlation: {metrics_dict['first_diff_corr']:.4f}")
    
    # Function to calculate metrics for aggregated data
    def aggregate_metrics(y_true: pd.Series, y_pred: pd.Series, freq: str) -> dict:
        """Helper function to calculate metrics for aggregated data."""
        y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq)
        if freq in ['28D', 'W']:
            da, corr = 1, 1  # for weekly and 28 days, directional accuracy are always close to 1
        else:
            da, corr = directional_accuracy(y_true_agg, y_pred_agg)
        return {
            'rmse': np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)),
            'mae': mean_absolute_error(y_true_agg, y_pred_agg),
            'mape': mean_absolute_percentage_error(y_true_agg, y_pred_agg),
            'dir_accuracy': da,
            'first_diff_corr': corr
        }
    data_freq = data_freq.capitalize()

    # Create metrics dictionary
    metrics = {
        'device_id': device_id,
        'period': period,
    }
    
    # Calculate metrics for original data
    da_orig, corr_orig = directional_accuracy(y_true, y_pred)
    original_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'dir_accuracy': da_orig,
        'first_diff_corr': corr_orig
    }
    
    # Add metrics for original data
    if data_freq not in ['1H', 'H', 'D', 'W', '28D']:
        metrics[data_freq] = original_metrics
        
    # Add hourly metrics (either from original data if data_freq is 'H' or by aggregation)
    metrics['hourly'] = original_metrics if data_freq == 'H' else aggregate_metrics(y_true, y_pred, 'H')
    
    # Add other aggregation metrics
    metrics['daily'] = aggregate_metrics(y_true, y_pred, 'D')
    metrics['weekly'] = aggregate_metrics(y_true, y_pred, 'W')
    metrics['28_days'] = aggregate_metrics(y_true, y_pred, '28D')
    
    if verbose:
        print("=" * 50)
        print(f"Metrics for Device: {device_id} | Period: {period}")
        print("=" * 50)
        
        # Display metrics for original frequency if available
        if data_freq not in ['1H', 'H', 'D', 'W', '28D']:
            display_metrics(data_freq, metrics[data_freq])
            
        # Display metrics for each aggregation period
        for agg_period in ['hourly', 'daily', 'weekly', '28_days']:
            if agg_period != data_freq:
                print("-" * 50)
                display_metrics(agg_period, metrics[agg_period])
                
        print("=" * 50)

    if y_true.isnull().any():
        print(['[! WARNING !] There are missing values in the y_true!'])
    if y_pred.isnull().any():
        print(['[! WARNING !] There are missing values in the y_pred!'])

    return metrics

def all_time_aggregation_metrics(period_metrics: Dict[str, List[Dict]], data_freq: str, verbose=True) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Aggregate metrics across multiple periods.

    Args:
        period_metrics (dict): Dictionary containing metrics for different periods.
        data_freq (str): Frequency of original data. Can be '1s', '5min', '15min', 'H', 'D', etc.
        verbose (bool): If True, print the aggregated metrics.

    Returns:
        dict: A dictionary containing the aggregated metrics for each device.
    """
    aggregated_metrics: dict[str, # device_id
                             dict[str, # aggregation name (5_min, hourly, etc.)
                                  dict[   
                                    str, list[float]  # metric dict (rmse, mae, mape, etc.)
                                  ]
                            ]
                        ] = {}
    metric_names = ['rmse', 'mae', 'mape', 'dir_accuracy', 'first_diff_corr']
    for device_id, all_period_list in period_metrics.items():
        if device_id not in aggregated_metrics:
            aggregated_metrics[device_id] = {}
            if data_freq not in ['1H', 'H', 'D', 'W', '28D']:
                aggregated_metrics[device_id][data_freq] = {k: [] for k in metric_names}
            aggregated_metrics[device_id]['hourly'] = {k: [] for k in metric_names}
            aggregated_metrics[device_id]['daily'] = {k: [] for k in metric_names}
            aggregated_metrics[device_id]['weekly'] = {k: [] for k in metric_names}
            aggregated_metrics[device_id]['28_days'] = {k: [] for k in metric_names}
            
        for period_dict in all_period_list:
            for agg_name in aggregated_metrics[device_id].keys():
                for metric in metric_names:
                    value = period_dict[agg_name][metric]
                    aggregated_metrics[device_id][agg_name][metric].append(value)

    # Calculate mean and std for each metric
    for device_id, agg_dict in aggregated_metrics.items():
        for agg_name, metrics in agg_dict.items():
            for metric, values in metrics.items():
                mean_value = np.mean(values)
                std_value = np.std(values)
                aggregated_metrics[device_id][agg_name][metric] = {
                    'mean': mean_value,
                    'std': std_value
                }

    if verbose:
        print("\n\n" + "=" * 50)
        print("Aggregated Metrics Across Periods")
        print("=" * 50)
        for device_id, agg_dict in aggregated_metrics.items():
            print(f"Device ID: {device_id}")
            print("-" * 50)
            for agg_name, metrics in agg_dict.items():
                print(f"{agg_name.capitalize()} Metrics:")
                for metric, values in metrics.items():
                    if metric == 'mape':
                        print(f"\t{metric.upper()}: {values['mean']:.4%} ± {values['std']:.4%}")
                    elif metric == 'dir_accuracy':
                        print(f"\t{metric.replace('_', ' ').title()}: {values['mean']:.4%} ± {values['std']:.4%}")
                    elif metric == 'first_diff_corr':
                        print(f"\tFirst-Diff Correlation: {values['mean']:.4f} ± {values['std']:.4f}")
                    else:
                        print(f"\t{metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")
            print("=" * 50)

    return aggregated_metrics
