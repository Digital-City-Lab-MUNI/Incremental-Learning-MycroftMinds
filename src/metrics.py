from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from typing import Dict, List, Any

# Import aggregate_values from utils to avoid circular import
from src.utils import aggregate_values

def calculate_metrics(y_true, y_pred, device_id:str, period: str, verbose=True) -> dict:
    """
    Calculate RMSE, MAE, and MAPE for predictions.

    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        device_id (str): Device type, either 'cpu' or 'gpu'.
        period (str): The period for which the metrics are calculated. Ideally in the format `'YYYY-MM-DD HH:MM:SS' - 'YYYY-MM-DD HH:MM:SS'`.
        verbose (bool): If True, print the metrics.

    Returns:
        dict: A dictionary containing the metrics for different aggregation levels.
    """
    def aggregate_metrics(y_true: pd.Series, y_pred: pd.Series, freq: str) -> dict:
        """Helper function to calculate metrics for aggregated data."""
        y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq)
        return {
            'rmse': np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)),
            'mae': mean_absolute_error(y_true_agg, y_pred_agg),
            'mape': mean_absolute_percentage_error(y_true_agg, y_pred_agg)
        }

    metrics = {
        'device_id': device_id,
        'period': period,
        '5_min': {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred)
        },
        'hourly': aggregate_metrics(y_true, y_pred, 'H'),
        'daily': aggregate_metrics(y_true, y_pred, 'D'),
        'weekly': aggregate_metrics(y_true, y_pred, 'W'),
        '28_days': aggregate_metrics(y_true, y_pred, '28D')
    }
    
    if verbose:
        print("=" * 50)
        print(f"Metrics for Device: {device_id} | Period: {period}")
        print("=" * 50)
        print("5-Minute Metrics:")
        print("-" * 50)
        print(f"\tRMSE: {metrics['5_min']['rmse']:.4f}")
        print(f"\tMAE: {metrics['5_min']['mae']:.4f}")
        print(f"\tMAPE: {metrics['5_min']['mape']:.4%}")
        for agg_period, agg_metrics in metrics.items():
            if agg_period not in ['period', '5_min', 'device_id']:
                print("-" * 50)
                print(f"{agg_period.capitalize()} Metrics:")
                print("-" * 50)
                print(f"\tRMSE: {agg_metrics['rmse']:.4f}")
                print(f"\tMAE: {agg_metrics['mae']:.4f}")
                print(f"\tMAPE: {agg_metrics['mape']:.4%}")
        print("=" * 50)

    return metrics

def all_time_aggregation_metrics(period_metrics: Dict[str, List[Dict]], verbose=True) -> Dict[str, Dict[str, List[Dict[str, float]]]]:
    """
    Aggregate metrics across multiple periods.

    Args:
        period_metrics (dict): Dictionary containing metrics for different periods.
        verbose (bool): If True, print the aggregated metrics.

    Returns:
        dict: A dictionary containing the aggregated metrics for each device.
    """
    aggregated_metrics: dict[str, # device_id
                             dict[str, # aggregation name (5_min, hourly, etc.)
                                  list[   
                                    dict[str, float]  # metric dict (rmse, mae, mape)
                                  ]
                            ]
                        ] = {}
    for device_id, all_period_list in period_metrics.items():
        if device_id not in aggregated_metrics:
            aggregated_metrics[device_id] = {
                '5_min': {'rmse': [], 'mae': [], 'mape': []},
                'hourly': {'rmse': [], 'mae': [], 'mape': []},
                'daily': {'rmse': [], 'mae': [], 'mape': []},
                'weekly': {'rmse': [], 'mae': [], 'mape': []},
                '28_days': {'rmse': [], 'mae': [], 'mape': []}
            }
        for period_dict in all_period_list:
            for agg_name in aggregated_metrics[device_id].keys():
                for metric, value in period_dict[agg_name].items():
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
                    print(f"\t{metric.upper()}: {values['mean']:.4f} ± {values['std']:.4f}")
            print("=" * 50)

    return aggregated_metrics
