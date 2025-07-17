from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, mean_absolute_error
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from typing import Dict, List, Any
from scipy.signal import find_peaks

from src.utils.utils import aggregate_values

def normalize_zscore(y):
    return (y - np.mean(y)) / np.std(y)

def robust_zscore(y):
    y = np.array(y)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return (y - med) / (mad if mad > 1e-6 else 1)

def detect_spikes_and_drops(y, prominence, normalize_robust):
    """
    Detect spikes (peaks) and drops (valleys) in time series data.
    
    Args:
        y (array-like): Input signal.
        prominence (float): The prominence parameter for spike detection metrics. The data is normalized before calculating the spikes, so the prominence now means distance in std or MAD units (1.5 => 1.5*std from mean).
        normalize_robust (bool): If True, use robust (median&MAD) normalization, else use regular z-score (mean&std)normalization.

    Returns:
        tuple: Indices of peaks and indices of drops.
    """
    if normalize_robust:
        y = robust_zscore(y)
    else:
        y = normalize_zscore(y)
    peaks, _ = find_peaks(y, prominence=prominence)
    drops, _ = find_peaks(-y, prominence=prominence)
    return peaks, drops


def match_events(true_idxs, pred_idxs, delta):
    """
    Match detected events (spikes or drops) between true and predicted data.
    
    Args:
        true_idxs (array-like): Indices of events in true data.
        pred_idxs (array-like): Indices of events in predicted data.
        delta (int): Maximum allowed distance between matching events.
        
    Returns:
        tuple: Number of matched events, total true events, total predicted events.
    """
    matched = []
    used_pred = set()
    for t in true_idxs:
        for p in pred_idxs:
            if abs(p - t) <= delta and p not in used_pred:
                matched.append((t, p))
                used_pred.add(p)
                break
    return len(matched), len(true_idxs), len(pred_idxs)


def spike_drop_match_score(y_true, y_pred, prominence, normalize_robust, delta=0):
    """
    Calculate metrics for spike and drop detection between true and predicted values.
    
    Args:
        y_true (array-like): True values.
        y_pred (array-like): Predicted values.
        prominence (float): The prominence parameter for spike detection metrics. The data is normalized before calculating the spikes, so the prominence now means distance in std or MAD units (1.5 => 1.5*std from mean).
        normalize_robust (bool): If True, use robust (median&MAD) normalization, else use regular z-score (mean&std)normalization.
        delta (int): Maximum allowed time distance for matching peaks.
        
    Returns:
        dict: Dictionary with precision, recall, and F1 scores for spikes and drops.
    """
    true_peaks, true_drops = detect_spikes_and_drops(y_true, prominence, normalize_robust=normalize_robust)
    pred_peaks, pred_drops = detect_spikes_and_drops(y_pred, prominence, normalize_robust=normalize_robust)

    # Match spikes
    spike_match, n_true_spikes, n_pred_spikes = match_events(true_peaks, pred_peaks, delta)
    drop_match, n_true_drops, n_pred_drops = match_events(true_drops, pred_drops, delta)

    def score(n_match, n_true, n_pred):
        precision = n_match / n_pred if n_pred > 0 else 0
        recall = n_match / n_true if n_true > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        return precision, recall, f1

    spike_score = score(spike_match, n_true_spikes, n_pred_spikes)
    drop_score = score(drop_match, n_true_drops, n_pred_drops)

    return {
        "spikes": {"precision": spike_score[0], "recall": spike_score[1], "f1": spike_score[2],
                  "true_indices": true_peaks, "pred_indices": pred_peaks},
        "drops":  {"precision": drop_score[0], "recall": drop_score[1], "f1": drop_score[2],
                  "true_indices": true_drops, "pred_indices": pred_drops},
    }


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

def calculate_metrics(y_true: pd.Series, y_pred: pd.Series, device_id: str, period: str, data_freq: str = "5min", 
                      spike_prominence=10.0, normalize_robust=True, verbose=True) -> dict:
    """
    Calculate RMSE, MAE, MAPE, directional accuracy, first-difference correlation, and spike/drop metrics for predictions.

    Args:
        y_true (pd.Series): True values.
        y_pred (pd.Series): Predicted values.
        device_id (str): Device's ID for which the analysis was done.
        period (str): The period for which the metrics are calculated. Ideally in the format `'YYYY-MM-DD HH:MM:SS' - 'YYYY-MM-DD HH:MM:SS'`.
        data_freq (str): Frequency of original data. Can be '1s', '5min', '15min', 'H', 'D', etc.
        spike_prominence (float): The prominence parameter for spike detection metrics. The data is normalized before calculating the spikes, so the prominence now means distance in std or MAD units (1.5 => 1.5*std from mean).
        normalize_robust (bool): If True, use robust (median&MAD) normalization, else use regular z-score (mean&std)normalization.
        verbose (bool): If True, print the metrics.

    Returns:
        dict: A dictionary containing the metrics for different aggregation levels.
    """
    # Function to display metrics for a given period
    def display_metrics(period_name, metrics_dict):
        if metrics_dict:
            print(f"{period_name.capitalize()} Metrics:")
            print("-" * 50)
            
            # Basic accuracy metrics
            print("Basic Prediction Metrics:")
            print(f"\tRMSE: {metrics_dict['rmse']:.4f}")
            print(f"\tMAE: {metrics_dict['mae']:.4f}")
            print(f"\tMAPE: {metrics_dict['mape']:.4%}")
            
            # Only show directional and spike metrics for sub-weekly aggregations
            if period_name not in ['weekly', '28_days']:
                # Directional metrics
                print("\nDirectional Metrics:")
                print(f"\tDirectional Accuracy: {metrics_dict['dir_accuracy']:.4%}")
                print(f"\tFirst-Diff Correlation: {metrics_dict['first_diff_corr']:.4f}")
                
                # Spike and drop detection - immediate (delta=0)
                print("\nImmediate Detection (delta=0):")
                print(f"\tSpikes:")
                print(f"\t\tPrecision: {metrics_dict['spikes_immediate']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['spikes_immediate']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['spikes_immediate']['f1']:.4f}")
                print(f"\tDrops:")
                print(f"\t\tPrecision: {metrics_dict['drops_immediate']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['drops_immediate']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['drops_immediate']['f1']:.4f}")
                
                # Spike and drop detection - delayed (delta=1)
                print("\nDelayed Detection (delta=1 => 1-sec or 1-hour, or 1-day detection delay):")
                print(f"\tSpikes:")
                print(f"\t\tPrecision: {metrics_dict['spikes_delayed']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['spikes_delayed']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['spikes_delayed']['f1']:.4f}")
                print(f"\tDrops:")
                print(f"\t\tPrecision: {metrics_dict['drops_delayed']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['drops_delayed']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['drops_delayed']['f1']:.4f}")
    
    # Function to calculate metrics for aggregated data
    def aggregate_metrics(y_true: pd.Series, y_pred: pd.Series, freq: str) -> dict:
        """Helper function to calculate metrics for aggregated data."""

        freq_td = pd.Timedelta(freq)

        # If the frequency is higher than 1 hour, first aggregate to hourly (mean), then to the target freq (sum)
        # This is done to get the correct kWh (energy) values from kW (power) data
        if freq_td > pd.Timedelta("1H"):
            # First aggregate to hourly by mean
            y_true_hourly, y_pred_hourly = aggregate_values(y_true, y_pred, "H", 'mean')
            # Then aggregate to the target frequency by sum
            y_true_agg, y_pred_agg = aggregate_values(y_true_hourly, y_pred_hourly, freq, 'sum')
        elif freq_td == pd.Timedelta("1H"):
            # Convert from kW to kWh by aggregating to hourly (mean)
            y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq, 'mean')
        else:
            # For lower frequencies (e.g., 15-min, 5-min), aggregate by mean
            print(f"[! WARNING !]: For frequencies lower than 1 hour, the aggregation is done by mean, which may not\
                   represent the actual `energy` consumption in kWh, but rather the average `power` (kW) consumption over the period.")
            y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq, 'mean')

        # Create basic result dictionary with accuracy metrics
        result = {
            'rmse': np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)),
            'mae': mean_absolute_error(y_true_agg, y_pred_agg),
            'mape': mean_absolute_percentage_error(y_true_agg, y_pred_agg),
        }
        
        # Only add directional and spike metrics for sub-weekly aggregations
        if freq_td < pd.Timedelta("7D"):
            da, corr = directional_accuracy(y_true_agg, y_pred_agg)
            
            # Calculate spike and drop metrics with delta=0 (immediate detection)
            spike_metrics_immediate = spike_drop_match_score(y_true_agg.values, y_pred_agg.values, prominence=spike_prominence,
                                                             normalize_robust=normalize_robust, delta=0)

            # Calculate spike and drop metrics with delta=1 (1-time_unit delayed detection)
            spike_metrics_delayed = spike_drop_match_score(y_true_agg.values, y_pred_agg.values, prominence=spike_prominence, 
                                                           normalize_robust=normalize_robust, delta=1)

            # Add directional and spike detection metrics
            result.update({
                'dir_accuracy': da,
                'first_diff_corr': corr,
                'spikes_immediate': spike_metrics_immediate['spikes'],
                'drops_immediate': spike_metrics_immediate['drops'],
                'spikes_delayed': spike_metrics_delayed['spikes'],
                'drops_delayed': spike_metrics_delayed['drops'],
            })
        else:
            # Add placeholder values for weekly and higher aggregations
            # This ensures the dictionary has all keys but with None values for weekly+ metrics
            result.update({
                'dir_accuracy': None,
                'first_diff_corr': None,
                'spikes_immediate': {'precision': None, 'recall': None, 'f1': None},
                'drops_immediate': {'precision': None, 'recall': None, 'f1': None},
                'spikes_delayed': {'precision': None, 'recall': None, 'f1': None},
                'drops_delayed': {'precision': None, 'recall': None, 'f1': None},
            })
            
        return result

    data_freq = data_freq.capitalize() if isinstance(data_freq, str) else data_freq

    # Create metrics dictionary
    metrics = {
        'device_id': device_id,
        'period': period,
    }
    
    # Calculate metrics for original data
    da_orig, corr_orig = directional_accuracy(y_true, y_pred)
    
    # Calculate spike and drop metrics for original data with delta=0 (immediate detection)  
    spike_metrics_immediate = spike_drop_match_score(y_true.values, y_pred.values, 
                                                   prominence=spike_prominence,
                                                   normalize_robust=normalize_robust,
                                                   delta=0)
    
    # Calculate spike and drop metrics for original data with delta=1 (1-hour delayed detection)
    spike_metrics_delayed = spike_drop_match_score(y_true.values, y_pred.values,
                                                   prominence=spike_prominence, 
                                                   normalize_robust=normalize_robust, 
                                                   delta=1)

    # Initialize original metrics with all values
    original_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'mape': mean_absolute_percentage_error(y_true, y_pred),
        'dir_accuracy': da_orig,
        'first_diff_corr': corr_orig,
        'spikes_immediate': spike_metrics_immediate['spikes'],
        'drops_immediate': spike_metrics_immediate['drops'],
        'spikes_delayed': spike_metrics_delayed['spikes'],
        'drops_delayed': spike_metrics_delayed['drops'],
    }
    
    # Add metrics for original data
    if data_freq not in ['1H', 'H', 'D', 'W', '28D']:
        metrics[data_freq] = original_metrics
        
    # Add hourly metrics (either from original data if data_freq is 'H' or by aggregation)
    metrics['hourly'] = original_metrics if data_freq == 'H' else aggregate_metrics(y_true, y_pred, '1H')
    
    # Add other aggregation metrics
    metrics['daily'] = aggregate_metrics(y_true, y_pred, '1D')
    metrics['weekly'] = aggregate_metrics(y_true, y_pred, '1W')
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
    
    # Immediate spike detection metrics (delta=0)
    immediate_metrics = ['spikes_immediate.precision', 'spikes_immediate.recall', 'spikes_immediate.f1', 
                         'drops_immediate.precision', 'drops_immediate.recall', 'drops_immediate.f1']
    
    # 1-Hour delayed spike metrics (delta=1)
    delayed_metrics = ['spikes_delayed.precision', 'spikes_delayed.recall', 'spikes_delayed.f1', 
                      'drops_delayed.precision', 'drops_delayed.recall', 'drops_delayed.f1']
    
    for device_id, all_period_list in period_metrics.items():
        if device_id not in aggregated_metrics:
            aggregated_metrics[device_id] = {}
            if data_freq not in ['1H', 'H', 'D', 'W', '28D']:
                aggregated_metrics[device_id][data_freq] = {k: [] for k in metric_names}
                # Add all spike metrics
                for sm in immediate_metrics + delayed_metrics:
                    aggregated_metrics[device_id][data_freq][sm] = []
            
            for agg_level in ['hourly', 'daily', 'weekly', '28_days']:
                aggregated_metrics[device_id][agg_level] = {k: [] for k in metric_names}
                # Add all spike metrics
                for sm in immediate_metrics + delayed_metrics:
                    aggregated_metrics[device_id][agg_level][sm] = []
            
        for period_dict in all_period_list:
            for agg_name in aggregated_metrics[device_id].keys():
                for metric in metric_names:
                    value = period_dict[agg_name][metric]
                    aggregated_metrics[device_id][agg_name][metric].append(value)
                                
                # Handle immediate spike metrics (delta=0)
                aggregated_metrics[device_id][agg_name]['spikes_immediate.precision'].append(period_dict[agg_name]['spikes_immediate']['precision'])
                aggregated_metrics[device_id][agg_name]['spikes_immediate.recall'].append(period_dict[agg_name]['spikes_immediate']['recall'])
                aggregated_metrics[device_id][agg_name]['spikes_immediate.f1'].append(period_dict[agg_name]['spikes_immediate']['f1'])
                
                aggregated_metrics[device_id][agg_name]['drops_immediate.precision'].append(period_dict[agg_name]['drops_immediate']['precision'])
                aggregated_metrics[device_id][agg_name]['drops_immediate.recall'].append(period_dict[agg_name]['drops_immediate']['recall'])
                aggregated_metrics[device_id][agg_name]['drops_immediate.f1'].append(period_dict[agg_name]['drops_immediate']['f1'])
                
                # Handle delayed spike metrics (delta=1)
                aggregated_metrics[device_id][agg_name]['spikes_delayed.precision'].append(period_dict[agg_name]['spikes_delayed']['precision'])
                aggregated_metrics[device_id][agg_name]['spikes_delayed.recall'].append(period_dict[agg_name]['spikes_delayed']['recall'])
                aggregated_metrics[device_id][agg_name]['spikes_delayed.f1'].append(period_dict[agg_name]['spikes_delayed']['f1'])
                
                aggregated_metrics[device_id][agg_name]['drops_delayed.precision'].append(period_dict[agg_name]['drops_delayed']['precision'])
                aggregated_metrics[device_id][agg_name]['drops_delayed.recall'].append(period_dict[agg_name]['drops_delayed']['recall'])
                aggregated_metrics[device_id][agg_name]['drops_delayed.f1'].append(period_dict[agg_name]['drops_delayed']['f1'])

    # Calculate mean and std for each metric
    for device_id, agg_dict in aggregated_metrics.items():
        for agg_name, metrics in agg_dict.items():
            for metric, values in metrics.items():
                # Handle None values for weekly+ aggregations
                if agg_name in ['weekly', '28_days'] and metric not in ['rmse', 'mae', 'mape']:
                    # For directional and spike/drop metrics in weekly+ aggregations, store None
                    aggregated_metrics[device_id][agg_name][metric] = {
                        'mean': None,
                        'std': None
                    }
                else:
                    # Filter out None values before calculation
                    valid_values = [v for v in values if v is not None]
                    if valid_values:
                        mean_value = np.mean(valid_values)
                        std_value = np.std(valid_values)
                    else:
                        mean_value = None
                        std_value = None
                        
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
                
                # Basic accuracy metrics
                for metric in ['rmse', 'mae', 'mape']:
                    if metrics[metric]['mean'] is not None:
                        if metric == 'mape':
                            print(f"\t{metric.upper()}: {metrics[metric]['mean']:.4%} ± {metrics[metric]['std']:.4%}")
                        else:
                            print(f"\t{metric.upper()}: {metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}")
                    else:
                        print(f"\t{metric.upper()}: N/A")
                
                # Only show directional and spike metrics for sub-weekly aggregations
                if agg_name not in ['weekly', '28_days']:
                    # Directional metrics
                    print("\n\tDirectional Metrics:")
                    print(f"\t\tDirectional Accuracy: {metrics['dir_accuracy']['mean']:.4%} ± {metrics['dir_accuracy']['std']:.4%}")
                    print(f"\t\tFirst-Diff Correlation: {metrics['first_diff_corr']['mean']:.4f} ± {metrics['first_diff_corr']['std']:.4f}")
                    
                    # Immediate detection metrics
                    print("\n\tImmediate Detection (delta=0):")
                    print(f"\t\tSpike Precision: {metrics['spikes_immediate.precision']['mean']:.4f} ± {metrics['spikes_immediate.precision']['std']:.4f}")
                    print(f"\t\tSpike Recall: {metrics['spikes_immediate.recall']['mean']:.4f} ± {metrics['spikes_immediate.recall']['std']:.4f}")
                    print(f"\t\tSpike F1: {metrics['spikes_immediate.f1']['mean']:.4f} ± {metrics['spikes_immediate.f1']['std']:.4f}")
                    print(f"\t\tDrop Precision: {metrics['drops_immediate.precision']['mean']:.4f} ± {metrics['drops_immediate.precision']['std']:.4f}")
                    print(f"\t\tDrop Recall: {metrics['drops_immediate.recall']['mean']:.4f} ± {metrics['drops_immediate.recall']['std']:.4f}")
                    print(f"\t\tDrop F1: {metrics['drops_immediate.f1']['mean']:.4f} ± {metrics['drops_immediate.f1']['std']:.4f}")
                    
                    # Delayed detection metrics
                    print("\n\t1-Hour Delayed Detection (delta=1):")
                    print(f"\t\tSpike Precision: {metrics['spikes_delayed.precision']['mean']:.4f} ± {metrics['spikes_delayed.precision']['std']:.4f}")
                    print(f"\t\tSpike Recall: {metrics['spikes_delayed.recall']['mean']:.4f} ± {metrics['spikes_delayed.recall']['std']:.4f}")
                    print(f"\t\tSpike F1: {metrics['spikes_delayed.f1']['mean']:.4f} ± {metrics['spikes_delayed.f1']['std']:.4f}")
                    print(f"\t\tDrop Precision: {metrics['drops_delayed.precision']['mean']:.4f} ± {metrics['drops_delayed.precision']['std']:.4f}")
                    print(f"\t\tDrop Recall: {metrics['drops_delayed.recall']['mean']:.4f} ± {metrics['drops_delayed.recall']['std']:.4f}")
                    print(f"\t\tDrop F1: {metrics['drops_delayed.f1']['mean']:.4f} ± {metrics['drops_delayed.f1']['std']:.4f}")
                
            print("=" * 50)

    return aggregated_metrics
