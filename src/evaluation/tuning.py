import itertools
import numpy as np
from datetime import datetime
from typing import Tuple, Optional, List, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm.notebook import tqdm

from src.evaluation.metrics import calculate_metrics, all_time_aggregation_metrics
from src.evaluation.visualization import plot_part_predictions, plot_hourly_predictions
from src.evaluation.training import all_valid_splits_seq_fit_predict, test_set_seq_fit_predict
from src.utils.logging import (
    setup_log_directory, create_log_file, log_and_print, 
    save_checkpoint_results, format_metric_value
)

def train_and_evaluate(model: BaseEstimator, 
                       params: dict, 
                       train:List[Tuple[np.ndarray, np.ndarray]], 
                       valid:List[Tuple[np.ndarray, np.ndarray]], 
                       data_freq='H',
                       forecast_horizon='1h',
                       update_freq='1h',
                       start_split: int = 0,
                       end_split: Optional[int] = None) -> Dict[str, Any]:

    """
    Fits and evaluate the model with given param sets

    Args:
        model (BaseEstimator): The model to train and evaluate.
        params (dict): Hyperparameters for the model.
        train (List[Tuple[np.ndarray, np.ndarray]]): Training data as a list of tuples (X, y).
        valid (List[Tuple[np.ndarray, np.ndarray]]): Validation data as a list of tuples (X, y).
        data_freq (str): Frequency of the data (default is 'H' for hourly).
        update_freq (str): Frequency of updates (default is '1h').
        start_split (int): Starting index for the splits (default is 0).
        end_split (Optional[int]): Ending index for the splits (default is None, which means all splits).
    
    Returns:
        model_all_time_valid_metrics (Dict[str, Any]): Aggregated validation metrics across all devices.
    """

    incremental = True
    # Check if the model is a class (not instantiated) and determine if it's batch learning
    if hasattr(model, '__name__'):
        if model.__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor']:
            incremental = False
    elif model.__class__.__name__ in ['RandomForestRegressor', 'GradientBoostingRegressor']:
        incremental = False
    
    model_all_train_preds, model_all_valid_preds, \
    model_all_train_metrics, model_all_valid_metrics = all_valid_splits_seq_fit_predict(
        model_class=model, 
        model_params=params,
        train=train, 
        valid=valid,
        incremental=incremental,
        data_freq=data_freq,
        forecast_horizon=forecast_horizon,
        update_freq=update_freq, 
        start_split=start_split, 
        end_split=end_split,
        predict_train=False,
        verbose=False
    )

    model_all_time_valid_metrics = all_time_aggregation_metrics(
        model_all_valid_metrics, 
        data_freq=data_freq, 
        verbose=False)

    return model_all_time_valid_metrics

def grid_search_tuning(param_grid: dict, 
                       model: BaseEstimator, 
                       train: List[Tuple[np.ndarray, np.ndarray]], 
                       valid: List[Tuple[np.ndarray, np.ndarray]], 
                       aggregation: str = 'hourly', 
                       data_freq: str = 'H', 
                       forecast_horizon: str = '1h',
                       update_freq: str = '1h',
                       start_split: int = 0,
                       end_split: Optional[int] = None,
                       log_dir: str = None,
                       log_prefix: str = None,
                       file_prefix: str = None,
                       save_intermediate: bool = True) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform grid search to tune hyperparameters with conservative checkpoint output and logging.
    Aggregates validation metrics across devices and returns
    the best params for each metric (rmse, mae, mape) separately.
    
    Args:
        param_grid (dict): Grid of hyperparameter values.
        model (BaseEstimator): The model to train and evaluate.
        train (List[Tuple[np.ndarray, np.ndarray]]): Training data as a list of tuples (X, y).
        valid (List[Tuple[np.ndarray, np.ndarray]]): Validation data as a list of tuples (X, y).
        aggregation (str): Aggregation method for metrics ('hourly', 'daily', etc.).
        data_freq (str): Frequency of the data (default is 'H' for hourly).
        forecast_horizon (str): Forecast horizon (default is '1h').
        update_freq (str): Frequency of updates (default is '1h').
        start_split (int): Starting index for the splits (default is 0).
        end_split (Optional[int]): Ending index for the splits (default is None, which means all splits).
        log_dir (str): Directory to save tuning results logs. If None, uses default.
        log_prefix (str): Prefix for log directory and files. If None, uses 'tuning'.
        file_prefix (str): Prefix for result filenames (e.g., 'dev_4_wfv'). If None, uses no prefix.
        save_intermediate (bool): Whether to save intermediate results.

    Returns:
        best_params (dict): Best hyperparameters for each metric.
        best_scores (dict): Best scores for each metric.
    """

    # Setup logging with defaults
    model_name = model.__name__ if hasattr(model, '__name__') else model.__class__.__name__
    
    # Setup directories with defaults
    if log_prefix is None:
        log_prefix = "hyperparameter_tuning"
    
    log_dir = setup_log_directory(log_dir, log_prefix)
    log_file = create_log_file(log_dir, model_name, file_prefix)
    
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Initialize metrics that are better when lower
    best_scores_lower_better = {
        'rmse': float('inf'),
        'mae': float('inf'),
        'mape': float('inf')
    }
    best_params_lower_better = {
        'rmse': None,
        'mae': None,
        'mape': None
    }
    
    # Initialize metrics that are better when higher
    best_scores_higher_better = {
        'dir_accuracy': float('-inf'),
        'first_diff_corr': float('-inf'),
        'spikes_f1': float('-inf'),
        'drops_f1': float('-inf')
    }
    best_params_higher_better = {
        'dir_accuracy': None,
        'first_diff_corr': None,
        'spikes_f1': None,
        'drops_f1': None
    }

    # Generate all parameter combinations
    param_combinations = list(itertools.product(*values))
    total_combinations = len(param_combinations)
    
    def print_checkpoint(iteration, params, current_metrics, improvements):
        separator = "=" * 80
        
        log_and_print(log_file, f"\n{separator}")
        log_and_print(log_file, f"CHECKPOINT: Iteration {iteration}/{total_combinations}")
        log_and_print(log_file, f"Current Parameters: {params}")
        log_and_print(log_file, separator)
        
        if improvements:
            log_and_print(log_file, "NEW BEST SCORES FOUND!")
            for metric, (old_score, new_score) in improvements.items():
                is_perc = metric in ['mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']
                old_str = format_metric_value(old_score, is_perc)
                new_str = format_metric_value(new_score, is_perc)
                direction = "decreased" if metric in ['rmse', 'mae', 'mape'] else "increased"
                log_and_print(log_file, f"   {metric.upper()}: {old_str} -> {new_str} ({direction})")
        
        log_and_print(log_file, "\nCURRENT ITERATION RESULTS:")
        for metric in ['rmse', 'mae', 'mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            if metric in current_metrics:
                is_perc = metric in ['mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']
                value_str = format_metric_value(current_metrics[metric], is_perc)
                log_and_print(log_file, f"   {metric.upper()}: {value_str}")
        
        log_and_print(log_file, "\nCURRENT BEST SCORES:")
        # Lower is better metrics
        for metric in ['rmse', 'mae', 'mape']:
            best_score = best_scores_lower_better[metric]
            best_params = best_params_lower_better[metric]
            score_str = format_metric_value(best_score, metric == 'mape')
            log_and_print(log_file, f"   {metric.upper()} (best: {score_str})")
            if best_params:
                log_and_print(log_file, f"     Parameters: {best_params}")
        
        # Higher is better metrics
        for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            best_score = best_scores_higher_better[metric]
            best_params = best_params_higher_better[metric]
            score_str = format_metric_value(best_score, True)
            log_and_print(log_file, f"   {metric.upper()} (best: {score_str})")
            if best_params:
                log_and_print(log_file, f"     Parameters: {best_params}")
        
        progress = (iteration / total_combinations) * 100
        log_and_print(log_file, f"\nProgress: {progress:.1f}% ({iteration}/{total_combinations} combinations)")
        log_and_print(log_file, separator)
    
    # Initial log setup
    log_and_print(log_file, f"Grid Search Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log_and_print(log_file, f"Model: {model_name}")
    log_and_print(log_file, f"File prefix: {file_prefix or 'None'}")
    log_and_print(log_file, f"Total parameter combinations: {total_combinations}")
    log_and_print(log_file, f"Aggregation: {aggregation} | Data Frequency: {data_freq} | Update Frequency: {update_freq}")
    log_and_print(log_file, f"Parameter grid: {param_grid}")
    log_and_print(log_file, f"Log directory: {log_dir}")
    log_and_print(log_file, "")
        
    for iteration, combination in enumerate(tqdm(param_combinations, desc="Grid Search Progress"), 1):
        params = dict(zip(keys, combination))
        results = train_and_evaluate(model, 
                                     params, 
                                     train=train, 
                                     valid=valid, 
                                     data_freq=data_freq, 
                                     update_freq=update_freq,
                                     start_split=start_split, 
                                     end_split=end_split)

        # Initialize aggregate metrics
        metric_sums = {
            'rmse': 0.0, 'mae': 0.0, 'mape': 0.0,
            'dir_accuracy': 0.0, 'first_diff_corr': 0.0,
            'spikes_f1': 0.0, 'drops_f1': 0.0
        }
        metric_counts = {
            'rmse': 0, 'mae': 0, 'mape': 0,
            'dir_accuracy': 0, 'first_diff_corr': 0,
            'spikes_f1': 0, 'drops_f1': 0
        }

        for device_metrics in results.values():
            hourly_metrics = device_metrics[aggregation]
            
            # Basic metrics (lower is better)
            for metric in ['rmse', 'mae', 'mape']:
                if metric in hourly_metrics:
                    mean_val = hourly_metrics[metric]['mean']
                    if np.isfinite(mean_val):  # skip NaNs/Infs
                        metric_sums[metric] += mean_val
                        metric_counts[metric] += 1
            
            # Higher is better metrics
            if 'dir_accuracy' in hourly_metrics:
                mean_val = hourly_metrics['dir_accuracy']['mean']
                if np.isfinite(mean_val):
                    metric_sums['dir_accuracy'] += mean_val
                    metric_counts['dir_accuracy'] += 1
            
            if 'first_diff_corr' in hourly_metrics:
                mean_val = hourly_metrics['first_diff_corr']['mean']
                if np.isfinite(mean_val):
                    metric_sums['first_diff_corr'] += mean_val
                    metric_counts['first_diff_corr'] += 1
            
            # Spikes F1
            if 'spikes_immediate.f1' in hourly_metrics:
                mean_val = hourly_metrics['spikes_immediate.f1']['mean']
                if np.isfinite(mean_val):
                    metric_sums['spikes_f1'] += mean_val
                    metric_counts['spikes_f1'] += 1
                    
            # Drops F1
            if 'drops_immediate.f1' in hourly_metrics:
                mean_val = hourly_metrics['drops_immediate.f1']['mean']
                if np.isfinite(mean_val):
                    metric_sums['drops_f1'] += mean_val
                    metric_counts['drops_f1'] += 1

        # Compute averages for all devices
        avg_metrics_lower_better = {
            metric: (metric_sums[metric] / metric_counts[metric])
            if metric_counts[metric] > 0 else float('inf')
            for metric in ['rmse', 'mae', 'mape']
        }
        
        avg_metrics_higher_better = {
            metric: (metric_sums[metric] / metric_counts[metric])
            if metric_counts[metric] > 0 else float('-inf')
            for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']
        }
        
        # Combine current metrics for display
        current_metrics = {**avg_metrics_lower_better, **avg_metrics_higher_better}
        
        # Track improvements
        improvements = {}
        
        # Update best scores for lower-better metrics
        for metric in ['rmse', 'mae', 'mape']:
            if avg_metrics_lower_better[metric] < best_scores_lower_better[metric]:
                improvements[metric] = (best_scores_lower_better[metric], avg_metrics_lower_better[metric])
                best_scores_lower_better[metric] = float(avg_metrics_lower_better[metric])
                best_params_lower_better[metric] = params.copy()
        
        # Update best scores for higher-better metrics
        for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            if avg_metrics_higher_better[metric] > best_scores_higher_better[metric]:
                improvements[metric] = (best_scores_higher_better[metric], avg_metrics_higher_better[metric])
                best_scores_higher_better[metric] = float(avg_metrics_higher_better[metric])
                best_params_higher_better[metric] = params.copy()
        
        # Print checkpoint every iteration
        print_checkpoint(iteration, params, current_metrics, improvements)
        
        # Save intermediate results periodically (every 10% or every 10 iterations)
        if save_intermediate and (iteration % max(1, total_combinations // 10) == 0 or iteration % 10 == 0):
            best_params_combined = {**best_params_lower_better, **best_params_higher_better}
            best_scores_combined = {**best_scores_lower_better, **best_scores_higher_better}
            save_checkpoint_results(log_dir, f"{model_name}_intermediate", best_params_combined, best_scores_combined, file_prefix)

    # Final summary
    log_and_print(log_file, "\n" + "=" * 80)
    log_and_print(log_file, "GRID SEARCH COMPLETED!")
    log_and_print(log_file, "FINAL BEST RESULTS:")
    log_and_print(log_file, "=" * 80)
    
    # Combine results
    best_params = {**best_params_lower_better, **best_params_higher_better}
    best_scores = {**best_scores_lower_better, **best_scores_higher_better}

    for metric in ['rmse', 'mae', 'mape']:
        score_str = format_metric_value(best_scores[metric], metric == 'mape')
        log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
        log_and_print(log_file, f"   Best Parameters: {best_params[metric]}")
        log_and_print(log_file, "")
    
    for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
        score_str = format_metric_value(best_scores[metric], True)
        log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
        log_and_print(log_file, f"   Best Parameters: {best_params[metric]}")
        log_and_print(log_file, "")
    
    log_and_print(log_file, "=" * 80)
    log_and_print(log_file, f"Grid Search Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Save final results
    if save_intermediate:
        save_checkpoint_results(log_dir, model_name, best_params, best_scores, file_prefix)

    return best_params, best_scores
