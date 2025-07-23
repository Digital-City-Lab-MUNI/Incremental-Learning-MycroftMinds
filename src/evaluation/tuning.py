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
        verbose=False,
        tqdm_on=False
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

    def print_checkpoint(iteration, params, current_metrics, improvements, show_per_device=True):
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
        
        log_and_print(log_file, "\nCURRENT ITERATION RESULTS (Averages across devices):")
        for metric in ['rmse', 'mae', 'mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            if metric in current_metrics:
                is_perc = metric in ['mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']
                value_str = format_metric_value(current_metrics[metric], is_perc)
                log_and_print(log_file, f"   {metric.upper()}: {value_str}")
        
        log_and_print(log_file, "\nCURRENT BEST SCORES (OVERALL):")
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
        
        # Show per-device best scores (can make this optional)
        if show_per_device and 'per_device_best_scores_lower_better' in globals():
            log_and_print(log_file, "\nPER-DEVICE BEST SCORES (SUMMARY):")
            # Display a summary of best scores per device for each metric
            for device_id in sorted(per_device_best_scores_lower_better.keys()):
                rmse = per_device_best_scores_lower_better[device_id]['rmse']
                mae = per_device_best_scores_lower_better[device_id]['mae']
                mape = per_device_best_scores_lower_better[device_id]['mape']
                rmse_str = format_metric_value(rmse, False)
                mae_str = format_metric_value(mae, False)
                mape_str = format_metric_value(mape, True)

                dir_acc = per_device_best_scores_higher_better[device_id]['dir_accuracy']
                first_diff_corr = per_device_best_scores_higher_better[device_id]['first_diff_corr']
                spikes_f1 = per_device_best_scores_higher_better[device_id]['spikes_f1']
                drops_f1 = per_device_best_scores_higher_better[device_id]['drops_f1']
                dir_acc_str = format_metric_value(dir_acc, True)
                first_diff_corr_str = format_metric_value(first_diff_corr, True)
                spikes_f1_str = format_metric_value(spikes_f1, True)
                drops_f1_str = format_metric_value(drops_f1, True)

                log_and_print(log_file, f"   Device {device_id}: RMSE={rmse_str}, MAE={mae_str}, MAPE={mape_str}, Dir_Acc={dir_acc_str}, First_Diff_Corr={first_diff_corr_str}, Spikes_F1={spikes_f1_str}, Drops_F1={drops_f1_str}")

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
                                     forecast_horizon=forecast_horizon,
                                     update_freq=update_freq,
                                     start_split=start_split, 
                                     end_split=end_split)

        # Track metrics per device
        per_device_metrics = {}
        all_device_improvements = {}
        
        # Initialize structure to store best metrics per device
        if iteration == 1:
            # Create device-specific structures for best scores
            for device_id in results.keys():
                # For lower-better metrics
                if 'per_device_best_scores_lower_better' not in locals():
                    per_device_best_scores_lower_better = {}
                    per_device_best_params_lower_better = {}
                
                per_device_best_scores_lower_better[device_id] = {
                    'rmse': float('inf'),
                    'mae': float('inf'),
                    'mape': float('inf')
                }
                per_device_best_params_lower_better[device_id] = {
                    'rmse': None,
                    'mae': None,
                    'mape': None
                }
                
                # For higher-better metrics
                if 'per_device_best_scores_higher_better' not in locals():
                    per_device_best_scores_higher_better = {}
                    per_device_best_params_higher_better = {}
                
                per_device_best_scores_higher_better[device_id] = {
                    'dir_accuracy': float('-inf'),
                    'first_diff_corr': float('-inf'),
                    'spikes_f1': float('-inf'),
                    'drops_f1': float('-inf')
                }
                per_device_best_params_higher_better[device_id] = {
                    'dir_accuracy': None,
                    'first_diff_corr': None,
                    'spikes_f1': None,
                    'drops_f1': None
                }

        # Process metrics for each device
        for device_id, device_metrics in results.items():
            hourly_metrics = device_metrics[aggregation]
            per_device_metrics[device_id] = {}
            all_device_improvements[device_id] = {}
            
            # Basic metrics (lower is better)
            for metric in ['rmse', 'mae', 'mape']:
                if metric in hourly_metrics:
                    mean_val = hourly_metrics[metric]['mean']
                    if np.isfinite(mean_val):  # skip NaNs/Infs
                        per_device_metrics[device_id][metric] = mean_val
                        
                        # Check for improvement for this device
                        if mean_val < per_device_best_scores_lower_better[device_id][metric]:
                            all_device_improvements[device_id][metric] = (
                                per_device_best_scores_lower_better[device_id][metric], 
                                mean_val
                            )
                            per_device_best_scores_lower_better[device_id][metric] = float(mean_val)
                            per_device_best_params_lower_better[device_id][metric] = params.copy()
            
            # Higher is better metrics
            for metric, hourly_key in [
                ('dir_accuracy', 'dir_accuracy'), 
                ('first_diff_corr', 'first_diff_corr'),
                ('spikes_f1', 'spikes_immediate.f1'), 
                ('drops_f1', 'drops_immediate.f1')
            ]:
                if hourly_key in hourly_metrics:
                    mean_val = hourly_metrics[hourly_key]['mean']
                    if np.isfinite(mean_val):
                        per_device_metrics[device_id][metric] = mean_val
                        
                        # Check for improvement for this device
                        if mean_val > per_device_best_scores_higher_better[device_id][metric]:
                            all_device_improvements[device_id][metric] = (
                                per_device_best_scores_higher_better[device_id][metric], 
                                mean_val
                            )
                            per_device_best_scores_higher_better[device_id][metric] = float(mean_val)
                            per_device_best_params_higher_better[device_id][metric] = params.copy()
        
        # For display purposes, compute aggregates across devices
        metric_names = ['rmse', 'mae', 'mape', 'dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']
        current_metrics = {metric: 0.0 for metric in metric_names}
        metric_counts = {metric: 0 for metric in metric_names}
        
        for device_metrics in per_device_metrics.values():
            for metric, value in device_metrics.items():
                if np.isfinite(value):
                    current_metrics[metric] += value
                    metric_counts[metric] += 1
        
        # Average for display
        for metric in metric_names:
            if metric_counts[metric] > 0:
                current_metrics[metric] = current_metrics[metric] / metric_counts[metric]
            else:
                current_metrics[metric] = float('inf') if metric in ['rmse', 'mae', 'mape'] else float('-inf')
        
        # Track improvements for display purposes
        improvements = {}
        
        # Consolidate device improvements to report overall best metrics
        for device_id, device_improvements in all_device_improvements.items():
            for metric, (old_val, new_val) in device_improvements.items():
                # Only track one improvement per metric for display simplicity
                if metric not in improvements:
                    improvements[metric] = (old_val, new_val)
        
        # Update best scores for lower-better metrics globally (for backward compatibility)
        for metric in ['rmse', 'mae', 'mape']:
            for device_id, device_metrics in per_device_metrics.items():
                if metric in device_metrics:
                    if device_metrics[metric] < best_scores_lower_better[metric]:
                        best_scores_lower_better[metric] = float(device_metrics[metric])
                        best_params_lower_better[metric] = params.copy()
        
        # Update best scores for higher-better metrics globally (for backward compatibility)
        for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            for device_id, device_metrics in per_device_metrics.items():
                if metric in device_metrics:
                    if device_metrics[metric] > best_scores_higher_better[metric]:
                        best_scores_higher_better[metric] = float(device_metrics[metric])
                        best_params_higher_better[metric] = params.copy()
        
        # Print checkpoint every iteration
        print_checkpoint(iteration, params, current_metrics, improvements, show_per_device=(iteration % 5 == 0))
        
        # Save intermediate results periodically (every 10% or every 10 iterations)
        if save_intermediate and (iteration % max(1, total_combinations // 10) == 0 or iteration % 10 == 0):
            # Save overall best metrics
            best_params_combined = {**best_params_lower_better, **best_params_higher_better}
            best_scores_combined = {**best_scores_lower_better, **best_scores_higher_better}
            save_checkpoint_results(log_dir, f"{model_name}_intermediate", 
                                  best_params_combined, best_scores_combined, file_prefix)
            
            # Save per-device results
            per_device_results = {
                'per_device_best_params_lower_better': per_device_best_params_lower_better,
                'per_device_best_scores_lower_better': per_device_best_scores_lower_better,
                'per_device_best_params_higher_better': per_device_best_params_higher_better,
                'per_device_best_scores_higher_better': per_device_best_scores_higher_better
            }
            save_checkpoint_results(log_dir, f"{model_name}_intermediate_per_device", 
                                  per_device_results, {}, file_prefix)

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

    # Log per-device results
    log_and_print(log_file, "\nPER-DEVICE BEST RESULTS:")
    log_and_print(log_file, "=" * 80)
    
    for device_id in sorted(per_device_best_scores_lower_better.keys()):
        log_and_print(log_file, f"\nDevice: {device_id}")
        log_and_print(log_file, "-" * 40)
        
        # Lower is better metrics
        for metric in ['rmse', 'mae', 'mape']:
            score = per_device_best_scores_lower_better[device_id][metric]
            params = per_device_best_params_lower_better[device_id][metric]
            score_str = format_metric_value(score, metric == 'mape')
            log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
            if params:
                log_and_print(log_file, f"   Best Parameters: {params}")
        
        # Higher is better metrics
        for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            score = per_device_best_scores_higher_better[device_id][metric]
            params = per_device_best_params_higher_better[device_id][metric]
            score_str = format_metric_value(score, True)
            log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
            if params:
                log_and_print(log_file, f"   Best Parameters: {params}")
    
    # Save final results
    if save_intermediate:
        # Save both overall and per-device results
        save_checkpoint_results(log_dir, model_name, best_params, best_scores, file_prefix)
        
        # Save per-device results separately
        per_device_results = {
            'per_device_best_params_lower_better': per_device_best_params_lower_better,
            'per_device_best_scores_lower_better': per_device_best_scores_lower_better,
            'per_device_best_params_higher_better': per_device_best_params_higher_better,
            'per_device_best_scores_higher_better': per_device_best_scores_higher_better
        }
        save_checkpoint_results(log_dir, f"{model_name}_per_device", 
                              per_device_results, {}, file_prefix)

    # Return both overall and per-device results
    return {
        'overall': {
            'best_params': best_params,
            'best_scores': best_scores
        },
        'per_device': {
            'best_params_lower_better': per_device_best_params_lower_better,
            'best_scores_lower_better': per_device_best_scores_lower_better, 
            'best_params_higher_better': per_device_best_params_higher_better,
            'best_scores_higher_better': per_device_best_scores_higher_better
        }
    }
