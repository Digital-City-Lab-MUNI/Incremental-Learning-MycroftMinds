import itertools
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from tqdm.notebook import tqdm  # Import tqdm for notebooks

from src.evaluation.metrics import calculate_metrics, all_time_aggregation_metrics
from src.evaluation.visualization import plot_part_predictions, plot_hourly_predictions
from src.evaluation.training import all_valid_splits_seq_fit_predict, test_set_seq_fit_predict

def train_and_evaluate(model: BaseEstimator, 
                       params: dict, 
                       train:List[Tuple[np.ndarray, np.ndarray]], 
                       valid:List[Tuple[np.ndarray, np.ndarray]], 
                       data_freq='H', 
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

    incremental=True
    if model in (RandomForestRegressor, GradientBoostingRegressor):
        incremental=False

    model = model(**params)
    
    model_all_train_preds, model_all_valid_preds, \
    model_all_train_metrics, model_all_valid_metrics = all_valid_splits_seq_fit_predict(
        model=model, 
        train=train, 
        valid=valid, 
        incremental=incremental,
        data_freq=data_freq,
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
                       update_freq: str = '1h',
                       start_split: int = 0,
                       end_split: Optional[int] = None) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Perform grid search to tune hyperparameters.
    Aggregates validation metrics across devices and returns
    the best params for each metric (rmse, mae, mape) separately.
    
    Args:
        param_grid (dict): Grid of hyperparameter values.
        model (BaseEstimator): The model to train and evaluate.
        train (List[Tuple[np.ndarray, np.ndarray]]): Training data as a list of tuples (X, y).
        valid (List[Tuple[np.ndarray, np.ndarray]]): Validation data as a list of tuples (X, y).
        aggregation (str): Aggregation method for metrics ('hourly', 'daily', etc.).
        data_freq (str): Frequency of the data (default is 'H' for hourly).
        update_freq (str): Frequency of updates (default is '1h').
        start_split (int): Starting index for the splits (default is 0).
        end_split (Optional[int]): Ending index for the splits (default is None, which means all splits).

    Returns:
        best_params (dict): Best hyperparameters for each metric based on average validation score across all devices for a given aggregation.
        best_scores (dict): Best scores for each metric based on average validation score across all devices for a given aggregation.
    """

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
    
    # Create progress bar
    for combination in tqdm(param_combinations, desc="Grid Search Progress"):
        params = dict(zip(keys, combination))
        results = train_and_evaluate(model, 
                                     params, 
                                     train=train, 
                                     valid=valid, 
                                     data_freq=data_freq, 
                                     update_freq=update_freq,
                                     start_split=start_split, 
                                     end_split=end_split)

        print('Training with:', params)

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

        # Update best scores for lower-better metrics
        for metric in ['rmse', 'mae', 'mape']:
            if avg_metrics_lower_better[metric] < best_scores_lower_better[metric]:
                best_scores_lower_better[metric] = float(avg_metrics_lower_better[metric])
                best_params_lower_better[metric] = params.copy()
        
        # Update best scores for higher-better metrics
        for metric in ['dir_accuracy', 'first_diff_corr', 'spikes_f1', 'drops_f1']:
            if avg_metrics_higher_better[metric] > best_scores_higher_better[metric]:
                best_scores_higher_better[metric] = float(avg_metrics_higher_better[metric])
                best_params_higher_better[metric] = params.copy()
                
        print('Scores (lower better):', avg_metrics_lower_better)
        print('Scores (higher better):', avg_metrics_higher_better)

    # Combine results
    best_params = {**best_params_lower_better, **best_params_higher_better}
    best_scores = {**best_scores_lower_better, **best_scores_higher_better}

    return best_params, best_scores
