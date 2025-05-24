import itertools
import numpy as np
from src.evaluation.metrics import calculate_metrics, all_time_aggregation_metrics
from src.evaluation.visualization import plot_part_predictions, plot_hourly_predictions
from src.evaluation.training import all_valid_splits_seq_fit_predict, test_set_seq_fit_predict

def train_and_evaluate(model, params, train, valid, data_freq='H', update_freq='1h'):

    """
    Fits and evaluate the model with given param sets

    Returns:
    - hoef_all_time_valid_metrics (dict): All time validation metrics.
    """

    model = model(**params)
    
    model_all_train_preds, model_all_valid_preds, \
    model_all_train_metrics, model_all_valid_metrics = all_valid_splits_seq_fit_predict(
        model=model, 
        train=train, 
        valid=valid, 
        incremental=True,
        data_freq=data_freq,
        update_freq=update_freq, 
        start_split=0, 
        #end_split=4,  # Use only the third and fourth splits
        verbose=False
    )

    model_all_time_valid_metrics = all_time_aggregation_metrics(model_all_valid_metrics, data_freq=data_freq, verbose=False)

    return model_all_time_valid_metrics

def grid_search_tuning(param_grid, model, train, valid, aggregation='hourly', data_freq='H', update_freq='1h'):
    """
    Perform grid search to tune hyperparameters.
    Aggregates validation metrics across devices and returns
    the best params for each metric (rmse, mae, mape) separately.
    
    Parameters:
    - param_grid (dict): Grid of hyperparameter values.
    - model (class): model to evaluate.
    - train: Training data.
    - valid: Validation data.

    Returns:
    - best_params (dict): Best hyperparameters for each metric based on average validation score across all devices for a given aggregation.
    - best_scores (dict): Best scores for each metric based on average validation score across all devices for a given aggregation.
    """
    keys = list(param_grid.keys())
    values = list(param_grid.values())

    best_scores = {
        'rmse': float('inf'),
        'mae': float('inf'),
        'mape': float('inf')
    }
    best_params = {
        'rmse': None,
        'mae': None,
        'mape': None
    }

    for combination in itertools.product(*values):
        params = dict(zip(keys, combination))
        results = train_and_evaluate(model, params, train, valid)

        print('training with:', params)

        # Aggregate hourly metrics across all devices
        metric_sums = {'rmse': 0.0, 'mae': 0.0, 'mape': 0.0}
        metric_counts = {'rmse': 0, 'mae': 0, 'mape': 0}

        for device_metrics in results.values():

            hourly_metrics = device_metrics[aggregation]
            for metric in ['rmse', 'mae', 'mape']:
                if metric in hourly_metrics:
                    mean_val = hourly_metrics[metric]['mean']
                    if np.isfinite(mean_val):  # skip NaNs/Infs
                        metric_sums[metric] += mean_val
                        metric_counts[metric] += 1

        # Compute averages for all devices
        avg_metrics = {metric: (metric_sums[metric] / metric_counts[metric])
                       if metric_counts[metric] > 0 else float('inf')
                       for metric in ['rmse', 'mae', 'mape']}

        for metric in ['rmse', 'mae', 'mape']:
            if avg_metrics[metric] < best_scores[metric]:
                best_scores[metric] = float(avg_metrics[metric])
                best_params[metric] = params.copy()
        #print('best scores:',  best_scores)
        print('scores:', avg_metrics)

    return best_params, best_scores
