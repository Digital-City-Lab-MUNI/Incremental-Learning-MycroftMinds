"""Hyperparameter tuning utilities.

This module provides functions to (a) run a single end-to-end training / validation
cycle for a given model + parameter set (`train_and_evaluate`) and (b) perform a
grid or random search over a parameter space (`hyperparameter_search_tuning`).

Key features
------------
* Handles both incremental (online / partial_fit / learn_one) and batch models.
* Supports per-split walk-forward validation through the shared sequential
    evaluation helpers in :mod:`src.evaluation.training`.
* Aggregates metrics across devices as well as per-device best tracking.
* Logs rich intermediate checkpoints (overall + per-device) that can be resumed
    or audited later.

Returned metric structure
-------------------------
For each parameter combination we first obtain the per-split per-device metrics
and then aggregate them using ``all_time_aggregation_metrics``. During tuning we
track best parameters for both *lower-is-better* metrics (``rmse``, ``mae``,
``mape``) and *higher-is-better* metrics (``dir_accuracy``, ``first_diff_corr``,
``spikes_immediate.f1``, ``drops_immediate.f1``).

Notes
-----
Imports related to plotting or unused training helpers are intentionally kept
minimal here to avoid unnecessary dependencies during search.
"""

import itertools
import numpy as np
import random
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple
from sklearn.base import BaseEstimator
from tqdm.notebook import tqdm

from src.evaluation.metrics import all_time_aggregation_metrics
from src.evaluation.training import all_valid_splits_seq_fit_predict
from src.utils.logging import (
    setup_log_directory,
    create_log_file,
    log_and_print,
    save_checkpoint_results,
    format_metric_value,
)
import src.evaluation.constants as C


def train_and_evaluate(
    model: BaseEstimator,
    params: dict,
    train: List[Tuple[np.ndarray, np.ndarray]],
    valid: List[Tuple[np.ndarray, np.ndarray]],
    use_residuals: bool,
    spike_prominence: float,
    data_freq: str,
    forecast_horizon: str,
    features: List[str],
    update_freq: str,
    start_split: int,
    end_split: int,
) -> Dict[str, Any]:
    """Train and evaluate a single parameter configuration.

    This helper runs sequential walk-forward validation over the provided splits
    and returns time-aggregation metrics computed on the validation parts for
    each device.

    Parameters
    ----------
    model : BaseEstimator | type
        Either an initialized estimator instance or an estimator class. If a
        class (not yet instantiated) is provided it will be constructed with
        ``params`` for each split.
    params : dict
        Hyperparameters to apply when instantiating (or re-instantiating) the
        model for batch learners or when configuring incremental learners.
    train : list[tuple[np.ndarray, np.ndarray]]
        List of training split tuples ``(X, y)`` across devices.
    valid : list[tuple[np.ndarray, np.ndarray]]
        List of validation split tuples ``(X, y)`` across devices.
    use_residuals : bool
        If True, model the residuals (1-step differencing) instead of the original target.
    spike_prominence : float
        Prominence used in spike / drop detection within metric calculation.
    data_freq : str
        Base temporal frequency of the data (e.g. 'H', '15min').
    forecast_horizon : str
        Prediction horizon used to align model update timing (incremental
        learners) and to filter horizon-specific engineered features.
    features : list[str]
        Subset of feature names to keep. If ``None`` all columns are used.
    update_freq : str
        Frequency at which batch models are retrained (or incremental models
        are allowed to update) relative to ``forecast_horizon``.
    start_split : int
        First split index (inclusive) to evaluate.
    end_split : int
        Last split index (exclusive). If ``None`` all remaining splits are used.

    Returns
    -------
    dict
        Nested structure of aggregated validation metrics per device produced
        by ``all_time_aggregation_metrics``. Keys are device identifiers; each
        value contains aggregation levels (e.g. 'hourly', 'daily', 'weekly',
        '28_days') with metric summaries (``mean``, ``std``).
    """
    incremental = True
    # Check if the model is a class (not instantiated) and determine if it's batch learning
    if hasattr(model, "__name__"):
        if model.__name__ in ["RandomForestRegressor", "GradientBoostingRegressor"]:
            incremental = False
    elif model.__class__.__name__ in [
        "RandomForestRegressor",
        "GradientBoostingRegressor",
    ]:
        incremental = False

    (
        model_all_train_preds,
        model_all_valid_preds,
        model_all_train_metrics,
        model_all_valid_metrics,
    ) = all_valid_splits_seq_fit_predict(
        model_class=model,
        model_params=params,
        train=train,
        valid=valid,
        use_residuals=use_residuals,
        incremental=incremental,
        data_freq=data_freq,
        forecast_horizon=forecast_horizon,
        features=features,
        update_freq=update_freq,
        spike_prominence=spike_prominence,
        start_split=start_split,
        end_split=end_split,
        predict_train=False,
        verbose=False,
        tqdm_on=False,
    )

    model_all_time_valid_metrics = all_time_aggregation_metrics(
        model_all_valid_metrics, data_freq=data_freq, verbose=False
    )

    return model_all_time_valid_metrics


def hyperparameter_search_tuning(
    param_grid: dict,
    model: BaseEstimator,
    train: List[Tuple[np.ndarray, np.ndarray]],
    valid: List[Tuple[np.ndarray, np.ndarray]],
    use_residuals: bool,
    spike_prominence: float,
    aggregation: str,
    data_freq: str,
    forecast_horizon: str,
    features: Optional[List[str]] = None,
    update_freq: str = "14D",
    start_split: int = 0,
    end_split: Optional[int] = None,
    search_type: str = "grid",
    n_iter: Optional[int] = None,
    log_dir: Optional[str] = None,
    log_prefix: Optional[str] = None,
    file_prefix: Optional[str] = None,
    save_intermediate: bool = True,
) -> Dict[str, Any]:
    """Run a grid or random hyperparameter search.

    Combines sequential walk-forward evaluation with parameter space exploration.
    Tracks both overall (across devices) and per-device best hyperparameters for
    multiple metrics, logging rich checkpoints for auditability.

    Parameters
    ----------
    param_grid : dict
        Dictionary mapping parameter names to an iterable of candidate values.
        For ``search_type='grid'`` the full Cartesian product is evaluated.
        For ``'random'`` a fixed number of combinations is sampled uniformly.
    model : BaseEstimator | type
        Estimator instance or class. If a class is provided it is (re)created per
        split for batch models. Incremental / online models can be reused.
    train : list[tuple[np.ndarray, np.ndarray]]
        Training splits per device ``[(X_device_i, y_device_i), ...]``.
    valid : list[tuple[np.ndarray, np.ndarray]]
        Validation splits per device aligned with ``train``.
    use_residuals : bool
        If True, model the residuals (1-step differencing) instead of the original target.
    spike_prominence : float
        Prominence passed through to spike / drop detection in metrics.
    aggregation : str, default 'hourly'
        Aggregation key to select from the metric dictionary for scoring (e.g.
        'hourly', 'daily', 'weekly').
    data_freq : str, default 'H'
        Native data sampling frequency (Pandas offset alias).
    forecast_horizon : str, default '1h'
        Prediction horizon used for feature filtering & update cadence gating.
    features : list[str] or None, optional
        Subset of features to retain; ``None`` keeps all provided columns.
    update_freq : str, default '1h'
        Frequency at which batch models are retrained or incremental models may
        update. Typically equals the forecast horizon for pure one-step ahead.
    start_split : int, default 0
        First split index (inclusive) to include in tuning.
    end_split : int or None, optional
        Final split index (exclusive). If ``None`` uses all remaining splits.
    search_type : {'grid', 'random'}, default 'grid'
        Strategy for exploring ``param_grid``.
    n_iter : int or None, optional
        Number of random combinations when ``search_type='random'``. If ``None``
        defaults to 1000 samples.
    log_dir : str or None, optional
        Root directory for log artifacts. Created if missing. If ``None`` a
        timestamped directory under the project default is used.
    log_prefix : str or None, optional
        Prefix for the created log directory name. Defaults to
        'hyperparameter_tuning'.
    file_prefix : str or None, optional
        Prefix inserted in saved result filenames (e.g. dataset identifier).
    save_intermediate : bool, default True
        If ``True`` save periodic checkpoint JSON files with both overall and
        per-device best metrics / params.

    Returns
    -------
    dict
        Dictionary with two top-level keys:

        ``overall``
            Contains ``best_params`` and ``best_scores`` (each a mapping from
            metric name to the parameter dict / best scalar value).
        ``per_device``
            Contains four dictionaries tracking best parameters and scores for
            lower-better and higher-better metrics keyed by device id.

    Notes
    -----
    Metrics tracked (overall & per device):

    * Lower is better: ``rmse``, ``mae``, ``mape``
    * Higher is better: ``dir_accuracy``, ``first_diff_corr``, ``spikes_immediate.f1``,
      ``drops_immediate.f1``
    """
    # Setup logging with defaults
    model_name = model.__name__ if hasattr(model, "__name__") else model.__class__.__name__

    # Setup directories with defaults
    if log_prefix is None:
        log_prefix = "hyperparameter_tuning"

    log_dir = setup_log_directory(log_dir, log_prefix)
    log_file = create_log_file(log_dir, model_name, file_prefix)

    keys = list(param_grid.keys())
    values = list(param_grid.values())

    # Generate parameter combinations based on search_type
    if search_type == "random":
        # default to 1000 iterations if not specified
        if n_iter is None:
            n_iter = 1000
        param_combinations = [tuple(random.choice(vals) for vals in values) for _ in range(n_iter)]
        total_combinations = len(param_combinations)
    else:
        param_combinations = list(itertools.product(*values))
        total_combinations = len(param_combinations)

    # Initialize metrics that are better when lower
    best_scores_lower_better = {
        C.Metric.RMSE: float("inf"),
        C.Metric.MAE: float("inf"),
        C.Metric.MAPE: float("inf"),
    }
    best_params_lower_better = {
        C.Metric.RMSE: None,
        C.Metric.MAE: None,
        C.Metric.MAPE: None,
    }

    # Initialize metrics that are better when higher
    best_scores_higher_better = {
        C.Metric.DIR_ACCURACY: float("-inf"),
        C.Metric.FIRST_DIFF_CORR: float("-inf"),
        C.EventMetric.SPIKES_IMMEDIATE_F1: float("-inf"),
        C.EventMetric.DROPS_IMMEDIATE_F1: float("-inf"),
    }
    best_params_higher_better = {
        C.Metric.DIR_ACCURACY: None,
        C.Metric.FIRST_DIFF_CORR: None,
        C.EventMetric.SPIKES_IMMEDIATE_F1: None,
        C.EventMetric.DROPS_IMMEDIATE_F1: None,
    }

    def print_checkpoint(
        iteration: int,
        params: dict,
        current_metrics: dict,
        improvements: dict,
        show_per_device: bool = True,
        per_device_best_scores_lower_better: dict = None,
        per_device_best_params_lower_better: dict = None,
        per_device_best_scores_higher_better: dict = None,
        per_device_best_params_higher_better: dict = None,
    ) -> None:
        separator = "=" * 80

        log_and_print(log_file, f"\n{separator}")
        log_and_print(log_file, f"CHECKPOINT: Iteration {iteration}/{total_combinations}")
        log_and_print(log_file, f"Current Parameters: {params}")
        log_and_print(log_file, separator)

        if improvements:
            log_and_print(log_file, "NEW BEST SCORES FOUND!")
            for metric, (old_score, new_score) in improvements.items():
                is_perc = metric in [
                    C.Metric.MAPE,
                    C.Metric.DIR_ACCURACY,
                    C.Metric.FIRST_DIFF_CORR,
                    C.EventMetric.SPIKES_IMMEDIATE_F1,
                    C.EventMetric.DROPS_IMMEDIATE_F1,
                ]
                old_str = format_metric_value(old_score, is_perc)
                new_str = format_metric_value(new_score, is_perc)
                direction = "decreased" if metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE] else "increased"
                log_and_print(
                    log_file,
                    f"   {metric.upper()}: {old_str} -> {new_str} ({direction})",
                )

        log_and_print(log_file, "\nCURRENT ITERATION RESULTS (Averages across devices):")
        for metric in [
            C.Metric.RMSE,
            C.Metric.MAE,
            C.Metric.MAPE,
            C.Metric.DIR_ACCURACY,
            C.Metric.FIRST_DIFF_CORR,
            C.EventMetric.SPIKES_IMMEDIATE_F1,
            C.EventMetric.DROPS_IMMEDIATE_F1,
        ]:
            if metric in current_metrics:
                is_perc = metric in [
                    C.Metric.MAPE,
                    C.Metric.DIR_ACCURACY,
                    C.Metric.FIRST_DIFF_CORR,
                    C.EventMetric.SPIKES_IMMEDIATE_F1,
                    C.EventMetric.DROPS_IMMEDIATE_F1,
                ]
                value_str = format_metric_value(current_metrics[metric], is_perc)
                log_and_print(log_file, f"   {metric.upper()}: {value_str}")

        log_and_print(log_file, "\nCURRENT BEST SCORES (OVERALL):")
        # Lower is better metrics
        for metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE]:
            best_score = best_scores_lower_better[metric]
            best_params = best_params_lower_better[metric]
            score_str = format_metric_value(best_score, metric == C.Metric.MAPE)
            log_and_print(log_file, f"   {metric.upper()} (best: {score_str})")
            if best_params:
                log_and_print(log_file, f"     Parameters: {best_params}")

        # Higher is better metrics
        for metric in [
            C.Metric.DIR_ACCURACY,
            C.Metric.FIRST_DIFF_CORR,
            C.EventMetric.SPIKES_IMMEDIATE_F1,
            C.EventMetric.DROPS_IMMEDIATE_F1,
        ]:
            best_score = best_scores_higher_better[metric]
            best_params = best_params_higher_better[metric]
            score_str = format_metric_value(best_score, True)
            log_and_print(log_file, f"   {metric.upper()} (best: {score_str})")
            if best_params:
                log_and_print(log_file, f"     Parameters: {best_params}")

        # Show per-device best scores (can make this optional)
        if show_per_device:
            log_and_print(log_file, "\nPER-DEVICE BEST SCORES (SUMMARY):")
            # Display a summary of best scores per device for each metric
            for device_id in sorted(per_device_best_scores_lower_better.keys()):
                # Lower is better metrics
                rmse = per_device_best_scores_lower_better[device_id][C.Metric.RMSE]
                rmse_params = per_device_best_params_lower_better[device_id][C.Metric.RMSE]
                mae = per_device_best_scores_lower_better[device_id][C.Metric.MAE]
                mae_params = per_device_best_params_lower_better[device_id][C.Metric.MAE]
                mape = per_device_best_scores_lower_better[device_id][C.Metric.MAPE]
                mape_params = per_device_best_params_lower_better[device_id][C.Metric.MAPE]
                rmse_str = format_metric_value(rmse, False)
                mae_str = format_metric_value(mae, False)
                mape_str = format_metric_value(mape, True)

                # Higher is better metrics
                dir_acc = per_device_best_scores_higher_better[device_id][C.Metric.DIR_ACCURACY]
                dir_acc_params = per_device_best_params_higher_better[device_id][C.Metric.DIR_ACCURACY]
                first_diff_corr = per_device_best_scores_higher_better[device_id][C.Metric.FIRST_DIFF_CORR]
                first_diff_corr_params = per_device_best_params_higher_better[device_id][C.Metric.FIRST_DIFF_CORR]
                spikes_f1_val = per_device_best_scores_higher_better[device_id][C.EventMetric.SPIKES_IMMEDIATE_F1]
                spikes_f1_params = per_device_best_params_higher_better[device_id][C.EventMetric.SPIKES_IMMEDIATE_F1]
                drops_f1_val = per_device_best_scores_higher_better[device_id][C.EventMetric.DROPS_IMMEDIATE_F1]
                drops_f1_params = per_device_best_params_higher_better[device_id][C.EventMetric.DROPS_IMMEDIATE_F1]
                dir_acc_str = format_metric_value(dir_acc, True)
                first_diff_corr_str = format_metric_value(first_diff_corr, True)
                spikes_f1_str = format_metric_value(spikes_f1_val, True)
                drops_f1_str = format_metric_value(drops_f1_val, True)

                # Log per-device best scores
                log_and_print(log_file, f"   Device {device_id}: ")
                log_and_print(log_file, f"      RMSE={rmse_str}, {rmse_params}")
                log_and_print(log_file, f"      MAE={mae_str}, {mae_params}")
                log_and_print(log_file, f"      MAPE={mape_str}, {mape_params}")
                log_and_print(log_file, f"      Dir_Acc={dir_acc_str}, {dir_acc_params}")
                log_and_print(
                    log_file,
                    f"      First_Diff_Corr={first_diff_corr_str}, {first_diff_corr_params}",
                )
                log_and_print(log_file, f"      Spikes_F1={spikes_f1_str}, {spikes_f1_params}")
                log_and_print(log_file, f"      Drops_F1={drops_f1_str}, {drops_f1_params}")

        progress = (iteration / total_combinations) * 100
        log_and_print(
            log_file,
            f"\nProgress: {progress:.1f}% ({iteration}/{total_combinations} combinations)",
        )
        log_and_print(log_file, separator)

    # Initial log setup
    log_and_print(log_file, f"Grid Search Started: {datetime.now().strftime(C.Format.DATETIME)}")
    log_and_print(log_file, f"Model: {model_name}")
    log_and_print(log_file, f"File prefix: {file_prefix or 'None'}")
    log_and_print(log_file, f"Total parameter combinations: {total_combinations}")
    log_and_print(
        log_file,
        f"Aggregation: {aggregation} | Data Frequency: {data_freq} | Update Frequency: {update_freq}",
    )
    log_and_print(log_file, f"Parameter grid: {param_grid}")
    log_and_print(log_file, f"Log directory: {log_dir}")
    log_and_print(log_file, "")

    for iteration, combination in enumerate(tqdm(param_combinations, desc="Grid Search Progress"), 1):
        params = dict(zip(keys, combination, strict=False))
        results = train_and_evaluate(
            model,
            params,
            train=train,
            valid=valid,
            use_residuals=use_residuals,
            spike_prominence=spike_prominence,
            data_freq=data_freq,
            forecast_horizon=forecast_horizon,
            features=features,
            update_freq=update_freq,
            start_split=start_split,
            end_split=end_split,
        )

        # Track metrics per device
        per_device_metrics = {}
        all_device_improvements = {}

        # Initialize structure to store best metrics per device
        if iteration == 1:
            # Create device-specific structures for best scores
            for device_id in results.keys():
                # For lower-better metrics
                if "per_device_best_scores_lower_better" not in locals():
                    per_device_best_scores_lower_better = {}
                    per_device_best_params_lower_better = {}

                per_device_best_scores_lower_better[device_id] = {
                    C.Metric.RMSE: float("inf"),
                    C.Metric.MAE: float("inf"),
                    C.Metric.MAPE: float("inf"),
                }
                per_device_best_params_lower_better[device_id] = {
                    C.Metric.RMSE: None,
                    C.Metric.MAE: None,
                    C.Metric.MAPE: None,
                }

                # For higher-better metrics
                if "per_device_best_scores_higher_better" not in locals():
                    per_device_best_scores_higher_better = {}
                    per_device_best_params_higher_better = {}

                per_device_best_scores_higher_better[device_id] = {
                    C.Metric.DIR_ACCURACY: float("-inf"),
                    C.Metric.FIRST_DIFF_CORR: float("-inf"),
                    C.EventMetric.SPIKES_IMMEDIATE_F1: float("-inf"),
                    C.EventMetric.DROPS_IMMEDIATE_F1: float("-inf"),
                }
                per_device_best_params_higher_better[device_id] = {
                    C.Metric.DIR_ACCURACY: None,
                    C.Metric.FIRST_DIFF_CORR: None,
                    C.EventMetric.SPIKES_IMMEDIATE_F1: None,
                    C.EventMetric.DROPS_IMMEDIATE_F1: None,
                }

        # Process metrics for each device
        for device_id, device_metrics in results.items():
            hourly_metrics = device_metrics[aggregation]
            per_device_metrics[device_id] = {}
            all_device_improvements[device_id] = {}

            # Basic metrics (lower is better)
            for metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE]:
                if metric in hourly_metrics:
                    mean_val = hourly_metrics[metric]["mean"]
                    if np.isfinite(mean_val):  # skip NaNs/Infs
                        per_device_metrics[device_id][metric] = mean_val

                        # Check for improvement for this device
                        if mean_val < per_device_best_scores_lower_better[device_id][metric]:
                            all_device_improvements[device_id][metric] = (
                                per_device_best_scores_lower_better[device_id][metric],
                                mean_val,
                            )
                            per_device_best_scores_lower_better[device_id][metric] = float(mean_val)
                            per_device_best_params_lower_better[device_id][metric] = params.copy()

            # Higher is better metrics
            for metric in [
                C.Metric.DIR_ACCURACY,
                C.Metric.FIRST_DIFF_CORR,
                C.EventMetric.SPIKES_IMMEDIATE_F1,
                C.EventMetric.DROPS_IMMEDIATE_F1,
            ]:
                if metric in hourly_metrics:
                    mean_val = hourly_metrics[metric]["mean"]
                    if np.isfinite(mean_val):
                        per_device_metrics[device_id][metric] = mean_val

                        # Check for improvement for this device
                        if mean_val > per_device_best_scores_higher_better[device_id][metric]:
                            all_device_improvements[device_id][metric] = (
                                per_device_best_scores_higher_better[device_id][metric],
                                mean_val,
                            )
                            per_device_best_scores_higher_better[device_id][metric] = float(mean_val)
                            per_device_best_params_higher_better[device_id][metric] = params.copy()

        # For display purposes, compute aggregates across devices
        metric_names = [
            C.Metric.RMSE,
            C.Metric.MAE,
            C.Metric.MAPE,
            C.Metric.DIR_ACCURACY,
            C.Metric.FIRST_DIFF_CORR,
            C.EventMetric.SPIKES_IMMEDIATE_F1,
            C.EventMetric.DROPS_IMMEDIATE_F1,
        ]
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
                current_metrics[metric] = (
                    float("inf") if metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE] else float("-inf")
                )

        # Track improvements for display purposes
        improvements = {}

        # Consolidate device improvements to report overall best metrics
        for _device_id, device_improvements in all_device_improvements.items():
            for metric, (old_val, new_val) in device_improvements.items():
                # Only track one improvement per metric for display simplicity
                if metric not in improvements:
                    improvements[metric] = (old_val, new_val)

        # Update best scores for lower-better metrics globally (for backward compatibility)
        for metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE]:
            for _device_id, device_metrics in per_device_metrics.items():
                if metric in device_metrics:
                    if device_metrics[metric] < best_scores_lower_better[metric]:
                        best_scores_lower_better[metric] = float(device_metrics[metric])
                        best_params_lower_better[metric] = params.copy()

        # Update best scores for higher-better metrics globally (for backward compatibility)
        for metric in [
            C.Metric.DIR_ACCURACY,
            C.Metric.FIRST_DIFF_CORR,
            C.EventMetric.SPIKES_IMMEDIATE_F1,
            C.EventMetric.DROPS_IMMEDIATE_F1,
        ]:
            for _device_id, device_metrics in per_device_metrics.items():
                if metric in device_metrics:
                    if device_metrics[metric] > best_scores_higher_better[metric]:
                        best_scores_higher_better[metric] = float(device_metrics[metric])
                        best_params_higher_better[metric] = params.copy()

        # Print checkpoint every iteration
        print_checkpoint(
            iteration,
            params,
            current_metrics,
            improvements,
            show_per_device=(iteration % 5 == 0),
            per_device_best_scores_lower_better=per_device_best_scores_lower_better,
            per_device_best_params_lower_better=per_device_best_params_lower_better,
            per_device_best_scores_higher_better=per_device_best_scores_higher_better,
            per_device_best_params_higher_better=per_device_best_params_higher_better,
        )

        # Save intermediate results periodically (every 10% or every 10 iterations)
        if save_intermediate and (iteration % max(1, total_combinations // 10) == 0 or iteration % 10 == 0):
            # Save overall best metrics
            best_params_combined = {
                **best_params_lower_better,
                **best_params_higher_better,
            }
            best_scores_combined = {
                **best_scores_lower_better,
                **best_scores_higher_better,
            }
            save_checkpoint_results(
                log_dir,
                f"{model_name}_intermediate",
                best_params_combined,
                best_scores_combined,
                file_prefix,
            )

            # Save per-device results
            per_device_results = {
                "per_device_best_params_lower_better": per_device_best_params_lower_better,
                "per_device_best_scores_lower_better": per_device_best_scores_lower_better,
                "per_device_best_params_higher_better": per_device_best_params_higher_better,
                "per_device_best_scores_higher_better": per_device_best_scores_higher_better,
            }
            save_checkpoint_results(
                log_dir,
                f"{model_name}_intermediate_per_device",
                per_device_results,
                {},
                file_prefix,
            )

    # Final summary
    log_and_print(log_file, "\n" + "=" * 80)
    log_and_print(log_file, "GRID SEARCH COMPLETED!")
    log_and_print(log_file, "FINAL BEST RESULTS:")
    log_and_print(log_file, "=" * 80)

    # Combine results
    best_params = {**best_params_lower_better, **best_params_higher_better}
    best_scores = {**best_scores_lower_better, **best_scores_higher_better}

    for metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE]:
        score_str = format_metric_value(best_scores[metric], metric == C.Metric.MAPE)
        log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
        log_and_print(log_file, f"   Best Parameters: {best_params[metric]}")
        log_and_print(log_file, "")

    for metric in [
        C.Metric.DIR_ACCURACY,
        C.Metric.FIRST_DIFF_CORR,
        C.EventMetric.SPIKES_IMMEDIATE_F1,
        C.EventMetric.DROPS_IMMEDIATE_F1,
    ]:
        score_str = format_metric_value(best_scores[metric], True)
        log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
        log_and_print(log_file, f"   Best Parameters: {best_params[metric]}")
        log_and_print(log_file, "")

    log_and_print(log_file, "=" * 80)
    log_and_print(
        log_file,
        f"Grid Search Completed: {datetime.now().strftime(C.Format.DATETIME)}",
    )

    # Log per-device results
    log_and_print(log_file, "\nPER-DEVICE BEST RESULTS:")
    log_and_print(log_file, "=" * 80)

    for device_id in sorted(per_device_best_scores_lower_better.keys()):
        log_and_print(log_file, f"\nDevice: {device_id}")
        log_and_print(log_file, "-" * 40)

        # Lower is better metrics
        for metric in [C.Metric.RMSE, C.Metric.MAE, C.Metric.MAPE]:
            score = per_device_best_scores_lower_better[device_id][metric]
            params = per_device_best_params_lower_better[device_id][metric]
            score_str = format_metric_value(score, metric == C.Metric.MAPE)
            log_and_print(log_file, f"BEST {metric.upper()}: {score_str}")
            if params:
                log_and_print(log_file, f"   Best Parameters: {params}")

        # Higher is better metrics
        for metric in [
            C.Metric.DIR_ACCURACY,
            C.Metric.FIRST_DIFF_CORR,
            C.EventMetric.SPIKES_IMMEDIATE_F1,
            C.EventMetric.DROPS_IMMEDIATE_F1,
        ]:
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
            "per_device_best_params_lower_better": per_device_best_params_lower_better,
            "per_device_best_scores_lower_better": per_device_best_scores_lower_better,
            "per_device_best_params_higher_better": per_device_best_params_higher_better,
            "per_device_best_scores_higher_better": per_device_best_scores_higher_better,
        }
        save_checkpoint_results(log_dir, f"{model_name}_per_device", per_device_results, {}, file_prefix)

    # Return both overall and per-device results
    return {
        "overall": {"best_params": best_params, "best_scores": best_scores},
        "per_device": {
            "best_params_lower_better": per_device_best_params_lower_better,
            "best_scores_lower_better": per_device_best_scores_lower_better,
            "best_params_higher_better": per_device_best_params_higher_better,
            "best_scores_higher_better": per_device_best_scores_higher_better,
        },
    }
