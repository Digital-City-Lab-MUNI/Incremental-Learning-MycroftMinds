"""Evaluation metrics and spike/drop detection utilities.

This module provides helper functions to compute:

* Standard regression accuracy metrics (RMSE, MAE, MAPE)
* Directional accuracy and first difference correlation
* Event matching metrics for spikes (peaks) and drops (valleys) at multiple
  aggregation granularities

Spike / drop detection is performed by normalizing the series (either using a
standard z-score or a robust median/MAD z-score) and then calling
``scipy.signal.find_peaks`` on the series and on its negation (for drops).

The main public entry points are ``calculate_metrics`` for per-period metrics
and ``all_time_aggregation_metrics`` for aggregation across periods and devices.
"""

from typing import Dict, List, Iterable, Tuple, Any
from sklearn.metrics import (
    mean_absolute_percentage_error,
    mean_squared_error,
    mean_absolute_error,
)
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from scipy.signal import find_peaks

from src.utils.utils import aggregate_values  # Removed unused memory import


def normalize_zscore(y: np.ndarray) -> np.ndarray:
    """Return standard z-score normalized array.

    Parameters
    ----------
    y : array-like
        Input numeric sequence.

    Returns
    -------
    numpy.ndarray
        Element-wise z-score (mean 0, std 1) of ``y``.
    """
    return (y - np.mean(y)) / np.std(y)


def robust_zscore(y: np.ndarray) -> np.ndarray:
    """Return robust (median/MAD) z-score normalized array.

    The robust score is computed as ``(y - median) / MAD`` where MAD is the
    median absolute deviation (with a small safeguard to avoid division by a
    near-zero MAD).

    Parameters
    ----------
    y : array-like
        Input numeric sequence.

    Returns
    -------
    numpy.ndarray
        Robust z-score values.
    """
    y = np.array(y)
    med = np.median(y)
    mad = np.median(np.abs(y - med))
    return (y - med) / (mad if mad > 1e-6 else 1)


def detect_spikes_and_drops(y: np.ndarray, prominence: float, normalize_robust: bool) -> Tuple[np.ndarray, np.ndarray]:
    """Detect spike (peak) and drop (valley) indices.

    Parameters
    ----------
    y : array-like
        Input time-ordered signal.
    prominence : float
        Minimum prominence for a spike/drop after normalization. Interpreted in
        standard deviation units (z-score) or MAD units (robust mode).
    normalize_robust : bool
        If True, apply robust (median/MAD) normalization; otherwise standard
        mean/std normalization.

    Returns
    -------
    tuple[numpy.ndarray, numpy.ndarray]
        ``(peaks, drops)`` indices of detected events.
    """
    if normalize_robust:
        y = robust_zscore(y)
    else:
        y = normalize_zscore(y)
    peaks, _ = find_peaks(y, prominence=prominence)
    drops, _ = find_peaks(-y, prominence=prominence)
    return peaks, drops


def match_events(true_idxs: Iterable[int], pred_idxs: Iterable[int], delta: int) -> Tuple[int, int, int]:
    """Match detected events (spikes or drops) between true and predicted data.

    Each true index is matched to at most one predicted index (first eligible
    in order) whose absolute difference is ``<= delta``.

    Parameters
    ----------
    true_idxs : array-like
        Indices of true events (e.g. spike positions).
    pred_idxs : array-like
        Indices of predicted events.
    delta : int
        Maximum allowed absolute index distance for a match.

    Returns
    -------
    tuple[int, int, int]
        ``(n_matches, n_true, n_pred)`` counts for matched, true, and predicted
        events.
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


def spike_drop_match_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    prominence: float,
    normalize_robust: bool,
    delta: int = 0,
) -> Dict[str, Dict[str, Any]]:
    """Compute spike/drop precision, recall and F1 matching scores.

    Detection is performed on normalized series (standard or robust). Predicted
    and true event indices are greedily matched within ``delta`` positions.

    Parameters
    ----------
    y_true : array-like
        Ground truth numeric sequence.
    y_pred : array-like
        Predicted numeric sequence.
    prominence : float
        Minimum prominence threshold passed to ``find_peaks`` after
        normalization.
    normalize_robust : bool
        Whether to use robust (median/MAD) normalization.
    delta : int, default=0
        Maximum index distance allowed for matching.

    Returns
    -------
    dict
        Nested dictionary with keys ``spikes`` and ``drops``. Each contains
        ``precision``, ``recall``, ``f1``, ``true_indices`` and ``pred_indices``.
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
        "spikes": {
            "precision": spike_score[0],
            "recall": spike_score[1],
            "f1": spike_score[2],
            "true_indices": true_peaks,
            "pred_indices": pred_peaks,
        },
        "drops": {
            "precision": drop_score[0],
            "recall": drop_score[1],
            "f1": drop_score[2],
            "true_indices": true_drops,
            "pred_indices": pred_drops,
        },
    }


def directional_accuracy(
    y_true: Iterable[float] | np.ndarray | pd.Series,
    y_pred: Iterable[float] | np.ndarray | pd.Series,
) -> Tuple[float, float]:
    """Compute direction hit-rate and first-difference correlation.

    Directional accuracy counts how often the sign of the successive change
    (first difference) in ``y_pred`` matches that of ``y_true``.

    Parameters
    ----------
    y_true : pandas.Series or array-like
        True target values ordered in time.
    y_pred : pandas.Series or array-like
        Predicted target values (aligned with ``y_true``).

    Returns
    -------
    tuple[float, float]
        ``(directional_accuracy, diff_correlation)`` where the second value is
        the Pearson correlation coefficient of first differences.
    """
    dy_true = np.diff(y_true).reshape(-1, 1)
    dy_pred = np.diff(y_pred).reshape(-1, 1)

    corr, _ = pearsonr(dy_true, dy_pred)  # First-diff correlation

    return np.mean(np.sign(dy_true) == np.sign(dy_pred)), corr[0]


def calculate_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    device_id: str,
    period: str,
    data_freq: str = "5min",
    spike_prominence=10.0,
    normalize_robust=True,
    model=None,
    verbose=True,
) -> dict:
    """Compute accuracy, directional and spike/drop metrics at multiple scales.

    Metrics are produced for the original sampling frequency (unless already an
    aggregation label like 'H', 'D', etc.) and for hourly, daily, weekly and a
    28-day aggregation. Spike/drop precision/recall/F1 are reported for both
    immediate detection (``delta=0``) and a one-step delayed allowance
    (``delta=1``). Directional metrics are omitted for weekly and 28-day levels.

    Parameters
    ----------
    y_true : pandas.Series
        Ground truth target series indexed by timestamp.
    y_pred : pandas.Series
        Predicted target series aligned to ``y_true``.
    device_id : str
        Identifier of the device / source series.
    period : str
        Human readable period string (e.g. 'YYYY-MM-DD ... - YYYY-MM-DD ...').
    data_freq : str, default='5min'
        Original data frequency (e.g. '1s', '5min', '15min', 'H', 'D').
    spike_prominence : float, default=10.0
        Prominence threshold (in normalized units) for spike detection.
    normalize_robust : bool, default=True
        Whether to apply robust normalization (median/MAD) prior to peak
        detection.
    model : Any, optional
        Trained model instance (used only for potential memory measurements –
        currently commented out).
    verbose : bool, default=True
        If True, pretty-print metrics to stdout.

    Returns
    -------
    dict
        Dictionary keyed by aggregation level (original freq key + 'hourly',
        'daily', 'weekly', '28_days') each holding its metric sub-dictionary.
    """

    # Function to display metrics for a given period
    def display_metrics(period_name, metrics_dict):
        """Pretty-print metrics for a single aggregation level.

        Parameters
        ----------
        period_name : str
            Aggregation key label.
        metrics_dict : dict
            Metric dictionary for that aggregation level.
        """
        if metrics_dict:
            print(f"{period_name.capitalize()} Metrics:")
            print("-" * 50)

            # Basic accuracy metrics
            print("Basic Prediction Metrics:")
            print(f"\tRMSE: {metrics_dict['rmse']:.4f}")
            print(f"\tMAE: {metrics_dict['mae']:.4f}")
            print(f"\tMAPE: {metrics_dict['mape']:.4%}")

            # Memory usage metric
            # if 'memory' in metrics_dict and metrics_dict['memory'] is not None:
            #     print(f"\tModel Memory: {metrics_dict['memory']:.5f} MB")

            # Only show directional and spike metrics for sub-weekly aggregations
            if period_name not in ["weekly", "28_days"]:
                # Directional metrics
                print("\nDirectional Metrics:")
                print(f"\tDirectional Accuracy: {metrics_dict['dir_accuracy']:.4%}")
                print(f"\tFirst-Diff Correlation: {metrics_dict['first_diff_corr']:.4f}")

                # Spike and drop detection - immediate (delta=0)
                print("\nImmediate Detection (delta=0):")
                print("\tSpikes:")
                print(f"\t\tPrecision: {metrics_dict['spikes_immediate']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['spikes_immediate']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['spikes_immediate']['f1']:.4f}")
                print("\tDrops:")
                print(f"\t\tPrecision: {metrics_dict['drops_immediate']['precision']:.4f}")
                # Spike and drop detection - immediate (delta=0)
                print(f"\t\tRecall: {metrics_dict['drops_immediate']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['drops_immediate']['f1']:.4f}")
                print("\nDelayed Detection (delta=1 => 1-sec or 1-hour, or 1-day " "detection delay):")
                print("\tSpikes:")
                print(f"\t\tPrecision: {metrics_dict['spikes_delayed']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['spikes_delayed']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['spikes_delayed']['f1']:.4f}")
                print("\tDrops:")
                print(f"\t\tPrecision: {metrics_dict['drops_delayed']['precision']:.4f}")
                print(f"\t\tRecall: {metrics_dict['drops_delayed']['recall']:.4f}")
                print(f"\t\tF1: {metrics_dict['drops_delayed']['f1']:.4f}")

    # Function to calculate metrics for aggregated data
    def aggregate_metrics(y_true: pd.Series, y_pred: pd.Series, freq: str) -> dict:
        """Aggregate series to ``freq`` then compute metrics.

        Parameters
        ----------
        y_true : pandas.Series
            Original ground truth series.
        y_pred : pandas.Series
            Predicted series.
        freq : str
            Pandas offset alias for aggregation (e.g., '1H', '1D').

        Returns
        -------
        dict
            Metric dictionary for the aggregated series (similar schema as
            original frequency metrics) omitting directional / spike metrics
            for weekly and longer periods.
        """

        freq_td = pd.Timedelta(freq)

        # If the frequency is higher than 1 hour, first aggregate to hourly (mean), then to the target freq (sum)
        # This is done to get the correct kWh (energy) values from kW (power) data
        if freq_td > pd.Timedelta("1H"):
            # First aggregate to hourly by mean
            y_true_hourly, y_pred_hourly = aggregate_values(y_true, y_pred, "H", "mean")
            # Then aggregate to the target frequency by sum
            y_true_agg, y_pred_agg = aggregate_values(y_true_hourly, y_pred_hourly, freq, "sum")
        elif freq_td == pd.Timedelta("1H"):
            # Convert from kW to kWh by aggregating to hourly (mean)
            y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq, "mean")
        else:
            # For lower frequencies (e.g., 15-min, 5-min), aggregate by mean
            print(
                "[! WARNING !]: For frequencies lower than 1 hour, the aggregation is done by mean, which may not\
                   represent the actual `energy` consumption in kWh, but rather the average `power` (kW) consumption over the period."
            )
            y_true_agg, y_pred_agg = aggregate_values(y_true, y_pred, freq, "mean")

        # Create basic result dictionary with accuracy metrics
        result = {
            "rmse": np.sqrt(mean_squared_error(y_true_agg, y_pred_agg)),
            "mae": mean_absolute_error(y_true_agg, y_pred_agg),
            "mape": mean_absolute_percentage_error(y_true_agg, y_pred_agg),
            # 'memory': None,  # Placeholder for memory, will be set only for original frequency
        }

        # Only add directional and spike metrics for sub-weekly aggregations
        if freq_td < pd.Timedelta("7D"):
            da, corr = directional_accuracy(y_true_agg, y_pred_agg)

            # Calculate spike and drop metrics with delta=0 (immediate detection)
            spike_metrics_immediate = spike_drop_match_score(
                y_true_agg.values,
                y_pred_agg.values,
                prominence=spike_prominence,
                normalize_robust=normalize_robust,
                delta=0,
            )

            # Calculate spike and drop metrics with delta=1 (1-time_unit delayed detection)
            spike_metrics_delayed = spike_drop_match_score(
                y_true_agg.values,
                y_pred_agg.values,
                prominence=spike_prominence,
                normalize_robust=normalize_robust,
                delta=1,
            )

            # Add directional and spike detection metrics
            result.update(
                {
                    "dir_accuracy": da,
                    "first_diff_corr": corr,
                    "spikes_immediate": spike_metrics_immediate["spikes"],
                    "drops_immediate": spike_metrics_immediate["drops"],
                    "spikes_delayed": spike_metrics_delayed["spikes"],
                    "drops_delayed": spike_metrics_delayed["drops"],
                }
            )
        else:
            # Add placeholder values for weekly and higher aggregations
            # This ensures the dictionary has all keys but with None values for weekly+ metrics
            result.update(
                {
                    "dir_accuracy": None,
                    "first_diff_corr": None,
                    "spikes_immediate": {"precision": None, "recall": None, "f1": None},
                    "drops_immediate": {"precision": None, "recall": None, "f1": None},
                    "spikes_delayed": {"precision": None, "recall": None, "f1": None},
                    "drops_delayed": {"precision": None, "recall": None, "f1": None},
                }
            )

        return result

    data_freq = data_freq.capitalize() if isinstance(data_freq, str) else data_freq

    # Create metrics dictionary
    metrics = {
        "device_id": device_id,
        "period": period,
    }

    # Calculate metrics for original data
    da_orig, corr_orig = directional_accuracy(y_true, y_pred)

    # Calculate spike and drop metrics for original data with delta=0 (immediate detection)
    spike_metrics_immediate = spike_drop_match_score(
        y_true.values,
        y_pred.values,
        prominence=spike_prominence,
        normalize_robust=normalize_robust,
        delta=0,
    )

    # Calculate spike and drop metrics for original data with delta=1 (1-hour delayed detection)
    spike_metrics_delayed = spike_drop_match_score(
        y_true.values,
        y_pred.values,
        prominence=spike_prominence,
        normalize_robust=normalize_robust,
        delta=1,
    )

    # Calculate memory usage if model is provided
    # memory_usage = None
    # if model is not None:
    #     memory_usage = calculate_memory_usage(model)

    # Initialize original metrics with all values
    original_metrics = {
        "rmse": np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "dir_accuracy": da_orig,
        "first_diff_corr": corr_orig,
        "spikes_immediate": spike_metrics_immediate["spikes"],
        "drops_immediate": spike_metrics_immediate["drops"],
        "spikes_delayed": spike_metrics_delayed["spikes"],
        "drops_delayed": spike_metrics_delayed["drops"],
        # 'memory': memory_usage,  # Add memory usage to metrics
    }

    # Add metrics for original data
    if data_freq not in ["1H", "H", "D", "W", "28D"]:
        metrics[data_freq] = original_metrics

    # Add hourly metrics (either from original data if data_freq is 'H' or by aggregation)
    metrics["hourly"] = original_metrics if data_freq == "H" else aggregate_metrics(y_true, y_pred, "1H")

    # Add other aggregation metrics
    metrics["daily"] = aggregate_metrics(y_true, y_pred, "1D")
    metrics["weekly"] = aggregate_metrics(y_true, y_pred, "1W")
    metrics["28_days"] = aggregate_metrics(y_true, y_pred, "28D")

    if verbose:
        print("=" * 50)
        print(f"Metrics for Device: {device_id} | Period: {period}")
        print("=" * 50)

        # Display metrics for original frequency if available
        if data_freq not in ["1H", "H", "D", "W", "28D"]:
            display_metrics(data_freq, metrics[data_freq])

        # Display metrics for each aggregation period
        for agg_period in ["hourly", "daily", "weekly", "28_days"]:
            if agg_period != data_freq:
                print("-" * 50)
                display_metrics(agg_period, metrics[agg_period])

        print("=" * 50)

    if y_true.isnull().any():
        print(["[! WARNING !] There are missing values in the y_true!"])
    if y_pred.isnull().any():
        print(["[! WARNING !] There are missing values in the y_pred!"])

    return metrics


def all_time_aggregation_metrics(
    period_metrics: Dict[str, List[Dict[str, dict]]],
    data_freq: str,
    verbose: bool = True,
) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
    """Aggregate per-period metrics across all periods for each device.

    For each device and aggregation level (original frequency plus 'hourly',
    'daily', 'weekly', '28_days') this function collates lists of metric
    values across multiple evaluation periods and summarizes them by their
    mean and standard deviation. Directional and spike/drop metrics are
    omitted (set to ``None``) for weekly and 28-day aggregations, mirroring
    the per-period metric generation logic.

    Parameters
    ----------
    period_metrics : dict
        Mapping ``device_id -> List[period_metric_dict]`` where each
        ``period_metric_dict`` is the output of :func:`calculate_metrics` for a
        single time window (i.e., contains keys for each aggregation level).
    data_freq : str
        Original data frequency label (e.g. '5min', '15min', 'H'). Used to
        decide whether to include the original frequency bucket in the
        aggregated statistics.
    verbose : bool, default=True
        If True, prints a formatted summary to stdout.

    Returns
    -------
    dict
        Nested structure:

        ``{device_id: {aggregation_level: {metric_name: {"mean": float | None,
        "std": float | None}}}}``

        Metrics include 'rmse', 'mae', 'mape', 'dir_accuracy',
        'first_diff_corr' and spike/drop precision, recall, F1 for immediate
        and delayed detection (unless weekly / 28_days where these are None).
    """
    aggregated_metrics: dict[
        str,  # device_id
        dict[
            str,  # aggregation name (5_min, hourly, etc.)
            dict[str, list[float]],  # metric dict (rmse, mae, mape, etc.)
        ],
    ] = {}
    metric_names = [
        "rmse",
        "mae",
        "mape",
        "dir_accuracy",
        "first_diff_corr",
        # 'memory'
    ]

    # Immediate spike detection metrics (delta=0)
    immediate_metrics = [
        "spikes_immediate.precision",
        "spikes_immediate.recall",
        "spikes_immediate.f1",
        "drops_immediate.precision",
        "drops_immediate.recall",
        "drops_immediate.f1",
    ]

    # 1-Hour delayed spike metrics (delta=1)
    delayed_metrics = [
        "spikes_delayed.precision",
        "spikes_delayed.recall",
        "spikes_delayed.f1",
        "drops_delayed.precision",
        "drops_delayed.recall",
        "drops_delayed.f1",
    ]

    for device_id, all_period_list in period_metrics.items():
        if device_id not in aggregated_metrics:
            aggregated_metrics[device_id] = {}
            if data_freq not in ["1H", "H", "D", "W", "28D"]:
                aggregated_metrics[device_id][data_freq] = {k: [] for k in metric_names}
                # Add all spike metrics
                for sm in immediate_metrics + delayed_metrics:
                    aggregated_metrics[device_id][data_freq][sm] = []

            for agg_level in ["hourly", "daily", "weekly", "28_days"]:
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
                aggregated_metrics[device_id][agg_name]["spikes_immediate.precision"].append(
                    period_dict[agg_name]["spikes_immediate"]["precision"]
                )
                aggregated_metrics[device_id][agg_name]["spikes_immediate.recall"].append(
                    period_dict[agg_name]["spikes_immediate"]["recall"]
                )
                aggregated_metrics[device_id][agg_name]["spikes_immediate.f1"].append(
                    period_dict[agg_name]["spikes_immediate"]["f1"]
                )

                aggregated_metrics[device_id][agg_name]["drops_immediate.precision"].append(
                    period_dict[agg_name]["drops_immediate"]["precision"]
                )
                aggregated_metrics[device_id][agg_name]["drops_immediate.recall"].append(
                    period_dict[agg_name]["drops_immediate"]["recall"]
                )
                aggregated_metrics[device_id][agg_name]["drops_immediate.f1"].append(
                    period_dict[agg_name]["drops_immediate"]["f1"]
                )

                # Handle delayed spike metrics (delta=1)
                aggregated_metrics[device_id][agg_name]["spikes_delayed.precision"].append(
                    period_dict[agg_name]["spikes_delayed"]["precision"]
                )
                aggregated_metrics[device_id][agg_name]["spikes_delayed.recall"].append(
                    period_dict[agg_name]["spikes_delayed"]["recall"]
                )
                aggregated_metrics[device_id][agg_name]["spikes_delayed.f1"].append(
                    period_dict[agg_name]["spikes_delayed"]["f1"]
                )

                aggregated_metrics[device_id][agg_name]["drops_delayed.precision"].append(
                    period_dict[agg_name]["drops_delayed"]["precision"]
                )
                aggregated_metrics[device_id][agg_name]["drops_delayed.recall"].append(
                    period_dict[agg_name]["drops_delayed"]["recall"]
                )
                aggregated_metrics[device_id][agg_name]["drops_delayed.f1"].append(
                    period_dict[agg_name]["drops_delayed"]["f1"]
                )

    # Calculate mean and std for each metric
    for device_id, agg_dict in aggregated_metrics.items():
        for agg_name, metrics in agg_dict.items():
            for metric, values in metrics.items():
                # Handle None values for weekly+ aggregations
                if agg_name in ["weekly", "28_days"] and metric not in [
                    "rmse",
                    "mae",
                    "mape",
                ]:
                    # For directional and spike/drop metrics in weekly+ aggregations, store None
                    aggregated_metrics[device_id][agg_name][metric] = {
                        "mean": None,
                        "std": None,
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
                        "mean": mean_value,
                        "std": std_value,
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
                for metric in [
                    "rmse",
                    "mae",
                    "mape",
                    #    'memory'
                ]:
                    if metrics[metric]["mean"] is not None:
                        if metric == "mape":
                            print(f"\t{metric.upper()}: {metrics[metric]['mean']:.4%} ± {metrics[metric]['std']:.4%}")
                        # elif metric == 'memory':
                        #     print(f"\tMEMORY: {metrics[metric]['mean']:.5f} MB ± {metrics[metric]['std']:.5f} MB")
                        else:
                            print(f"\t{metric.upper()}: {metrics[metric]['mean']:.4f} ± {metrics[metric]['std']:.4f}")
                    else:
                        print(f"\t{metric.upper()}: N/A")

                # Only show directional and spike metrics for sub-weekly aggregations
                if agg_name not in ["weekly", "28_days"]:
                    # Directional metrics
                    print("\n\tDirectional Metrics:")
                    print(
                        f"\t\tDirectional Accuracy: {metrics['dir_accuracy']['mean']:.4%} ± {metrics['dir_accuracy']['std']:.4%}"
                    )
                    print(
                        f"\t\tFirst-Diff Correlation: {metrics['first_diff_corr']['mean']:.4f} ± {metrics['first_diff_corr']['std']:.4f}"
                    )

                    # Immediate detection metrics
                    print("\n\tImmediate Detection (delta=0):")
                    print(
                        f"\t\tSpike Precision: {metrics['spikes_immediate.precision']['mean']:.4f} ± {metrics['spikes_immediate.precision']['std']:.4f}"
                    )
                    print(
                        f"\t\tSpike Recall: {metrics['spikes_immediate.recall']['mean']:.4f} ± {metrics['spikes_immediate.recall']['std']:.4f}"
                    )
                    print(
                        f"\t\tSpike F1: {metrics['spikes_immediate.f1']['mean']:.4f} ± {metrics['spikes_immediate.f1']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop Precision: {metrics['drops_immediate.precision']['mean']:.4f} ± {metrics['drops_immediate.precision']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop Recall: {metrics['drops_immediate.recall']['mean']:.4f} ± {metrics['drops_immediate.recall']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop F1: {metrics['drops_immediate.f1']['mean']:.4f} ± {metrics['drops_immediate.f1']['std']:.4f}"
                    )

                    # Delayed detection metrics
                    print("\n\t1-Hour Delayed Detection (delta=1):")
                    print(
                        f"\t\tSpike Precision: {metrics['spikes_delayed.precision']['mean']:.4f} ± {metrics['spikes_delayed.precision']['std']:.4f}"
                    )
                    print(
                        f"\t\tSpike Recall: {metrics['spikes_delayed.recall']['mean']:.4f} ± {metrics['spikes_delayed.recall']['std']:.4f}"
                    )
                    print(
                        f"\t\tSpike F1: {metrics['spikes_delayed.f1']['mean']:.4f} ± {metrics['spikes_delayed.f1']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop Precision: {metrics['drops_delayed.precision']['mean']:.4f} ± {metrics['drops_delayed.precision']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop Recall: {metrics['drops_delayed.recall']['mean']:.4f} ± {metrics['drops_delayed.recall']['std']:.4f}"
                    )
                    print(
                        f"\t\tDrop F1: {metrics['drops_delayed.f1']['mean']:.4f} ± {metrics['drops_delayed.f1']['std']:.4f}"
                    )

            print("=" * 50)

    return aggregated_metrics
