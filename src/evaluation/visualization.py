"""Visualization helpers for model evaluation.

This module provides convenience plotting functions used during exploratory
analysis and model evaluation:

* ``plot_date_range`` - focus on a specific calendar interval (optionally
    aggregated) with outlier and spike/drop highlighting.
* ``plot_part_predictions`` - tri-panel (beginning/middle/end) snapshot to
    quickly inspect temporal drift or regime changes.
* ``plot_hourly_predictions`` - hourly aggregation view with accompanying
    printed metrics (directional + spike/drop detection summaries).

Spike/Drop Detection
--------------------
All spike / drop annotations reuse the detection logic from
``detect_spikes_and_drops``. When ``normalize_robust`` is True the underlying
series is robustly normalized (median & MAD); otherwise standard z-score.

Notes
-----
The functions avoid returning objects and instead immediately render figures to
matplotlib's current backend for interactive workflows (e.g. Jupyter). They
raise a ``ValueError`` if the requested date range is empty.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Union, Tuple

from src.utils.utils import aggregate_values
from src.evaluation.metrics import calculate_metrics, detect_spikes_and_drops


def plot_date_range(
    y_true: pd.Series,
    y_pred: pd.Series,
    start_date: Union[str, pd.Timestamp],
    end_date: Union[str, pd.Timestamp],
    device_id: Optional[str] = None,
    outlier_threshold: float = 0.9,
    figure_size: Tuple[int, int] = (14, 8),
    aggregation: Optional[str] = None,
    show_spikes: bool = True,
    spike_prominence: float = 10.0,
    normalize_robust: bool = True,
) -> None:
    """Plot predictions vs. actual values for a date range.

    Optionally aggregates the series (e.g. '15min', '1H', 'D'), highlights large
    absolute percentage deviations ("outliers"), and annotates detected spikes /
    drops.

    Parameters
    ----------
    y_true : pandas.Series
        Ground truth target values indexed by timestamp.
    y_pred : pandas.Series
        Predicted values aligned to ``y_true`` index.
    start_date : str or datetime-like
        Start of plotting window (inclusive); string inputs are parsed via
        ``pandas.to_datetime``.
    end_date : str or datetime-like
        End of plotting window (inclusive); string inputs are parsed via
        ``pandas.to_datetime``.
    device_id : str, optional
        Device identifier displayed in the title.
    outlier_threshold : float, default 0.9
        Relative absolute error (|y - y_hat| / (y + 1e-8)) above which points are
        flagged as outliers.
    figure_size : tuple(int, int), default (14, 8)
        Matplotlib figure size in inches.
    aggregation : str or None, optional
        Resampling rule (Pandas offset alias). If provided both series are
        aggregated via ``aggregate_values``.
    show_spikes : bool, default True
        Whether to annotate detected spikes and drops.
    spike_prominence : float, default 10.0
        Prominence threshold passed to ``detect_spikes_and_drops`` after
        normalization.
    normalize_robust : bool, default True
        If True use robust normalization (median & MAD); else mean/std z-score.

    Raises
    ------
    ValueError
        If no data lies within the requested date range.
    """
    # Convert string dates to datetime if needed
    if isinstance(start_date, str):
        start_date = pd.to_datetime(start_date)
    if isinstance(end_date, str):
        end_date = pd.to_datetime(end_date)

    # Filter data based on date range
    mask = (y_true.index >= start_date) & (y_true.index <= end_date)
    y_true_filtered = y_true[mask]
    y_pred_filtered = y_pred[mask]

    if len(y_true_filtered) == 0:
        raise ValueError("No data found in the specified date range")

    # Parse aggregation string to a standard format if needed
    metrics_key = "5_min"  # Default metrics key
    agg_label = "Raw"  # Default label for plot

    # Apply aggregation if specified
    if aggregation:
        # Normalize aggregation string to pandas format
        agg_norm = aggregation.lower()

        # Convert common variations to pandas format
        if agg_norm.endswith("m") and agg_norm[:-1].isdigit():
            agg_norm = f"{agg_norm[:-1]}min"
        elif agg_norm.endswith("minutes"):
            agg_norm = f"{agg_norm.replace('minutes', '')}min"
        elif agg_norm.endswith("h") and agg_norm[:-1].isdigit():
            agg_norm = f"{agg_norm[:-1]}H"
        elif agg_norm.endswith("hours"):
            agg_norm = f"{agg_norm.replace('hours', '')}H"

        # Determine display label and metrics key
        if "min" in agg_norm:
            agg_label = f"{agg_norm.replace('min', ' Minute')}"
            metrics_key = "5_min" if agg_norm == "5min" else "5_min"  # Fallback to 5min metrics
        elif "h" in agg_norm.lower():
            agg_label = f"{agg_norm.replace('H', ' Hour').replace('h', ' Hour')}"
            metrics_key = "hourly"
        elif agg_norm == "d":
            agg_label = "Daily"
            metrics_key = "daily"
        elif agg_norm == "w":
            agg_label = "Weekly"
        elif agg_norm == "m":
            agg_label = "Monthly"
        else:
            agg_label = aggregation

        # Apply aggregation
        y_true_filtered, y_pred_filtered = aggregate_values(y_true_filtered, y_pred_filtered, agg_norm)

    # Calculate percent difference to identify outliers
    pct_diff = abs(y_true_filtered - y_pred_filtered) / (y_true_filtered + 1e-8)
    outliers = pct_diff > outlier_threshold

    # Get period information for title
    period = f"{start_date.strftime('%Y-%m-%d %H:%M:%S')} - {end_date.strftime('%Y-%m-%d %H:%M:%S')}"

    # Calculate metrics for legend (metrics are for the filtered period only)
    metrics = calculate_metrics(
        y_true_filtered,
        y_pred_filtered,
        device_id or "Unknown",
        period,
        verbose=False,
        spike_prominence=spike_prominence,
        normalize_robust=normalize_robust,
    )

    # Create the plot
    plt.figure(figsize=figure_size)

    # Plot regular points
    plt.plot(
        y_true_filtered.index,
        y_true_filtered.values,
        label="Original Values",
        marker="o",
    )
    plt.plot(
        y_pred_filtered.index,
        y_pred_filtered.values,
        label="Predicted Values",
        marker="x",
    )

    # Highlight outliers
    if outliers.any():
        plt.plot(
            y_pred_filtered[outliers].index,
            y_pred_filtered[outliers].values,
            "ro",
            markersize=10,
            markerfacecolor="none",
            label=f"Outliers (>{outlier_threshold * 100}% diff)",
        )

    # Add spike and drop annotations if requested
    if show_spikes:
        # Detect spikes and drops
        true_peaks, true_drops = detect_spikes_and_drops(
            y_true_filtered.values,
            prominence=spike_prominence,
            normalize_robust=normalize_robust,
        )

        # Calculate offset for better visibility (5% of the data range)
        y_range = y_true_filtered.max() - y_true_filtered.min()
        offset = y_range * 0.05

        # Plot true spikes with red triangle markers (positioned higher)
        if len(true_peaks) > 0:
            plt.plot(
                y_true_filtered.index[true_peaks],
                y_true_filtered.values[true_peaks] + offset,
                "^r",
                markersize=6,
                label="Spikes",
            )

        # Plot true drops with blue triangle markers (positioned lower)
        if len(true_drops) > 0:
            plt.plot(
                y_true_filtered.index[true_drops],
                y_true_filtered.values[true_drops] - offset,
                "vb",
                markersize=6,
                label="Drops",
            )

    # Add metrics to legend (metrics for filtered period only)
    if metrics_key in metrics:
        metrics_text = (
            f"Metrics for selected period:\nRMSE: {metrics[metrics_key]['rmse']:.2f}, "
            f"MAE: {metrics[metrics_key]['mae']:.2f}, "
            f"MAPE: {metrics[metrics_key]['mape']:.2%}"
        )
    else:
        metrics_text = (
            f"Metrics for selected period:\nRMSE: {metrics['5_min']['rmse']:.2f}, "
            f"MAE: {metrics['5_min']['mae']:.2f}, "
            f"MAPE: {metrics['5_min']['mape']:.2%}"
        )
    plt.plot([], [], " ", label=metrics_text)

    # Set title and labels
    title = f"{agg_label} Original vs Predicted Values for Selected Date Range"
    if device_id:
        title = f"{title}\nDevice: {device_id}"
    title = f"{title}\nPeriod: {period}"
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Total Power Active")

    # Format x-axis for better readability
    plt.gcf().autofmt_xdate()

    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_part_predictions(
    y_true: Union[pd.Series, np.ndarray],
    y_pred: Union[pd.Series, np.ndarray],
    data_show: int = 100,
    outlier_threshold: float = 0.9,
    device_id: Optional[str] = None,
    show_spikes: bool = True,
    spike_prominence: float = 10.0,
    normalize_robust: bool = True,
) -> None:
    """
    Plot the original vs predicted values for the beginning, middle, and end of the dataset.

    Parameters
    ----------
    y_true : numpy.ndarray or pandas.Series
        Original target values.
    y_pred : numpy.ndarray or pandas.Series
        Predicted target values.
    data_show : int, optional
        Number of data points to show in each plot (default is 100).
    outlier_threshold : float, optional
        Threshold to mark outliers (default is 0.9, i.e., 90% difference).
    device_id : str, optional
        Device ID to display in the title.
    show_spikes : bool, optional
        Whether to highlight spikes and drops in the plot.
    spike_prominence : float, optional
        Required prominence for spike detection.
    normalize_robust : bool, optional
        If True, use robust (median & MAD) normalization for spike detection.
    """
    # Get period information for title
    start_time = (
        y_true.index.min().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(y_true.index.min(), "strftime")
        else str(y_true.index.min())
    )
    end_time = (
        y_true.index.max().strftime("%Y-%m-%d %H:%M:%S")
        if hasattr(y_true.index.max(), "strftime")
        else str(y_true.index.max())
    )
    period = f"{start_time} - {end_time}"

    # Calculate metrics for legend
    metrics = calculate_metrics(
        y_true,
        y_pred,
        device_id or "Unknown",
        period,
        verbose=False,
        spike_prominence=spike_prominence,
    )

    # Calculate percent difference to identify outliers
    pct_diff = abs(y_true - y_pred) / (y_true + 1e-8)  # Add small epsilon to avoid division by zero
    outliers = pct_diff > outlier_threshold

    # Extract timestamps and values for the beginning, middle, and end
    total_length = len(y_true)
    data_show = min(data_show, total_length // 3)  # Ensure we don't try to show more than a third of the data

    # Get slices for beginning, middle, and end
    beginning_slice = slice(0, data_show)
    middle_start = (total_length - data_show) // 2
    middle_slice = slice(middle_start, middle_start + data_show)
    end_slice = slice(total_length - data_show, total_length)

    # Create series for these sections
    y_true_beginning = y_true.iloc[beginning_slice]
    y_pred_beginning = y_pred.iloc[beginning_slice]
    outliers_beginning = outliers.iloc[beginning_slice]

    y_true_middle = y_true.iloc[middle_slice]
    y_pred_middle = y_pred.iloc[middle_slice]
    outliers_middle = outliers.iloc[middle_slice]

    y_true_end = y_true.iloc[end_slice]
    y_pred_end = y_pred.iloc[end_slice]
    outliers_end = outliers.iloc[end_slice]

    # Detect spikes and drops for each section if requested
    if show_spikes:
        beginning_true_peaks, beginning_true_drops = detect_spikes_and_drops(
            y_true_beginning.values,
            prominence=spike_prominence,
            normalize_robust=normalize_robust,
        )
        middle_true_peaks, middle_true_drops = detect_spikes_and_drops(
            y_true_middle.values,
            prominence=spike_prominence,
            normalize_robust=normalize_robust,
        )
        end_true_peaks, end_true_drops = detect_spikes_and_drops(
            y_true_end.values,
            prominence=spike_prominence,
            normalize_robust=normalize_robust,
        )

    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))

    # Add metrics to graph titles
    metrics_text = (
        f"RMSE: {metrics['5_min']['rmse']:.2f}, "
        f"\nMAE: {metrics['5_min']['mae']:.2f}, "
        f"\nMAPE: {metrics['5_min']['mape']:.2%}"
    )

    # Title prefix
    title_prefix = "Original vs Predicted Values"
    if device_id:
        title_prefix = f"{title_prefix} - Device: {device_id}"

    # Beginning plot
    axes[0].plot(
        y_true_beginning.index,
        y_true_beginning.values,
        label="Original Values",
        marker="o",
    )
    axes[0].plot(
        y_pred_beginning.index,
        y_pred_beginning.values,
        label="Predicted Values",
        marker="x",
    )
    if outliers_beginning.any():
        axes[0].plot(
            y_pred_beginning[outliers_beginning].index,
            y_pred_beginning[outliers_beginning].values,
            "ro",
            markersize=10,
            markerfacecolor="none",
            label=f"Outliers (>{outlier_threshold * 100}% diff)",
        )

    # Add spike and drop annotations for beginning section
    if show_spikes:
        # Calculate offset for better visibility (5% of the data range)
        y_range_begin = y_true_beginning.max() - y_true_beginning.min()
        offset_begin = y_range_begin * 0.05

        y_range_middle = y_true_middle.max() - y_true_middle.min()
        offset_middle = y_range_middle * 0.05

        y_range_end = y_true_end.max() - y_true_end.min()
        offset_end = y_range_end * 0.05

        # Plot spikes and drops for beginning section
        if len(beginning_true_peaks) > 0:
            axes[0].plot(
                y_true_beginning.index[beginning_true_peaks],
                y_true_beginning.values[beginning_true_peaks] + offset_begin,
                "^r",
                markersize=6,
                label="Spikes",
            )

        if len(beginning_true_drops) > 0:
            axes[0].plot(
                y_true_beginning.index[beginning_true_drops],
                y_true_beginning.values[beginning_true_drops] - offset_begin,
                "vb",
                markersize=6,
                label="Drops",
            )

    axes[0].plot([], [], " ", label=metrics_text)  # Add metrics to legend
    if show_spikes:
        axes[0].plot([], [], " ", marker="^", markersize=10, color="red", label="Spikes")
        axes[0].plot([], [], " ", marker="v", markersize=10, color="blue", label="Drops")

    axes[0].set_title(f"{title_prefix}\nBeginning: {y_true_beginning.index.min()} - {y_true_beginning.index.max()}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Total Power Active")
    axes[0].legend()
    axes[0].grid(True)

    # Middle plot
    axes[1].plot(y_true_middle.index, y_true_middle.values, label="Original Values", marker="o")
    axes[1].plot(y_pred_middle.index, y_pred_middle.values, label="Predicted Values", marker="x")
    if outliers_middle.any():
        axes[1].plot(
            y_pred_middle[outliers_middle].index,
            y_pred_middle[outliers_middle].values,
            "ro",
            markersize=10,
            markerfacecolor="none",
            label=f"Outliers (>{outlier_threshold * 100}% diff)",
        )

    # Add spike and drop annotations for middle section
    if show_spikes:
        # Plot spikes and drops for middle section
        if len(middle_true_peaks) > 0:
            axes[1].plot(
                y_true_middle.index[middle_true_peaks],
                y_true_middle.values[middle_true_peaks] + offset_middle,
                "^r",
                markersize=6,
                label="Spikes",
            )

        if len(middle_true_drops) > 0:
            axes[1].plot(
                y_true_middle.index[middle_true_drops],
                y_true_middle.values[middle_true_drops] - offset_middle,
                "vb",
                markersize=6,
                label="Drops",
            )

    axes[1].plot([], [], " ", label=metrics_text)  # Add metrics to legend
    if show_spikes:
        axes[1].plot([], [], " ", marker="^", markersize=10, color="red", label="Spikes")
        axes[1].plot([], [], " ", marker="v", markersize=10, color="blue", label="Drops")

    axes[1].set_title(f"{title_prefix}\nMiddle: {y_true_middle.index.min()} - {y_true_middle.index.max()}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Total Power Active")
    axes[1].legend()
    axes[1].grid(True)

    # End plot
    axes[2].plot(y_true_end.index, y_true_end.values, label="Original Values", marker="o")
    axes[2].plot(y_pred_end.index, y_pred_end.values, label="Predicted Values", marker="x")
    if outliers_end.any():
        axes[2].plot(
            y_pred_end[outliers_end].index,
            y_pred_end[outliers_end].values,
            "ro",
            markersize=10,
            markerfacecolor="none",
            label=f"Outliers (>{outlier_threshold * 100}% diff)",
        )

    # Add spike and drop annotations for end section
    if show_spikes:
        # Plot spikes and drops for end section
        if len(end_true_peaks) > 0:
            axes[2].plot(
                y_true_end.index[end_true_peaks],
                y_true_end.values[end_true_peaks] + offset_end,
                "^r",
                markersize=6,
                label="Spikes",
            )

        if len(end_true_drops) > 0:
            axes[2].plot(
                y_true_end.index[end_true_drops],
                y_true_end.values[end_true_drops] - offset_end,
                "vb",
                markersize=6,
                label="Drops",
            )

    axes[2].plot([], [], " ", label=metrics_text)  # Add metrics to legend
    if show_spikes:
        axes[2].plot([], [], " ", marker="^", markersize=10, color="red", label="Spikes")
        axes[2].plot([], [], " ", marker="v", markersize=10, color="blue", label="Drops")

    axes[2].set_title(f"{title_prefix}\nEnd: {y_true_end.index.min()} - {y_true_end.index.max()}")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Total Power Active")
    axes[2].legend()
    axes[2].grid(True)

    # Format x-axis for better readability
    for ax in axes:
        plt.sca(ax)
        plt.gcf().autofmt_xdate()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def plot_hourly_predictions(
    y_true: pd.Series,
    y_pred: pd.Series,
    data_freq: str,
    device_id: Optional[str] = None,
    outlier_threshold: float = 0.9,
    data_show: Optional[int] = None,
    show_spikes: bool = True,
    spike_prominence: float = 10.0,
    normalize_robust: bool = True,
) -> None:
    """Plot hourly aggregated predictions vs. actuals with metrics.

    Aggregates to hourly resolution (using mean) and highlights large relative
    errors plus optional spike / drop annotations. After rendering the figure a
    textual summary of metrics is printed.

    Parameters
    ----------
    y_true : pandas.Series
        Ground truth target values indexed by timestamp (possibly higher
        frequency than hourly).
    y_pred : pandas.Series
        Predicted values aligned to ``y_true`` index.
    data_freq : str
        Original sampling frequency (Pandas offset alias) used when computing
        metrics.
    device_id : str, optional
        Device identifier displayed in the title.
    outlier_threshold : float, default 0.9
        Relative absolute error above which points are flagged as outliers.
    data_show : int or None, optional
        If provided, only the most recent ``data_show`` hourly points are kept.
    show_spikes : bool, default True
        Whether to annotate detected spikes and drops.
    spike_prominence : float, default 10.0
        Prominence threshold for spike/drop detection after normalization.
    normalize_robust : bool, default True
        If True use robust normalization (median & MAD); else mean/std z-score.
    """
    # Aggregate hourly
    y_true_hourly, y_pred_hourly = aggregate_values(
        y_true, y_pred, "H", "mean"
    )  # We're working with electricity data, so if we get lower granularities (e.g., 15-min) in kW,
    # we can use 'mean' aggregation to get the consumption in kWh.

    # Calculate percent difference to identify outliers
    pct_diff = abs(y_true_hourly - y_pred_hourly) / (
        y_true_hourly + 1e-8
    )  # Add small epsilon to avoid division by zero
    outliers = pct_diff > outlier_threshold

    # Limit data points if specified
    if data_show is not None:
        if data_show < len(y_true_hourly):
            y_true_hourly = y_true_hourly[-data_show:]
            y_pred_hourly = y_pred_hourly[-data_show:]
            outliers = outliers[-data_show:]

    # Get period information
    start_time = y_true_hourly.index.min().strftime("%Y-%m-%d %H:%M:%S")
    end_time = y_true_hourly.index.max().strftime("%Y-%m-%d %H:%M:%S")
    period = f"{start_time} - {end_time}"

    # Calculate metrics for display after the plot
    metrics = calculate_metrics(
        y_true,
        y_pred,
        device_id or "Unknown",
        period,
        data_freq,
        verbose=False,
        spike_prominence=spike_prominence,
        normalize_robust=normalize_robust,
    )
    hourly_metrics = metrics["hourly"]

    # Create the plot
    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot regular points
    ax.plot(y_true_hourly.index, y_true_hourly.values, label="Original Values", marker="o")
    ax.plot(y_pred_hourly.index, y_pred_hourly.values, label="Predicted Values", marker="x")

    # Highlight outliers
    if outliers.any():
        ax.plot(
            y_pred_hourly[outliers].index,
            y_pred_hourly[outliers],
            "ro",
            markersize=10,
            markerfacecolor="none",
            label=f"Outliers (>{outlier_threshold * 100}% diff)",
        )

    # Add spike and drop marks if requested
    if show_spikes:
        # Detect spikes and drops
        true_peaks, true_drops = detect_spikes_and_drops(
            y_true_hourly.values,
            prominence=spike_prominence,
            normalize_robust=normalize_robust,
        )

        # Calculate offset for better visibility (5% of the data range)
        y_range = y_true_hourly.max() - y_true_hourly.min()
        offset = y_range * 0.05

        # Plot spikes with red triangle markers (positioned higher)
        if len(true_peaks) > 0:
            # Add offset to y-values for better visibility
            peak_y_values = y_true_hourly.values[true_peaks] + offset
            ax.plot(
                y_true_hourly.index[true_peaks],
                peak_y_values,
                "^r",
                markersize=6,
                label="Spikes",
            )

        # Plot drops with blue triangle markers (positioned lower)
        if len(true_drops) > 0:
            # Subtract offset from y-values for better visibility
            drop_y_values = y_true_hourly.values[true_drops] - offset
            ax.plot(
                y_true_hourly.index[true_drops],
                drop_y_values,
                "vb",
                markersize=6,
                label="Drops",
            )

    # Set title and labels
    title = "Hourly Aggregated Values"
    if device_id:
        title = f"{title}\nDevice: {device_id}"
    title = f"{title}\nPeriod: {period}"
    ax.set_title(title)
    ax.set_xlabel("Time")
    ax.set_ylabel("Total Power Active")

    # Format x-axis for better readability
    fig.autofmt_xdate()

    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()

    # Print metrics after showing the plot
    print("\n" + "=" * 50)
    print("Performance Metrics for Hourly Aggregated Data")
    print("=" * 50)

    # Basic prediction metrics
    print("Basic Prediction Metrics:")
    print(f"RMSE: {hourly_metrics['rmse']:.4f}")
    print(f"MAE: {hourly_metrics['mae']:.4f}")
    print(f"MAPE: {hourly_metrics['mape']:.4%}")

    # Directional metrics
    print("\nDirectional Metrics:")
    print(f"Directional Accuracy: {hourly_metrics['dir_accuracy']:.4%}")
    print(f"First-Diff Correlation: {hourly_metrics['first_diff_corr']:.4f}")

    if show_spikes:
        # Immediate detection metrics
        print("\nImmediate Detection (delta=0):")
        print("Spikes:")
        print(f"  Precision: {hourly_metrics['spikes_immediate']['precision']:.4f}")
        print(f"  Recall: {hourly_metrics['spikes_immediate']['recall']:.4f}")
        print(f"  F1 Score: {hourly_metrics['spikes_immediate']['f1']:.4f}")
        print("Drops:")
        print(f"  Precision: {hourly_metrics['drops_immediate']['precision']:.4f}")
        print(f"  Recall: {hourly_metrics['drops_immediate']['recall']:.4f}")
        print(f"  F1 Score: {hourly_metrics['drops_immediate']['f1']:.4f}")

        # Delayed detection metrics
        print("\n1-Hour Delayed Detection (delta=1):")
        print("Spikes:")
        print(f"  Precision: {hourly_metrics['spikes_delayed']['precision']:.4f}")
        print(f"  Recall: {hourly_metrics['spikes_delayed']['recall']:.4f}")
        print(f"  F1 Score: {hourly_metrics['spikes_delayed']['f1']:.4f}")
        print("Drops:")
        print(f"  Precision: {hourly_metrics['drops_delayed']['precision']:.4f}")
        print(f"  Recall: {hourly_metrics['drops_delayed']['recall']:.4f}")
        print(f"  F1 Score: {hourly_metrics['drops_delayed']['f1']:.4f}")

    print("=" * 50)
