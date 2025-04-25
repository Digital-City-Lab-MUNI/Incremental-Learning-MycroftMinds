import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional

# Import functions needed for plot calculations
from src.utils import aggregate_values
from src.metrics import calculate_metrics

def plot_predictions(X, y_true, y_pred, data_show=100, outlier_threshold=0.9, device_id=None) -> None:
    """
    Plots the original vs predicted values for the beginning, middle, and end of the dataset.

    Args:
        X (numpy.ndarray): input data containing timestamps and other features.
        y_true (numpy.ndarray): or pandas.Series, original target values.
        y_pred (numpy.ndarray): or pandas.Series, predicted target values.
        data_show (int): number of data points to show in each plot (default is 100).
        outlier_threshold (float): threshold to mark outliers (default 0.9 = 90% difference).
        device_id (str): optional device ID to display in the title.
    """
    # Ensure y_true and y_pred are pandas Series for easier processing
    if not isinstance(y_true, pd.Series):
        if isinstance(X[:, 0][0], (str, pd.Timestamp)):
            y_true = pd.Series(y_true, index=pd.to_datetime(X[:, 0]))
        else:
            # Assume timestamps are already datetime objects
            y_true = pd.Series(y_true, index=X[:, 0])
    
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)
    
    # Get period information for title
    start_time = y_true.index.min().strftime('%Y-%m-%d %H:%M:%S') if hasattr(y_true.index.min(), 'strftime') else str(y_true.index.min())
    end_time = y_true.index.max().strftime('%Y-%m-%d %H:%M:%S') if hasattr(y_true.index.max(), 'strftime') else str(y_true.index.max())
    period = f"{start_time} - {end_time}"
    
    # Calculate metrics for legend
    metrics = calculate_metrics(y_true, y_pred, device_id or "Unknown", period, verbose=False)
    
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
    
    # Plot the results
    fig, axes = plt.subplots(3, 1, figsize=(14, 18))
    
    # Add metrics to graph titles
    metrics_text = (f'RMSE: {metrics["5_min"]["rmse"]:.2f}, '
                    f'\nMAE: {metrics["5_min"]["mae"]:.2f}, '
                    f'\nMAPE: {metrics["5_min"]["mape"]:.2%}')
    
    # Title prefix
    title_prefix = "Original vs Predicted Values"
    if device_id:
        title_prefix = f"{title_prefix} - Device: {device_id}"
    
    # Beginning plot
    axes[0].plot(y_true_beginning.index, y_true_beginning.values, label='Original Values', marker='o')
    axes[0].plot(y_pred_beginning.index, y_pred_beginning.values, label='Predicted Values', marker='x')
    if outliers_beginning.any():
        axes[0].plot(y_pred_beginning[outliers_beginning].index, 
                    y_pred_beginning[outliers_beginning].values,
                    'ro', markersize=10, markerfacecolor='none', 
                    label=f'Outliers (>{outlier_threshold*100}% diff)')
    axes[0].plot([], [], ' ', label=metrics_text)  # Add metrics to legend
    axes[0].set_title(f'{title_prefix}\nBeginning: {y_true_beginning.index.min()} - {y_true_beginning.index.max()}')
    axes[0].set_xlabel('Time')
    axes[0].set_ylabel('Total Power Active')
    axes[0].legend()
    axes[0].grid(True)

    # Middle plot
    axes[1].plot(y_true_middle.index, y_true_middle.values, label='Original Values', marker='o')
    axes[1].plot(y_pred_middle.index, y_pred_middle.values, label='Predicted Values', marker='x')
    if outliers_middle.any():
        axes[1].plot(y_pred_middle[outliers_middle].index, 
                    y_pred_middle[outliers_middle].values,
                    'ro', markersize=10, markerfacecolor='none', 
                    label=f'Outliers (>{outlier_threshold*100}% diff)')
    axes[1].plot([], [], ' ', label=metrics_text)  # Add metrics to legend
    axes[1].set_title(f'{title_prefix}\nMiddle: {y_true_middle.index.min()} - {y_true_middle.index.max()}')
    axes[1].set_xlabel('Time')
    axes[1].set_ylabel('Total Power Active')
    axes[1].legend()
    axes[1].grid(True)

    # End plot
    axes[2].plot(y_true_end.index, y_true_end.values, label='Original Values', marker='o')
    axes[2].plot(y_pred_end.index, y_pred_end.values, label='Predicted Values', marker='x')
    if outliers_end.any():
        axes[2].plot(y_pred_end[outliers_end].index, 
                    y_pred_end[outliers_end].values,
                    'ro', markersize=10, markerfacecolor='none', 
                    label=f'Outliers (>{outlier_threshold*100}% diff)')
    axes[2].plot([], [], ' ', label=metrics_text)  # Add metrics to legend
    axes[2].set_title(f'{title_prefix}\nEnd: {y_true_end.index.min()} - {y_true_end.index.max()}')
    axes[2].set_xlabel('Time')
    axes[2].set_ylabel('Total Power Active')
    axes[2].legend()
    axes[2].grid(True)

    # Format x-axis for better readability
    for ax in axes:
        plt.sca(ax)
        plt.gcf().autofmt_xdate()

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()

def plot_hourly_predictions(X, y_true, y_pred, device_id=None, outlier_threshold=0.9, data_show=None) -> None:
    """
    Plots hourly aggregated predictions against actual values with outliers highlighted.
    
    Args:
        X (numpy.ndarray): input data containing timestamps and other features.
        y_true (pandas.Series):or array-like, original target values with timestamps as index.
        y_pred (pandas.Series):or array-like, predicted target values with timestamps as index.
        device_id (str): the device ID for display in the title.
        outlier_threshold: (float): threshold to mark outliers (default 0.9 = 90% difference).
        data_show (int): number of data points to show (default is all).
    """
    # Ensure data has datetime index
    if not isinstance(y_true, pd.Series):
        if isinstance(X[:, 0][0], (str, pd.Timestamp)):
            y_true = pd.Series(y_true, index=pd.to_datetime(X[:, 0]))
        else:
            # Assume timestamps are already datetime objects
            y_true = pd.Series(y_true, index=X[:, 0])
    
    if not isinstance(y_pred, pd.Series):
        y_pred = pd.Series(y_pred, index=y_true.index)
    
    # Aggregate hourly
    y_true_hourly, y_pred_hourly = aggregate_values(y_true, y_pred, 'H')
    
    # Calculate percent difference to identify outliers
    pct_diff = abs(y_true_hourly - y_pred_hourly) / (y_true_hourly + 1e-8)  # Add small epsilon to avoid division by zero
    outliers = pct_diff > outlier_threshold
    
    # Limit data points if specified
    if data_show is not None:
        if data_show < len(y_true_hourly):
            y_true_hourly = y_true_hourly[-data_show:]
            y_pred_hourly = y_pred_hourly[-data_show:]
            outliers = outliers[-data_show:]
    
    # Get period information
    start_time = y_true_hourly.index.min().strftime('%Y-%m-%d %H:%M:%S')
    end_time = y_true_hourly.index.max().strftime('%Y-%m-%d %H:%M:%S')
    period = f"{start_time} - {end_time}"
    
    # Calculate metrics for legend
    metrics = calculate_metrics(y_true, y_pred, device_id or "Unknown", period, verbose=False)
    hourly_metrics = metrics['hourly']
    
    # Create the plot
    plt.figure(figsize=(14, 8))
    
    # Plot regular points
    plt.plot(y_true_hourly.index, y_true_hourly.values, label='Original Values', marker='o')
    plt.plot(y_pred_hourly.index, y_pred_hourly.values, label='Predicted Values', marker='x')
    
    # Highlight outliers
    if outliers.any():
        plt.plot(y_pred_hourly[outliers].index, y_pred_hourly[outliers], 
                 'ro', markersize=10, markerfacecolor='none', label=f'Outliers (>{outlier_threshold*100}% diff)')
    
    # Add metrics to legend
    metrics_text = (f'RMSE: {hourly_metrics["rmse"]:.2f}, '
                    f'MAE: {hourly_metrics["mae"]:.2f}, '
                    f'MAPE: {hourly_metrics["mape"]:.2%}')
    plt.plot([], [], ' ', label=metrics_text)
    
    # Set title and labels
    title = f"Hourly Aggregated Values"
    if device_id:
        title = f"{title}\nDevice: {device_id}"
    title = f"{title}\nPeriod: {period}"
    plt.title(title)
    plt.xlabel('Time')
    plt.ylabel('Total Power Active')
    
    # Format x-axis for better readability
    plt.gcf().autofmt_xdate()
    
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
