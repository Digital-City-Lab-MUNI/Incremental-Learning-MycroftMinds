"""Logging helpers for consistent project-wide logs and checkpoints.

Provides convenience utilities to create log directories, write log lines,
and save/load checkpointed tuning results and metrics.
"""

import os
import glob
import pickle
from datetime import datetime
from typing import Dict, Any, List, Optional


def setup_log_directory(log_dir: Optional[str] = None, log_prefix: Optional[str] = None) -> str:
    """
    Create log directory if it doesn't exist and return the path.

    Parameters
    ----------
    log_dir : str, optional
        Directory to save tuning results logs. If None, uses default.
    log_prefix : str, optional
        Prefix for the log directory name. If None, uses default.

    Returns
    -------
    str
        The log directory path.
    """
    if log_dir is None:
        if log_prefix is None:
            log_prefix = "tuning"
        log_dir = f"../../results/{log_prefix}"

    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_log_file(log_dir: str, model_name: str, file_prefix: Optional[str] = None) -> str:
    """
    Create a timestamped log file for a specific model with optional prefix.

    Parameters
    ----------
    log_dir : str
        Directory to save the log file.
    model_name : str
        Name of the model being tuned.
    file_prefix : str, optional
        Optional prefix for the filename (e.g., 'dev_4_wfv').

    Returns
    -------
    str
        Path to the log file.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if file_prefix:
        filename = f"{timestamp}_{file_prefix}_{model_name}_tuning.txt"
    else:
        filename = f"{timestamp}_{model_name}_tuning.txt"

    log_file = os.path.join(log_dir, filename)
    return log_file


def log_and_print(log_file: str, message: str, print_console: bool = False) -> None:
    """
    Log message to file and optionally print to console.

    Parameters
    ----------
    log_file : str
        Path to the log file.
    message : str
        Message to log.
    print_console : bool, optional
        Whether to print to console.
    """
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(message + "\n")
    # if print_console:
    #     print(message)


def save_checkpoint_results(
    results_dir: str,
    model_name: str,
    best_params: Dict[str, Any],
    best_scores: Dict[str, Any],
    file_prefix: Optional[str] = None,
) -> None:
    """
    Save checkpoint results for a model with optional prefix.

    Parameters
    ----------
    results_dir : str
        Directory to save results.
    model_name : str
        Name of the model.
    best_params : Dict[str, Any]
        Best parameters found (can be a nested dictionary for per-device results).
    best_scores : Dict[str, Any]
        Best scores achieved (can be a nested dictionary for per-device results).
    file_prefix : str, optional
        Optional prefix for the filename (e.g., 'dev_4_wfv').
    """
    os.makedirs(results_dir, exist_ok=True)

    if file_prefix:
        filename = f"{file_prefix}_{model_name}_results.pkl"
    else:
        filename = f"{model_name}_results.pkl"

    checkpoint_file = os.path.join(results_dir, filename)

    # Check if we're saving per-device results (has a different structure)
    if "per_device_best_params_lower_better" in best_params or "per_device_best_scores_lower_better" in best_scores:
        # This is a per-device results format
        with open(checkpoint_file, "wb") as f:
            pickle.dump(best_params, f)  # We're passing the whole dictionary structure
    else:
        # Standard format
        with open(checkpoint_file, "wb") as f:
            pickle.dump({"best_params": best_params, "best_scores": best_scores}, f)

    print(f"Results saved to {checkpoint_file}")


def load_checkpoint_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all available checkpoint results.

    Parameters
    ----------
    results_dir : str
        Directory containing result files.

    Returns
    -------
    Dict[str, Any]
        Dictionary containing loaded results.
    """
    results = {}

    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return results

    result_files = glob.glob(f"{results_dir}/*_results.pkl")
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace("_results.pkl", "")
        try:
            with open(file_path, "rb") as f:
                results[model_name] = pickle.load(f)
            print(f"Loaded {model_name} results")

        except FileNotFoundError:
            print(f"Results file {file_path} does not exist.")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")

    return results


def list_log_files(results_dir: str) -> List[str]:
    """List all log files in the results directory."""
    try:
        log_files = glob.glob(f"{results_dir}/*.txt")
        log_files.sort()

        print(f"Available log files in {results_dir}:")
        for log_file in log_files:
            file_size = os.path.getsize(log_file)
            print(f"  {os.path.basename(log_file)} ({file_size} bytes)")

        return log_files

    except FileNotFoundError:
        print(f"Results directory {results_dir} does not exist.")
    except Exception as e:
        print(f"Error listing log files: {e}")
        return []


def view_log_tail(log_file: str, lines: int = 20) -> None:
    """View the last N lines of a log file."""
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:]
            print(f"\nLast {lines} lines of {os.path.basename(log_file)}:")
            print("=" * 60)
            for line in tail_lines:
                print(line.rstrip())
    except FileNotFoundError:
        print(f"Log file {log_file} does not exist.")
    except Exception as e:
        print(f"Error reading log file: {e}")


def format_metric_value(value: float, is_percentage: bool = False) -> str:
    """Format the numeric value based on its type.

    * For percentage metrics, the value is formatted as a percentage (e.g., ``75.00%``).
    * For other metrics, the value is formatted as a float with three decimal places (e.g., ``123.456``).

    Parameters
    ----------
    value : float
        The metric value.
    is_percentage : bool, optional
        Whether to format as percentage.

    Returns
    -------
    str
        Formatted string.
    """
    if value == float("inf") or value == float("-inf"):
        return "N/A"
    if is_percentage:
        return f"{value * 100:.3f}%"
    return f"{value:.3f}"
