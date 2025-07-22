import os
import glob
import pickle
from datetime import datetime
from typing import Dict, Any, List


def setup_log_directory(log_dir: str = None, log_prefix: str = None) -> str:
    """
    Create log directory if it doesn't exist and return the path.
    
    Args:
        log_dir: Directory to save tuning results logs. If None, uses default.
        log_prefix: Prefix for the log directory name. If None, uses default.
    
    Returns:
        str: The log directory path
    """
    if log_dir is None:
        if log_prefix is None:
            log_prefix = "tuning"
        log_dir = f"../results/{log_prefix}_results"
    
    os.makedirs(log_dir, exist_ok=True)
    return log_dir


def create_log_file(log_dir: str, model_name: str, file_prefix: str = None) -> str:
    """
    Create a timestamped log file for a specific model with optional prefix.
    
    Args:
        log_dir: Directory to save the log file
        model_name: Name of the model being tuned
        file_prefix: Optional prefix for the filename (e.g., 'dev_4_wfv')
    
    Returns:
        str: Path to the log file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if file_prefix:
        filename = f"{timestamp}_{file_prefix}_{model_name}_tuning.txt"
    else:
        filename = f"{timestamp}_{model_name}_tuning.txt"
    
    log_file = os.path.join(log_dir, filename)
    return log_file


def log_and_print(log_file: str, message: str, print_console: bool = True):
    """
    Log message to file and optionally print to console.
    
    Args:
        log_file: Path to the log file
        message: Message to log
        print_console: Whether to print to console
    """
    with open(log_file, 'a', encoding='utf-8') as f:
        f.write(message + '\n')
    # if print_console:
    #     print(message)


def save_checkpoint_results(results_dir: str, model_name: str, best_params: Dict[str, Any], 
                          best_scores: Dict[str, Any], file_prefix: str = None):
    """
    Save checkpoint results for a model with optional prefix.
    
    Args:
        results_dir: Directory to save results
        model_name: Name of the model
        best_params: Best parameters found
        best_scores: Best scores achieved
        file_prefix: Optional prefix for the filename (e.g., 'dev_4_wfv')
    """
    os.makedirs(results_dir, exist_ok=True)
    
    if file_prefix:
        filename = f"{file_prefix}_{model_name}_results.pkl"
    else:
        filename = f"{model_name}_results.pkl"
    
    checkpoint_file = os.path.join(results_dir, filename)
    with open(checkpoint_file, 'wb') as f:
        pickle.dump({'best_params': best_params, 'best_scores': best_scores}, f)
    
    print(f"Results saved to {checkpoint_file}")


def load_checkpoint_results(results_dir: str) -> Dict[str, Any]:
    """
    Load all available checkpoint results.
    
    Args:
        results_dir: Directory containing result files
    
    Returns:
        Dict containing loaded results
    """
    results = {}
    
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return results
    
    result_files = glob.glob(f"{results_dir}/*_results.pkl")
    for file_path in result_files:
        model_name = os.path.basename(file_path).replace('_results.pkl', '')
        try:
            with open(file_path, 'rb') as f:
                results[model_name] = pickle.load(f)
            print(f"Loaded {model_name} results")
        except Exception as e:
            print(f"Failed to load {model_name}: {e}")
    
    return results


def list_log_files(results_dir: str) -> List[str]:
    """
    List all log files in the results directory.
    
    Args:
        results_dir: Directory containing log files
    
    Returns:
        List of log file paths
    """
    if not os.path.exists(results_dir):
        print(f"Results directory {results_dir} does not exist.")
        return []
    
    log_files = glob.glob(f"{results_dir}/*.txt")
    log_files.sort()
    
    print(f"Available log files in {results_dir}:")
    for log_file in log_files:
        file_size = os.path.getsize(log_file)
        print(f"  {os.path.basename(log_file)} ({file_size} bytes)")
    
    return log_files


def view_log_tail(log_file: str, lines: int = 20):
    """
    View the last N lines of a log file.
    
    Args:
        log_file: Path to the log file
        lines: Number of lines to show from the end
    """
    try:
        with open(log_file, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
            tail_lines = all_lines[-lines:]
            print(f"\nLast {lines} lines of {os.path.basename(log_file)}:")
            print("=" * 60)
            for line in tail_lines:
                print(line.rstrip())
    except Exception as e:
        print(f"Error reading log file: {e}")


def format_metric_value(value, is_percentage: bool = False) -> str:
    """
    Format metric values for display.
    
    Args:
        value: The metric value
        is_percentage: Whether to format as percentage
    
    Returns:
        Formatted string
    """
    if value == float('inf') or value == float('-inf'):
        return "N/A"
    if is_percentage:
        return f"{value*100:.3f}%"
    return f"{value:.3f}"
