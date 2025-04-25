import sys
import numpy as np
import pandas as pd
import traceback
from typing import Any, Tuple

def calculate_memory_usage(obj: Any) -> float:
    """Calculate the approximate memory usage of an object in MB.
    
    Args:
        obj: The object to measure
        
    Returns:
        Memory usage in MB
    """
    def _sizeof(obj):
        if isinstance(obj, (np.ndarray)):
            return obj.nbytes
        elif isinstance(obj, (pd.DataFrame, pd.Series)):
            return obj.memory_usage(deep=True).sum()
        elif isinstance(obj, (list, tuple)):
            return sum(_sizeof(x) for x in obj)
        else:
            return sys.getsizeof(obj)
            
    return _sizeof(obj) / (1024 * 1024)  # Convert to MB

def optimize_memory_usage(data_tuple: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Optimize memory usage of a data tuple by converting to efficient dtypes.
    
    Args:
        data_tuple: A tuple of (X, y) where X is a feature matrix and y is a target vector
        
    Returns:
        The optimized (X, y) tuple
    """
    X, y = data_tuple
    
    # Handle non-numpy arrays
    if not isinstance(X, np.ndarray) or not isinstance(y, np.ndarray):
        return X, y
        
    # Skip optimization for object dtypes (strings, timestamps, etc.)
    if X.dtype == object or y.dtype == object:
        return X, y
    
    # Optimize numeric data
    def optimize_array(arr):
        # For floating point arrays
        if np.issubdtype(arr.dtype, np.floating):
            # If the range fits in float32, use that
            if np.nanmin(arr) >= np.finfo(np.float32).min and np.nanmax(arr) <= np.finfo(np.float32).max:
                return arr.astype(np.float32)
        # For integer arrays
        elif np.issubdtype(arr.dtype, np.integer):
            # Find the smallest integer type that fits the data range
            min_val, max_val = np.nanmin(arr), np.nanmax(arr)
            for dtype in [np.int8, np.uint8, np.int16, np.uint16, np.int32]:
                info = np.iinfo(dtype)
                if min_val >= info.min and max_val <= info.max:
                    return arr.astype(dtype)
        return arr
        
    # Apply optimization if possible
    try:
        # Handle numeric columns in X
        if X.ndim == 2:
            # Skip first 2 columns (timestamp and device_id) as they're not numeric
            numeric_cols = X[:, 2:].astype(float)
            optimized_cols = optimize_array(numeric_cols)
            X_optimized = np.column_stack((X[:, :2], optimized_cols))
        else:
            X_optimized = optimize_array(X)
            
        y_optimized = optimize_array(y)
        
        return X_optimized, y_optimized
        
    except Exception as e:
        print(f"[WARNING] Memory optimization failed: {e}")
        traceback.print_exc()
        return X, y

def print_memory_stats(train, valid, test):
    """Print detailed memory usage statistics for datasets.
    
    Args:
        train: Training dataset
        valid: Validation dataset
        test: Test dataset
    """
    # For walk-forward validation structure
    if isinstance(train, list):
        train_mem = sum(calculate_memory_usage(ts) for ts in train)
        valid_mem = sum(calculate_memory_usage(vs) for vs in valid)
        test_mem = calculate_memory_usage(test)
        
        # Print summary statistics
        print(f"Memory usage:")
        print(f"  Train: {train_mem:.2f} MB ({len(train)} splits)")
        print(f"  Valid: {valid_mem:.2f} MB ({len(valid)} splits)")
        print(f"  Test:  {test_mem:.2f} MB")
        print(f"  Total: {train_mem + valid_mem + test_mem:.2f} MB")
        
    else:
        # For standard dataset structure
        train_mem = calculate_memory_usage(train)
        valid_mem = calculate_memory_usage(valid)
        test_mem = calculate_memory_usage(test)
        
        # Print summary statistics
        print(f"Memory usage:")
        print(f"  Train: {train_mem:.2f} MB")
        print(f"  Valid: {valid_mem:.2f} MB")
        print(f"  Test:  {test_mem:.2f} MB")
        print(f"  Total: {train_mem + valid_mem + test_mem:.2f} MB")
