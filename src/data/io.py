import os
import numpy as np
import traceback
import pandas as pd
import sys
import pickle
from typing import List, Tuple, Dict, Optional, Union, Any

def save_data_numpy(data: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]], 
                   filename: str, 
                   save_dir: str = '../data') -> str:
    """Save data using NumPy's compressed format for efficient storage and fast I/O.
    
    Args:
        data: The data to save, either a list of (X, y) tuples or a single (X, y) tuple
        filename: The filename without extension
        save_dir: Directory to save the data in
        
    Returns:
        The path to the saved file
    """
    save_path = os.path.join(save_dir, f'{filename}.npz')
    
    try:
        # Handle list of tuples (multiple splits)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
            # Create dictionaries to store X and y for each split
            X_arrays = {}
            y_arrays = {}
            
            for i, (X, y) in enumerate(data):
                X_arrays[f'X_{i}'] = X
                y_arrays[f'y_{i}'] = y
            
            # Save the arrays
            np.savez_compressed(save_path, 
                              **X_arrays, 
                              **y_arrays, 
                              num_splits=len(data),
                              data_type="split_tuples")
            
        # Handle single tuple (X, y)
        elif isinstance(data, tuple) and len(data) == 2:
            X, y = data
            np.savez_compressed(save_path, X=X, y=y, data_type="single_tuple")
        
        else:
            raise ValueError("Data format not recognized for NumPy saving")
            
        print(f"Data saved to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save data using NumPy: {e}")
        traceback.print_exc()
        return ""

def save_data_pickle(data: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
                    filename: str,
                    save_dir: str = '../data') -> str:
    """Save data using pickle format.
    
    Args:
        data: The data to save, either a list of (X, y) tuples or a single (X, y) tuple
        filename: The filename without extension
        save_dir: Directory to save the data in
        
    Returns:
        The path to the saved file
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    save_path = os.path.join(save_dir, f'{filename}.pkl')
    
    try:
        with open(save_path, 'wb') as f:
            pickle.dump(data, f)
        print(f"Data saved to {save_path}")
        return save_path
        
    except Exception as e:
        print(f"[ERROR] Failed to save data using pickle: {e}")
        traceback.print_exc()
        return ""

def load_data_numpy(filename: str, 
                   load_dir: str = '../data') -> Union[List[Tuple[np.ndarray, np.ndarray]], 
                                                       Tuple[np.ndarray, np.ndarray],
                                                       None]:
    """Load data saved with NumPy's compressed format.
    
    Args:
        filename: The filename without extension
        load_dir: Directory to load the data from
        
    Returns:
        Either a list of tuples containing (X, y) for each split or a single (X, y) tuple
    """
    load_path = os.path.join(load_dir, f'{filename}.npz')
    
    if not os.path.exists(load_path):
        print(f"[ERROR] File {load_path} does not exist.")
        return None
    
    try:
        # Load the compressed arrays
        data = np.load(load_path, allow_pickle=True)
        data_type = str(data.get('data_type', 'split_tuples'))
        
        if data_type == "split_tuples":
            # Get the number of splits
            num_splits = int(data['num_splits'])
            
            # Reconstruct the list of tuples
            result = []
            for i in range(num_splits):
                X = data[f'X_{i}']
                y = data[f'y_{i}']
                result.append((X, y))
            
            return result
        
        elif data_type == "single_tuple":
            # Return a single tuple
            return data['X'], data['y']
        
        else:
            print(f"[ERROR] Unknown data type: {data_type}")
            return None
            
    except Exception as e:
        print(f"[ERROR] Failed to load data using NumPy: {e}")
        traceback.print_exc()
        return None

def load_data_pickle(filename: str,
                    load_dir: str = '../data') -> Union[List[Tuple[np.ndarray, np.ndarray]], 
                                                       Tuple[np.ndarray, np.ndarray],
                                                       None]:
    """Load data saved with pickle format.
    
    Args:
        filename: The filename without extension
        load_dir: Directory to load the data from
        
    Returns:
        Either a list of tuples containing (X, y) for each split or a single (X, y) tuple
    """
    load_path = os.path.join(load_dir, f'{filename}.pkl')
    
    if not os.path.exists(load_path):
        print(f"[ERROR] File {load_path} does not exist.")
        return None
    
    try:
        with open(load_path, 'rb') as f:
            data = pickle.load(f)
        return data
        
    except Exception as e:
        print(f"[ERROR] Failed to load data using pickle: {e}")
        traceback.print_exc()
        return None

def save_dataset(train, valid, test, data_dir='../data', prefix='', use_numpy=True):
    """Save the train, validation, and test datasets.
    
    Args:
        train: Training dataset
        valid: Validation dataset
        test: Test dataset
        data_dir: Directory to save the data in
        prefix: Optional prefix for filenames
        use_numpy: Whether to use NumPy (True) or pickle (False) for saving
        
    Returns:
        Tuple of paths to the saved files (train_path, valid_path, test_path)
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        
    # Set filenames based on prefix
    prefix = f"{prefix}_" if prefix else ""
    
    if use_numpy:
        print(f"[INFO] Saving train data to {data_dir}...")
        train_path = save_data_numpy(train, f"{prefix}train", data_dir)
        print(f"[INFO] Train data saved: {len(train)} samples")
        print(f"[INFO] Saving valid data to {data_dir}...")
        valid_path = save_data_numpy(valid, f"{prefix}valid", data_dir)
        print(f"[INFO] Valid data saved: {len(valid)} samples")
        print(f"[INFO] Saving test data to {data_dir}...")
        test_path = save_data_numpy(test, f"{prefix}test", data_dir)
        print(f"[INFO] Test data saved: {len(test)} samples")
    else:
        train_path = save_data_pickle(train, f"{prefix}train", data_dir)
        valid_path = save_data_pickle(valid, f"{prefix}valid", data_dir)
        test_path = save_data_pickle(test, f"{prefix}test", data_dir)
            
    return train_path, valid_path, test_path

def load_dataset(data_dir='../data', prefix='', use_numpy=True, train=True, valid=True, test=True):
    """Load the train, validation, and test datasets.
    
    Args:
        data_dir: Directory to load the data from
        prefix: Optional prefix for filenames
        use_numpy: Whether to use NumPy (True) or pickle (False) for loading
        train: Whether to load the training dataset
        valid: Whether to load the validation dataset
        test: Whether to load the test dataset
        
    Returns:
        Tuple of (train, valid, test) datasets
    """
    # Set filenames based on prefix
    prefix = f"{prefix}_" if prefix else ""
    
    if use_numpy:
        if train:
            print('[INFO] Loading train data...')
            train = load_data_numpy(f"{prefix}train", data_dir)
            print(f'[INFO] Train data loaded: {len(train)} samples')
        if valid:
            print('[INFO] Loading valid data...')
            valid = load_data_numpy(f"{prefix}valid", data_dir)
            print(f'[INFO] Valid data loaded: {len(valid)} samples')
        if test:
            print('[INFO] Loading test data...')
            test = load_data_numpy(f"{prefix}test", data_dir)
            print(f'[INFO] Test data loaded: {len(test)} samples')
    
    else:
        train_file = os.path.join(data_dir, f'{prefix}train.pkl')
        valid_file = os.path.join(data_dir, f'{prefix}valid.pkl')    
        test_file = os.path.join(data_dir, f'{prefix}test.pkl')
        
        # Check if all files exist before loading
        if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
            missing = [f for f in [train_file, valid_file, test_file] if not os.path.exists(f)]
            print(f"[ERROR] The following files do not exist: {missing}")
            return None, None, None
        
        # Load data using pickle
        print('[INFO] Loading train data...')
        train = load_data_pickle(f"{prefix}train", data_dir)
        print(f'[INFO] Train data loaded: {len(train)} samples')

        print('[INFO] Loading valid data...')
        valid = load_data_pickle(f"{prefix}valid", data_dir)
        print(f'[INFO] Valid data loaded: {len(valid)} samples')

        print('[INFO] Loading test data...')
        test = load_data_pickle(f"{prefix}test", data_dir)
        print(f'[INFO] Test data loaded: {len(test)} samples')
            
    return train, valid, test