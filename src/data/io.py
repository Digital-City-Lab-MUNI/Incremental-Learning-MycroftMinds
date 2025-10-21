"""I/O helpers for persisting split datasets.

This module centralizes lightweight save / load utilities used for experiment
reproducibility and faster iteration. Two serialization backends are supported:

* NumPy ``.npz`` compressed archives (fast numeric loading, metadata via keys).
* Python ``pickle`` (fallback for arbitrary Python objects / structures).

Each dataset can be either a single ``(X, y)`` tuple or a list of such tuples
representing multiple temporal or device splits. The save functions embed a
``data_type`` flag so the corresponding load function can reconstruct the
original structure automatically.
"""

import os
import numpy as np
import traceback
import pickle
from typing import List, Tuple, Union, Any


def save_data_numpy(
    data: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    filename: str,
    save_dir: str = "../data",
) -> str:
    """Persist data splits to a compressed ``.npz`` archive.

    Parameters
    ----------
    data : list[tuple[numpy.ndarray, numpy.ndarray]] | tuple[numpy.ndarray, numpy.ndarray]
        Either a list of ``(X, y)`` tuples (multiple splits) or a single
        ``(X, y)`` tuple.
    filename : str
        Basename (without extension) of the output file.
    save_dir : str, default '../data'
        Directory in which to write the file (created if missing).

    Returns
    -------
    str
        Absolute or relative path to the created ``.npz`` file. Empty string on
        failure.
    """
    save_path = os.path.join(save_dir, f"{filename}.npz")

    try:
        # Handle list of tuples (multiple splits)
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], tuple):
            # Create dictionaries to store X and y for each split
            X_arrays = {}
            y_arrays = {}

            for i, (X, y) in enumerate(data):
                X_arrays[f"X_{i}"] = X
                y_arrays[f"y_{i}"] = y

            # Save the arrays
            np.savez_compressed(
                save_path,
                **X_arrays,
                **y_arrays,
                num_splits=len(data),
                data_type="split_tuples",
            )

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


def save_data_pickle(
    data: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    filename: str,
    save_dir: str = "../data",
) -> str:
    """Persist data splits using Python pickle.

    Parameters
    ----------
    data : list[tuple[numpy.ndarray, numpy.ndarray]] | tuple[numpy.ndarray, numpy.ndarray]
        List of splits or single split to serialize.
    filename : str
        Basename (no extension) for output.
    save_dir : str, default '../data'
        Output directory (created if necessary).

    Returns
    -------
    str
        Path to the written pickle file or empty string on failure.
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_path = os.path.join(save_dir, f"{filename}.pkl")

    try:
        with open(save_path, "wb") as f:
            pickle.dump(data, f)
        print(f"Data saved to {save_path}")
        return save_path

    except Exception as e:
        print(f"[ERROR] Failed to save data using pickle: {e}")
        traceback.print_exc()
        return ""


def load_data_numpy(
    filename: str, load_dir: str = "../data", max_length=-1
) -> Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], None]:
    """Load splits from a ``.npz`` archive produced by :func:`save_data_numpy`.

    Parameters
    ----------
    filename : str
        Basename (without extension) of the ``.npz`` file.
    load_dir : str, default '../data'
        Directory from which to read.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]] | tuple[numpy.ndarray, numpy.ndarray] | None
        List of splits, a single split tuple, or ``None`` if the file doesn't
        exist or loading fails.
    """
    load_path = os.path.join(load_dir, f"{filename}.npz")

    if not os.path.exists(load_path):
        print(f"[ERROR] File {load_path} does not exist.")
        return None

    try:
        # Load the compressed arrays
        data = np.load(load_path, allow_pickle=True)
        data_type = str(data.get("data_type", "split_tuples"))

        if data_type == "split_tuples":
            # Get the number of splits
            num_splits = int(data["num_splits"])

            # Reconstruct the list of tuples
            result = []
            for i in range(num_splits):
                X = data[f"X_{i}"]
                y = data[f"y_{i}"]
                result.append((X, y))

            return result[:max_length]

        elif data_type == "single_tuple":
            # Return a single tuple
            return data["X"], data["y"]

        else:
            print(f"[ERROR] Unknown data type: {data_type}")
            return None

    except Exception as e:
        print(f"[ERROR] Failed to load data using NumPy: {e}")
        traceback.print_exc()
        return None


def load_data_pickle(
    filename: str, load_dir: str = "../data", max_length=-1
) -> Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray], None]:
    """Load splits from a pickle file produced by :func:`save_data_pickle`.

    Parameters
    ----------
    filename : str
        Basename (without extension) of the pickle file.
    load_dir : str, default '../data'
        Directory from which to read.

    Returns
    -------
    list[tuple[numpy.ndarray, numpy.ndarray]] | tuple[numpy.ndarray, numpy.ndarray] | None
        Restored list of splits, single split, or ``None`` on failure.
    """
    load_path = os.path.join(load_dir, f"{filename}.pkl")

    if not os.path.exists(load_path):
        print(f"[ERROR] File {load_path} does not exist.")
        return None

    try:
        with open(load_path, "rb") as f:
            data = pickle.load(f)[:max_length]
        return data

    except Exception as e:
        print(f"[ERROR] Failed to load data using pickle: {e}")
        traceback.print_exc()
        return None


def save_dataset(
    train: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    valid: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    test: Union[List[Tuple[np.ndarray, np.ndarray]], Tuple[np.ndarray, np.ndarray]],
    data_dir: str = "../data",
    prefix: str = "",
    use_numpy: bool = True,
) -> Tuple[str, str, str]:
    """Save train/validation/test splits using a chosen backend.

    Parameters
    ----------
    train, valid, test : list[tuple[numpy.ndarray, numpy.ndarray]] | tuple[numpy.ndarray, numpy.ndarray]
        Dataset splits to store.
    data_dir : str, default '../data'
        Target directory for saved artifacts.
    prefix : str, default ''
        Optional filename prefix (e.g. experiment id) applied to each file.
    use_numpy : bool, default True
        If True use compressed NumPy archives; otherwise pickle.

    Returns
    -------
    tuple[str, str, str]
        Paths to the saved (train_path, valid_path, test_path).
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


def load_dataset(
    data_dir: str = "../data",
    prefix: str = "",
    use_numpy: bool = True,
    train: bool = True,
    valid: bool = True,
    test: bool = True,
    max_length=-1,
) -> Tuple[Any, Any, Any]:
    """Load persisted dataset splits.

    Parameters
    ----------
    data_dir : str, default '../data'
        Directory containing saved artifacts.
    prefix : str, default ''
        Filename prefix used during saving.
    use_numpy : bool, default True
        If True expect NumPy archives; otherwise pickle files.
    train, valid, test : bool, default True
        Flags controlling which splits to load. Unrequested splits retain their
        input argument value (which callers typically ignore) if False.

    Returns
    -------
    tuple
        Tuple ``(train, valid, test)`` where each element is either a list of
        splits, a single split tuple, or ``None`` if not loaded / unavailable.
    """
    # Set filenames based on prefix
    prefix = f"{prefix}_" if prefix else ""

    if use_numpy:
        if train:
            print("[INFO] Loading train data...")
            train = load_data_numpy(f"{prefix}train", data_dir, max_length=max_length)
            print(f"[INFO] Train data loaded: {len(train)} samples")
        if valid:
            print("[INFO] Loading valid data...")
            valid = load_data_numpy(f"{prefix}valid", data_dir, max_length=max_length)
            print(f"[INFO] Valid data loaded: {len(valid)} samples")
        if test:
            print("[INFO] Loading test data...")
            test = load_data_numpy(f"{prefix}test", data_dir, max_length=max_length)
            print(f"[INFO] Test data loaded: {len(test)} samples")

    else:
        train_file = os.path.join(data_dir, f"{prefix}train.pkl")
        valid_file = os.path.join(data_dir, f"{prefix}valid.pkl")
        test_file = os.path.join(data_dir, f"{prefix}test.pkl")

        # Check if all files exist before loading
        if not all(os.path.exists(f) for f in [train_file, valid_file, test_file]):
            missing = [f for f in [train_file, valid_file, test_file] if not os.path.exists(f)]
            print(f"[ERROR] The following files do not exist: {missing}")
            return None, None, None

        # Load data using pickle
        print("[INFO] Loading train data...")
        train = load_data_pickle(f"{prefix}train", data_dir, max_length=max_length)
        print(f"[INFO] Train data loaded: {len(train)} samples")

        print("[INFO] Loading valid data...")
        valid = load_data_pickle(f"{prefix}valid", data_dir, max_length=max_length)
        print(f"[INFO] Valid data loaded: {len(valid)} samples")

        print("[INFO] Loading test data...")
        test = load_data_pickle(f"{prefix}test", data_dir, max_length=max_length)
        print(f"[INFO] Test data loaded: {len(test)} samples")

    return train, valid, test
