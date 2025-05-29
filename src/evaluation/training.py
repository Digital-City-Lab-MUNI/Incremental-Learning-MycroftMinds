import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator
import time

# Import from our project's modules
from src.evaluation.metrics import calculate_metrics
from src.utils.memory import calculate_memory_usage
from src.utils.utils import get_device_data

def _partial_fit_model(model: BaseEstimator,
                      X: np.array,
                      Y: pd.Series) -> None:
    """
    Fit the model using partial_fit or learn_one method.
    Args:
        model (sklearn.BaseEstimator): The model to be fitted.
        X (np.ndarray): Feature data.
        y (pd.Series): Target data.
    """
    if hasattr(model, 'partial_fit'):
        model.partial_fit(X, Y)
    else:
        # Adaptation for River or other models with learn_one method
        X = pd.DataFrame(X).to_dict(orient='records')
        for x, y in zip(X, Y):
            model.learn_one(x, y)

def _predict_model(model: BaseEstimator,
                    X: np.array) -> np.array:
    """
    Predict using the model.
    Args:
        model (sklearn.BaseEstimator): The model to be used for prediction.
        X (np.ndarray): Feature data.
    Returns:
        np.ndarray: Predicted values.
    """
    if hasattr(model, 'predict'):
        return model.predict(X)
    else:
        # Adaptation for River or other models with predict_one method
        return np.array([model.predict_one(x) for x in X])


def seq_fit_predict_batch_model(model: BaseEstimator, 
                              X_train: np.array, 
                              y_train: pd.Series, 
                              X_valid: np.array, 
                              y_valid: pd.Series,
                              update_freq: str = "14 days",
                              predict_train: bool = True,
                              index_cols=2,
                              verbose=False) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using regular batch-learning models by updating 
    the training data with each new point. You may also specify the time-frequency at which to update the model.

    Args:
        model (sklearn.BaseEstimator): The regular batch learning model to use for predictions. Must have `.fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        update_freq (str, optional): Frequency at which to update the model (e.g., "28 days", "14 days", "1 hour"). Default is "14 days".
        predict_train (bool, optional): Whether to predict on the initial training set. Default is True.
        index_cols (int, optional): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.
        verbose (bool, optional): If True, print progress and memory usage. Default is False.

    Returns:
        Tuple(Optional[pd.Series], pd.Series): A tuple containing
            - y_pred_train (pd.Series | None): Predictions for the training set, if `predict_train` is True.
            - y_pred_valid (pd.Series): Sequential predictions for the validation set.
    """
    # Initialize predictions list
    y_pred_train = [] if predict_train else None
    y_pred_valid = []

    # Copy training data to avoid modifying the original
    X_train_seq = X_train.copy()
    y_train_seq = y_train.copy()
    
    index_cols = index_cols  
    update_freq = pd.Timedelta(update_freq) 

    # Iterate through the validation set in chunks based on update frequency
    current_chunk_start = 0
    while current_chunk_start < len(X_valid):
        # Determine the end of the current chunk using timestamp manipulation
        current_time = X_valid[current_chunk_start, 0]  # Assuming the first column in X_valid is the timestamp
        chunk_end_time = current_time + update_freq
        chunk_end = np.searchsorted(X_valid[:, 0], chunk_end_time, side='right')

        # Retrain the model with the updated training set
        if verbose:
            start = time.time()
            print(f"\nRetraining model on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}")
            model.fit(X_train_seq[:, index_cols:], y_train_seq)
            print(f"\nTime: {time.time() - start:.2f} seconds")
        else:
            model.fit(X_train_seq[:, index_cols:], y_train_seq)
        if verbose:
            # Print memory usage after retraining
            memory_usage = calculate_memory_usage(model)
            print(f"Memory usage after retraining: {memory_usage:.4f} MB")

        # Predict for the initial training set if required
        if predict_train and y_train.shape[0] == y_train_seq.shape[0]:
            y_pred_train = model.predict(X_train_seq[:, index_cols:])
            y_pred_train = pd.Series(y_pred_train, index=y_train.index)
            print("Got predictions on training set") if verbose else None

        # Predict for the current chunk
        if verbose:
            print(f"Predicting for chunk from {X_valid[current_chunk_start, 0]} to {X_valid[chunk_end-1, 0]}")
            print()
            print("-" * 25)
        y_pred_valid_chunk = model.predict(X_valid[current_chunk_start:chunk_end, index_cols:])
        y_pred_valid.extend(y_pred_valid_chunk)

        # Add the current chunk of validation data to the training set
        X_train_seq = np.vstack([X_train_seq, X_valid[current_chunk_start:chunk_end]])
        y_train_seq = pd.concat([y_train_seq, pd.Series(y_valid.iloc[current_chunk_start:chunk_end], 
                                                      index=y_valid.index[current_chunk_start:chunk_end])])

        # Move to the next chunk
        current_chunk_start = chunk_end

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)

    return y_pred_train, y_pred_valid

def seq_fit_predict_incremental_model(model: BaseEstimator, 
                                    X_train: np.array, 
                                    y_train: pd.Series, 
                                    X_valid: np.array, 
                                    y_valid: pd.Series,
                                    update_freq: str = "5 min",
                                    predict_train: bool = True,
                                    index_cols=2,
                                    verbose = True) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using incremental learning models by updating 
    the model with each new point. You may also specify the time-frequency at which to update the model.

    Args:
        model (sklearn.BaseEstimator): The incremental learning model to use for predictions. Must have `.partial_fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        update_freq (str, optional): Frequency at which to update the model (e.g., "28 days", "14 days", "1 hour"). Default is "5 min".
        predict_train (bool, optional): Whether to predict on the training set. Default is False.
        index_cols (int, optional): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.

    Returns:
        Tuple(Optional[pd.Series], pd.Series): A tuple containing
            - y_pred_train (pd.Series | None): Predictions for the training set, if `predict_train` is True.
            - y_pred_valid (pd.Series): Sequential predictions for the validation set.
    """
    # Initialize predictions lists
    y_pred_train = [] if predict_train else None
    y_pred_valid = []

    # Copy training data to avoid modifying the original
    X_train_seq = X_train.copy()
    y_train_seq = y_train.copy()
    
    update_freq = pd.Timedelta(update_freq) 

    # Fit the model on the initial training data
    if verbose:
        print(f"Initial training on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}")
    
    _partial_fit_model(model, X_train_seq[:, index_cols:], y_train_seq)

    # Predict on the training set if required
    if predict_train:
        y_pred_train = _predict_model(model, X_train_seq[:, index_cols:])
        y_pred_train = pd.Series(y_pred_train, index=y_train.index)

        if verbose:
            print("Train predictions completed.")

    # Iterate through the validation set
    for i in range(len(X_valid)):
        # Predict the current validation point
        y_pred = _predict_model(model, X_valid[i:i+1, index_cols:])
        y_pred_valid.extend(y_pred)

        # Update the model with the current validation point
        if i + 1 < len(X_valid):  # Ensure we don't go out of bounds
            current_time = X_valid[i, 0]  # Assuming the first column in X_valid is the timestamp
            next_time = X_valid[i + 1, 0]
            if pd.Timestamp(next_time) - pd.Timestamp(current_time) >= update_freq:
                #print(f"Updating model with data up to {next_time}")
                _partial_fit_model(model, X_valid[i:i+1, index_cols:], y_valid.iloc[i:i+1])

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)
    if verbose:
        print("Validation predictions completed.")

    return y_pred_train, y_pred_valid


def all_valid_splits_seq_fit_predict(model: BaseEstimator,
                             train: List[Tuple[np.ndarray, np.ndarray]], 
                             valid: List[Tuple[np.ndarray, np.ndarray]], 
                             incremental: bool,
                             data_freq: str,
                             update_freq="14D",
                             spike_prominence=10.,
                             normalize_robust=True,
                             start_split: int = 0, 
                             end_split: Optional[int] = None,
                             predict_train: bool = True,
                             index_cols=2,
                             verbose: bool = True
                             ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit and predict for each device in the training and validation splits. The model is updated every update_freq days.

    Args:
        model (sklearn.BaseEstimator): The model to be used for fitting and predicting.
        train (List[Tuple[np.ndarray, np.ndarray]]): The training data splits.
        valid (List[Tuple[np.ndarray, np.ndarray]]): The validation data splits.
        incremental (bool): If True, use incremental learning; otherwise, use batch learning.
        data_freq (str): The frequency of the data (e.g., "1s", "5min", "H", etc.).
        update_freq (str): The frequency at which the model is updated. Must be a valid pandas offset string (e.g., "14D" for 14 days, "5m" for 5 minutes, etc.).
        spike_prominence (float): The prominence parameter for spike detection metrics. The data is normalized before calculating the spikes, so the prominence now means distance in std or MAD units (1.5 => 1.5*std from mean).
        normalize_robust (bool): If True, use robust (median&MAD) normalization, else use regular z-score (mean&std)normalization.
        start_split (int): The starting split for processing the splits.
        end_split (Optional[int]): The ending split for processing the splits. If None, it will process all splits.
        predict_train (bool): If True, predict on the training set.
        index_cols (int): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.
        verbose (bool): If True, print validation metrics for each device and split (period).

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            device_train_period_predictions (Dict[str, Any]): Predictions for the training period for each device.
            device_valid_period_predictions (Dict[str, Any]): Predictions for the validation period for each device.
            device_train_period_metrics (Dict[str, Any]): Metrics for the training period for each device.
            device_valid_period_metrics (Dict[str, Any]): Metrics for the validation period for each device.
    """

    device_train_period_metrics = {}
    device_train_period_predictions = {}
    device_valid_period_metrics = {}
    device_valid_period_predictions = {}
    end_split = len(train) if not end_split else end_split

    for split in tqdm(range(start_split, end_split), 
                      desc="Processing splits", 
                      position=0):
        for device_id in tqdm(set(train[split][0][:, 1]), 
                              desc=f"Processing devices in split {split}", 
                              position=1, 
                              leave=False):
            X_train, y_train = get_device_data(train[split], device_id=device_id)
            X_valid, y_valid = get_device_data(valid[split], device_id=device_id)

            if incremental:
                y_pred_train, y_pred_valid = seq_fit_predict_incremental_model(model, X_train, y_train, X_valid, y_valid, update_freq, predict_train, index_cols, verbose)
            else:
                y_pred_train, y_pred_valid = seq_fit_predict_batch_model(model, X_train, y_train, X_valid, y_valid, update_freq, predict_train, index_cols, verbose)

            train_period = f"{X_train[0, 0]} - {X_train[-1, 0]}"
            valid_period = f"{X_valid[0, 0]} - {X_valid[-1, 0]}"
            
            device_train_predictions = {
                'period': train_period,
                'y_true': y_train,
                'y_pred': y_pred_train
            }

            device_valid_predictions = {
                'period': valid_period,
                'y_true': y_valid,
                'y_pred': y_pred_valid
            }
            
            # Only calculate train metrics if predict_train is True
            if predict_train:
                device_train_metrics = calculate_metrics(
                    y_train,
                    y_pred_train,
                    device_id,
                    train_period,
                    data_freq=data_freq,
                    spike_prominence=spike_prominence,
                    normalize_robust=normalize_robust,
                    verbose=False
                    )
            else:
                device_train_metrics = None
                
            device_valid_metrics = calculate_metrics(
                y_valid,
                y_pred_valid,
                device_id,
                valid_period,
                data_freq=data_freq,
                spike_prominence=spike_prominence,
                normalize_robust=normalize_robust,
                verbose=verbose
                )

            if device_id not in device_train_period_predictions:
                device_train_period_predictions[device_id] = []
            device_train_period_predictions[device_id].append(device_train_predictions)

            if device_id not in device_valid_period_predictions:
                device_valid_period_predictions[device_id] = []
            device_valid_period_predictions[device_id].append(device_valid_predictions)

            if device_id not in device_train_period_metrics:
                device_train_period_metrics[device_id] = []
            device_train_period_metrics[device_id].append(device_train_metrics)
            
            if device_id not in device_valid_period_metrics:
                device_valid_period_metrics[device_id] = []
            device_valid_period_metrics[device_id].append(device_valid_metrics)

    # report missing values
    for device_id in device_train_period_predictions:
        for period in device_train_period_predictions[device_id]:
            if period['y_true'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_true for device {device_id} during the training period {period['period']}!")
            if period['y_pred'] is not None and period['y_pred'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the training period {period['period']}!")
    
    for device_id in device_valid_period_predictions:
        for period in device_valid_period_predictions[device_id]:
            if period['y_true'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_true for device {device_id} during the validation period {period['period']}!")
            if period['y_pred'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the validation period {period['period']}!")

    return (device_train_period_predictions, 
            device_valid_period_predictions, 
            device_train_period_metrics, 
            device_valid_period_metrics)

def test_set_seq_fit_predict(model: BaseEstimator,
                            train: List[Tuple[np.ndarray, np.ndarray]], 
                            valid: List[Tuple[np.ndarray, np.ndarray]],
                            test: List[Tuple[np.ndarray, np.ndarray]],
                            incremental: bool,
                            data_freq: str,
                            update_freq="14D",
                            spike_prominence=10.,
                            normalize_robust=True,
                            predict_train: bool = False,
                            index_cols=2,
                            verbose: bool = True
                            ) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit the model on combined data from the last training and validation splits, then predict on the test set.
    
    Args:
        model (sklearn.BaseEstimator): The model to be used for fitting and predicting.
        train (List[Tuple[np.ndarray, np.ndarray]]): The training data splits.
        valid (List[Tuple[np.ndarray, np.ndarray]]): The validation data splits.
        test (List[Tuple[np.ndarray, np.ndarray]]): The test data split.
        incremental (bool): If True, use incremental learning; otherwise, use batch learning.
        data_freq (str): The frequency of the data (e.g., "1s", "5min", "H", etc.).
        update_freq (str): The frequency at which the model is updated.
        spike_prominence (float): The prominence parameter for spike detection metrics. The data is normalized before calculating the spikes, so the prominence now means distance in std or MAD units (1.5 => 1.5*std from mean).
        normalize_robust (bool): If True, use robust (median&MAD) normalization, else use regular z-score (mean&std)normalization.
        predict_train (bool): If True, predict on the training set.
        index_cols (int): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.
        verbose (bool): If True, print test metrics for each device.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            device_train_period_predictions (Dict[str, Any]): Predictions for the training period for each device.
            device_test_period_predictions (Dict[str, Any]): Predictions for the test period for each device.
            device_train_period_metrics (Dict[str, Any]): Metrics for the training period for each device.
            device_test_period_metrics (Dict[str, Any]): Metrics for the test period for each device.
    """
    device_train_period_metrics = {}
    device_train_period_predictions = {}
    device_test_period_metrics = {}
    device_test_period_predictions = {}
    
    # Get the last split from training and validation
    last_train_split = train[-1]
    last_valid_split = valid[-1]
    
    for device_id in tqdm(set(last_train_split[0][:, 1]).union(set(last_valid_split[0][:, 1])), 
                          desc="Processing devices for test predictions", 
                          position=0):
        # Extract device data from last training and validation splits
        X_train, y_train = get_device_data(last_train_split, device_id=device_id)
        X_valid, y_valid = get_device_data(last_valid_split, device_id=device_id)
        
        # Combine training and validation data for model training
        X_combined = np.vstack([X_train, X_valid]) if len(X_train) > 0 and len(X_valid) > 0 else (X_train if len(X_train) > 0 else X_valid)
        y_combined = pd.concat([y_train, y_valid]) if len(y_train) > 0 and len(y_valid) > 0 else (y_train if len(y_train) > 0 else y_valid)
        
        # Get test data for this device
        X_test, y_test = get_device_data(test, device_id=device_id)  # Assuming test has only one split
        
        # Skip if device is not in test set
        if len(X_test) == 0:
            continue
            
        # Train model and predict on test data
        if incremental:
            y_pred_train, y_pred_test = seq_fit_predict_incremental_model(model, X_combined, y_combined, X_test, y_test, update_freq, predict_train, index_cols)
        else:
            y_pred_train, y_pred_test = seq_fit_predict_batch_model(model, X_combined, y_combined, X_test, y_test, update_freq, predict_train, index_cols)

        # Store periods info
        train_period = f"{X_combined[0, 0]} - {X_combined[-1, 0]}"
        test_period = f"{X_test[0, 0]} - {X_test[-1, 0]}"
        
        # Store predictions
        device_train_predictions = {
            'period': train_period,
            'y_true': y_combined,
            'y_pred': y_pred_train
        }
        
        device_test_predictions = {
            'period': test_period,
            'y_true': y_test,
            'y_pred': y_pred_test
        }
        
        # Calculate metrics
        if predict_train:
            device_train_metrics = calculate_metrics(
                y_combined,
                y_pred_train,
                device_id,
                train_period,
                data_freq=data_freq,
                spike_prominence=spike_prominence,
                normalize_robust=normalize_robust,
                verbose=False
            )
        else:
            device_train_metrics = None
        
        device_test_metrics = calculate_metrics(
            y_test,
            y_pred_test,
            device_id,
            test_period,
            data_freq=data_freq,
            spike_prominence=spike_prominence,
            normalize_robust=normalize_robust,
            verbose=verbose
        )

        # Store results
        if device_id not in device_train_period_predictions:
            device_train_period_predictions[device_id] = []
        device_train_period_predictions[device_id].append(device_train_predictions)
        
        if device_id not in device_test_period_predictions:
            device_test_period_predictions[device_id] = []
        device_test_period_predictions[device_id].append(device_test_predictions)
        
        if device_id not in device_train_period_metrics:
            device_train_period_metrics[device_id] = []
        if device_train_metrics is not None:  # Only append metrics if they were calculated
            device_train_period_metrics[device_id].append(device_train_metrics)
        
        if device_id not in device_test_period_metrics:
            device_test_period_metrics[device_id] = []
        device_test_period_metrics[device_id].append(device_test_metrics)
    
    # Report missing values
    for device_id in device_train_period_predictions:
        for period in device_train_period_predictions[device_id]:
            if period['y_true'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_true for device {device_id} during the training period {period['period']}!")
            if period['y_pred'] is not None and period['y_pred'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the training period {period['period']}!")
    
    for device_id in device_test_period_predictions:
        for period in device_test_period_predictions[device_id]:
            if period['y_true'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_true for device {device_id} during the test period {period['period']}!")
            if period['y_pred'].isnull().any():
                print(f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the test period {period['period']}!")

    return (device_train_period_predictions, 
            device_test_period_predictions, 
            device_train_period_metrics, 
            device_test_period_metrics)
