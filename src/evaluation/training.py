import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Type
from tqdm.notebook import tqdm
from sklearn.base import BaseEstimator
import time

# Import from our project's modules
from src.evaluation.metrics import calculate_metrics
from src.utils.memory import calculate_memory_usage
from src.utils.utils import get_device_data
from src.data.feature_engineering import filter_features_for_forecast_horizon


def _init_model_with_params(
    model_class: Type[BaseEstimator], model_params: Dict[str, Any]
) -> BaseEstimator:
    """
    Initialize a model with the given class and parameters.

    Args:
        model_class (Type[BaseEstimator]): The class of the model to be initialized.
        model_params (Dict[str, Any]): Parameters to initialize the model with.

    Returns:
        BaseEstimator: An instance of the model initialized with the provided parameters.
    """
    try:
        # Attempt to initialize the model with the provided parameters
        return model_class(**model_params)

    except Exception as e:
        print(
            f"[ERROR] Failed to initialize model '{model_class.__name__}' with parameters {model_params}."
        )
        raise


def _partial_fit_model(model: BaseEstimator, X: np.array, Y: pd.Series) -> None:
    """
    Fit the model using partial_fit or learn_one method.
    Args:
        model (sklearn.BaseEstimator): The model to be fitted.
        X (np.ndarray): Feature data.
        y (pd.Series): Target data.
    """
    if hasattr(model, "partial_fit"):
        model.partial_fit(X, Y)
    else:
        # Adaptation for River or other models with learn_one method (convert each row to a dict)
        X: dict = pd.DataFrame(X).to_dict(orient="records")
        for x, y in zip(X, Y):
            model.learn_one(x, y)


def _predict_model(model: BaseEstimator, X: np.array) -> np.array:
    """
    Predict using the model.
    Args:
        model (sklearn.BaseEstimator): The model to be used for prediction.
        X (np.ndarray): Feature data.
    Returns:
        np.ndarray: Predicted values.
    """
    if hasattr(model, "predict"):
        return model.predict(X)
    else:
        # Adaptation for River or other models with predict_one method (convert each row to a dict)
        X: dict = pd.DataFrame(X).to_dict(orient="records")
        return np.array([model.predict_one(x) for x in X])


def seq_fit_predict_batch_model(
    model: BaseEstimator,
    X_train: np.array,
    y_train: pd.Series,
    X_valid: np.array,
    y_valid: pd.Series,
    forecast_horizon: str,
    update_freq: str = "14 days",
    features=None,
    predict_train: bool = True,
    index_cols=2,
    verbose=0,
) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using regular batch-learning models by updating
    the training data with each new point. You may also specify the time-frequency at which to update the model.

    Args:
        model (sklearn.BaseEstimator): The regular batch learning model to use for predictions. Must have `.fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        forecast_horizon (str): How far ahead to forecast. Used to filter features that would not be available in real forecasting.
        update_freq (str, optional): Frequency at which to update the model (e.g., "28 days", "14 days", "1 hour"). Default is "14 days".
        features (List[str], optional): List of feature names. If provided, features that aren't available for the forecast horizon will be filtered.
        predict_train (bool, optional): Whether to predict on the initial training set. Default is True.
        index_cols (int, optional): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.
        verbose (int, optional): an integer indicating the verbosity level:
            - 0: No messages.
            - 1: Print progress messages and memory usage.
            - 2: Print progress messages, memory usage, and detailed steps.

    Returns:
        Tuple(Optional[pd.Series], pd.Series): A tuple containing
            - y_pred_train (pd.Series | None): Predictions for the training set, if `predict_train` is True.
            - y_pred_valid (pd.Series): Sequential predictions for the validation set.
    """
    # Initialize predictions list
    y_pred_train = [] if predict_train else None
    y_pred_valid = []

    # Filter features if features list and forecast_horizon are provided
    valid_indices = None
    if features and forecast_horizon:
        valid_features, excluded_features = filter_features_for_forecast_horizon(
            features[index_cols:], forecast_horizon
        )

        # Map feature names to column indices
        valid_indices = [
            index_cols + i
            for i, feat in enumerate(features[index_cols:])
            if feat in valid_features
        ]

        if verbose >= 2:
            print(f"Filtering features for forecast horizon {forecast_horizon}")
            print(f"Original feature count: {len(features) - index_cols}")
            print(f"Filtered feature count: {len(valid_indices)}")
            print(
                f"Excluded lag features: {', '.join(excluded_features) if excluded_features else 'None'}"
            )

    # Use filtered indices if available, otherwise use all features
    feature_indices = valid_indices if valid_indices else slice(index_cols, None)

    # Copy training data to avoid modifying the original
    X_train_seq = X_train.copy()
    y_train_seq = y_train.copy()

    update_freq = pd.Timedelta(update_freq)

    # Iterate through the validation set in chunks based on update frequency
    current_chunk_start = 0
    while current_chunk_start < len(X_valid):
        # Determine the end of the current chunk using timestamp manipulation
        current_time = X_valid[
            current_chunk_start, 0
        ]  # Assuming the first column in X_valid is the timestamp
        chunk_end_time = current_time + update_freq
        chunk_end = np.searchsorted(X_valid[:, 0], chunk_end_time, side="right")

        # Retrain the model with the updated training set
        if verbose >= 1:
            start = time.time()
            print(
                f"\nRetraining model on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}"
            )
            model.fit(X_train_seq[:, feature_indices], y_train_seq)
            print(f"\nTime: {time.time() - start:.2f} seconds")
        else:
            model.fit(X_train_seq[:, feature_indices], y_train_seq)

        # Predict for the initial training set if required
        if predict_train and y_train.shape[0] == y_train_seq.shape[0]:
            y_pred_train = model.predict(X_train_seq[:, feature_indices])
            y_pred_train = pd.Series(y_pred_train, index=y_train.index)
            if verbose == 2:
                print("Got predictions on training set")

        # Predict for the current chunk
        if verbose >= 1:
            print(
                f"Predicting for chunk from {X_valid[current_chunk_start, 0]} to {X_valid[chunk_end-1, 0]}"
            )
            print()
            print("-" * 25)
        y_pred_valid_chunk = model.predict(
            X_valid[current_chunk_start:chunk_end, feature_indices]
        )
        y_pred_valid.extend(y_pred_valid_chunk)

        # Add the current chunk of validation data to the training set
        X_train_seq = np.vstack([X_train_seq, X_valid[current_chunk_start:chunk_end]])
        y_train_seq = pd.concat(
            [
                y_train_seq,
                pd.Series(
                    y_valid.iloc[current_chunk_start:chunk_end],
                    index=y_valid.index[current_chunk_start:chunk_end],
                ),
            ]
        )

        # Move to the next chunk
        current_chunk_start = chunk_end

    if verbose == 2:
        print("Validation predictions completed.")

    # Print memory usage after retraining
    if verbose >= 1:
        memory_usage = calculate_memory_usage(model)
        print(f"Model's memory usage: {memory_usage:.6f} MB")

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)

    return y_pred_train, y_pred_valid


def seq_fit_predict_incremental_model(
    model: BaseEstimator,
    X_train: np.array,
    y_train: pd.Series,
    X_valid: np.array,
    y_valid: pd.Series,
    forecast_horizon: str,
    features=None,
    predict_train: bool = True,
    index_cols=2,
    verbose=0,
) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using incremental learning models by updating
    the model with each new point after the forecast horizon has passed.

    Args:
        model (sklearn.BaseEstimator): The incremental learning model to use for predictions. Must have `.partial_fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        forecast_horizon (str): How far ahead to forecast before updating the model with actual values.
        features (List[str], optional): List of feature names. If provided, features that aren't available for the forecast horizon will be filtered.
        predict_train (bool, optional): Whether to predict on the training set. Default is True.
        index_cols (int, optional): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.
        verbose (int, optional): an integer indicating the verbosity level:
            - 0: No messages.
            - 1: Print progress messages and memory usage.
            - 2: Print progress messages, memory usage, and detailed steps.

    Returns:
        Tuple(Optional[pd.Series], pd.Series): A tuple containing
            - y_pred_train (pd.Series | None): Predictions for the training set, if `predict_train` is True.
            - y_pred_valid (pd.Series): Sequential predictions for the validation set.
    """
    # Initialize predictions lists
    y_pred_train = [] if predict_train else None
    y_pred_valid = []

    # Filter features if features list and forecast_horizon are provided
    valid_indices = None
    if features and forecast_horizon:
        valid_features, excluded_features = filter_features_for_forecast_horizon(
            features[index_cols:], forecast_horizon
        )

        # Map feature names to column indices
        valid_indices = [
            index_cols + i
            for i, feat in enumerate(features[index_cols:])
            if feat in valid_features
        ]

        if verbose >= 1:
            print(f"Filtering features for forecast horizon {forecast_horizon}")
            print(f"Original feature count: {len(features) - index_cols}")
            print(f"Filtered feature count: {len(valid_indices)}")
            print(
                f"Excluded lag features: {', '.join(excluded_features) if excluded_features else 'None'}"
            )

    # Use filtered indices if available, otherwise use all features
    feature_indices = valid_indices if valid_indices else slice(index_cols, None)

    # Copy training data to avoid modifying the original
    X_train_seq = X_train.copy()
    y_train_seq = y_train.copy()

    forecast_horizon_td = pd.Timedelta(forecast_horizon)

    # Fit the model on the initial training data
    if verbose >= 1:
        print(
            f"Initial training on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}"
        )

    _partial_fit_model(model, X_train_seq[:, feature_indices], y_train_seq)

    # Predict on the training set if required
    if predict_train:
        y_pred_train = _predict_model(model, X_train_seq[:, feature_indices])
        y_pred_train = pd.Series(y_pred_train, index=y_train.index)

        if verbose >= 2:
            print("Train predictions completed.")

    # Initialize last_update_time with the last timestamp from training data
    last_update_time = pd.Timestamp(X_train_seq[-1, 0])

    # Iterate through the validation set
    for i in range(len(X_valid)):
        # Predict the current validation point
        y_pred = _predict_model(model, X_valid[i : i + 1, feature_indices])
        y_pred_valid.extend(y_pred)

        # Update the model with the current validation point only after forecast_horizon has passed
        current_time = pd.Timestamp(X_valid[i, 0])

        # Check if enough time has passed since the last update
        if current_time - last_update_time >= forecast_horizon_td:
            if verbose == 2:
                print(
                    f"Updating model at {current_time} (forecast horizon: {forecast_horizon})"
                )
            _partial_fit_model(
                model, X_valid[i : i + 1, feature_indices], y_valid.iloc[i : i + 1]
            )
            last_update_time = current_time

    if verbose == 2:
        print("Validation predictions completed.")

    # Print memory usage after retraining
    if verbose >= 1:
        memory_usage = calculate_memory_usage(model)
        print(f"Model's memory usage: {memory_usage:.6f} MB")

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)

    return y_pred_train, y_pred_valid


def all_valid_splits_seq_fit_predict(
    model_class: Type[BaseEstimator],
    model_params: Dict[str, Any],
    train: List[Tuple[np.ndarray, np.ndarray]],
    valid: List[Tuple[np.ndarray, np.ndarray]],
    incremental: bool,
    data_freq: str,
    forecast_horizon: str,
    features=None,  # Added features parameter
    update_freq="14D",
    spike_prominence=10.0,
    normalize_robust=True,
    start_split: int = 0,
    end_split: Optional[int] = None,
    predict_train: bool = True,
    index_cols=2,
    verbose: int = 1,
    tqdm_on: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Fit and predict for each device in the training and validation splits.

    Args:
        model (sklearn.BaseEstimator): The model to be used for fitting and predicting.
        train (List[Tuple[np.ndarray, np.ndarray]]): The training data splits.
        valid (List[Tuple[np.ndarray, np.ndarray]]): The validation data splits.
        incremental (bool): If True, use incremental learning; otherwise, use batch learning.
        data_freq (str): The frequency of the data (e.g., "1s", "5min", "H", etc.).
        forecast_horizon (str): How far ahead to forecast before updating with actual values.
        features (List[str], optional): List of feature names. If provided, features that aren't available
                                       for the forecast horizon will be filtered.
        update_freq (str): For batch models - frequency at which the model is retrained.
        spike_prominence (float): The prominence parameter for spike detection metrics.
        normalize_robust (bool): If True, use robust normalization, else use regular z-score normalization.
        start_split (int): The starting split for processing.
        end_split (Optional[int]): The ending split for processing.
        predict_train (bool): If True, predict on the training set.
        index_cols (int): Number of columns that are not features.
        verbose (int, optional): an integer indicating the verbosity level:
            - 0: No messages.
            - 1: Print progress messages and memory usage.
            - 2: Print progress messages, memory usage, and detailed steps.
        tqdm_on (bool): If True, use tqdm for progress bars; otherwise, disable them.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            device_train_period_predictions, device_valid_period_predictions,
            device_train_period_metrics, device_valid_period_metrics
    """
    device_train_period_metrics = {}
    device_train_period_predictions = {}
    device_valid_period_metrics = {}
    device_valid_period_predictions = {}
    end_split = len(train) if not end_split else end_split

    if tqdm_on:
        split_iter = tqdm(
            range(start_split, end_split), desc="Processing splits", position=0
        )
    else:
        split_iter = range(start_split, end_split)

    for split in split_iter:
        device_ids = set(train[split][0][:, 1])
        if tqdm_on:
            device_iter = tqdm(
                device_ids,
                desc=f"Processing devices in split {split}",
                position=1,
                leave=False,
            )
        else:
            device_iter = device_ids

        for device_id in device_iter:
            X_train, y_train = get_device_data(train[split], device_id=device_id)
            X_valid, y_valid = get_device_data(valid[split], device_id=device_id)

            if X_valid.shape[0] == 0:
                print(
                    f"[! WARNING !] No validation data for device {device_id} in split {split}. Skipping."
                )
                continue
            
            model = _init_model_with_params(model_class, model_params)

            if incremental:
                y_pred_train, y_pred_valid = seq_fit_predict_incremental_model(
                    model, X_train, y_train, X_valid, y_valid, 
                    forecast_horizon, features, predict_train, index_cols, verbose
                )
            else:
                y_pred_train, y_pred_valid = seq_fit_predict_batch_model(
                    model, X_train, y_train, X_valid, y_valid, 
                    forecast_horizon, update_freq, features, predict_train, index_cols, verbose
                )

            train_period = f"{X_train[0, 0]} - {X_train[-1, 0]}"
            valid_period = f"{X_valid[0, 0]} - {X_valid[-1, 0]}"
            
            device_train_predictions = {
                "period": train_period,
                "y_true": y_train,
                "y_pred": y_pred_train,
            }

            device_valid_predictions = {
                "period": valid_period,
                "y_true": y_valid,
                "y_pred": y_pred_valid,
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
                    model=model,  # Pass model for memory calculation
                    verbose=False,
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
                model=model,  # Pass model for memory calculation
                verbose=verbose,
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
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} during the training period {period['period']}!"
                )
            if period["y_pred"] is not None and period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the training period {period['period']}!"
                )

    for device_id in device_valid_period_predictions:
        for period in device_valid_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} during the validation period {period['period']}!"
                )
            if period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the validation period {period['period']}!"
                )

    return (
        device_train_period_predictions,
        device_valid_period_predictions,
        device_train_period_metrics,
        device_valid_period_metrics,
    )


def test_set_seq_fit_predict(
    model_class: Type[BaseEstimator],
    model_params: Dict[str, Any],
    train: List[Tuple[np.ndarray, np.ndarray]],
    valid: List[Tuple[np.ndarray, np.ndarray]],
    test: List[Tuple[np.ndarray, np.ndarray]],
    incremental: bool,
    data_freq: str,
    forecast_horizon: str,
    features=None,  # Added features parameter
    update_freq="14D",
    spike_prominence=10.0,
    normalize_robust=True,
    predict_train: bool = False,
    index_cols=2,
    verbose: int = 1,
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
        forecast_horizon (str): How far ahead to forecast before updating with actual values.
        features (List[str], optional): List of feature names. If provided, features that aren't available
                                       for the forecast horizon will be filtered.
        update_freq (str): For batch models - frequency at which the model is retrained.
        spike_prominence (float): The prominence parameter for spike detection metrics.
        normalize_robust (bool): If True, use robust normalization, else use regular z-score normalization.
        predict_train (bool): If True, predict on the training set.
        index_cols (int): Number of columns that are not features.
        verbose (int, optional): an integer indicating the verbosity level:
            - 0: No messages.
            - 1: Print progress messages and memory usage.
            - 2: Print progress messages, memory usage, and detailed steps.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
            device_train_period_predictions, device_test_period_predictions,
            device_train_period_metrics, device_test_period_metrics
    """
    device_train_period_metrics = {}
    device_train_period_predictions = {}
    device_test_period_metrics = {}
    device_test_period_predictions = {}

    # Get the last split from training and validation
    last_train_split = train[-1]
    last_valid_split = valid[-1]

    for device_id in tqdm(
        set(last_train_split[0][:, 1]).union(set(last_valid_split[0][:, 1])),
        desc="Processing devices for test predictions",
        position=0,
    ):
        # Extract device data from last training and validation splits
        X_train, y_train = get_device_data(last_train_split, device_id=device_id)
        X_valid, y_valid = get_device_data(last_valid_split, device_id=device_id)

        # Combine training and validation data for model training
        X_combined = (
            np.vstack([X_train, X_valid])
            if len(X_train) > 0 and len(X_valid) > 0
            else (X_train if len(X_train) > 0 else X_valid)
        )
        y_combined = (
            pd.concat([y_train, y_valid])
            if len(y_train) > 0 and len(y_valid) > 0
            else (y_train if len(y_train) > 0 else y_valid)
        )

        # Get test data for this device
        X_test, y_test = get_device_data(
            test, device_id=device_id
        )  # Assuming test has only one split

        # Skip if device is not in test set
        if len(X_test) == 0:
            continue

        model = _init_model_with_params(model_class, model_params)

        # Train model and predict on test data
        if incremental:
            y_pred_train, y_pred_test = seq_fit_predict_incremental_model(
                model,
                X_combined,
                y_combined,
                X_test,
                y_test,
                forecast_horizon,
                features,
                predict_train,
                index_cols,
                verbose,
            )
        else:
            y_pred_train, y_pred_test = seq_fit_predict_batch_model(
                model,
                X_combined,
                y_combined,
                X_test,
                y_test,
                forecast_horizon,
                update_freq,
                features,
                predict_train,
                index_cols,
                verbose,
            )

        # Store periods info
        train_period = f"{X_combined[0, 0]} - {X_combined[-1, 0]}"
        test_period = f"{X_test[0, 0]} - {X_test[-1, 0]}"

        # Store predictions
        device_train_predictions = {
            "period": train_period,
            "y_true": y_combined,
            "y_pred": y_pred_train,
        }

        device_test_predictions = {
            "period": test_period,
            "y_true": y_test,
            "y_pred": y_pred_test,
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
                model=model,  # Pass model for memory calculation
                verbose=False,
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
            model=model,  # Pass model for memory calculation
            verbose=verbose,
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
        if (
            device_train_metrics is not None
        ):  # Only append metrics if they were calculated
            device_train_period_metrics[device_id].append(device_train_metrics)

        if device_id not in device_test_period_metrics:
            device_test_period_metrics[device_id] = []
        device_test_period_metrics[device_id].append(device_test_metrics)

    # Report missing values
    for device_id in device_train_period_predictions:
        for period in device_train_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} during the training period {period['period']}!"
                )
            if period["y_pred"] is not None and period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the training period {period['period']}!"
                )

    for device_id in device_test_period_predictions:
        for period in device_test_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} during the test period {period['period']}!"
                )
            if period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} during the test period {period['period']}!"
                )

    return (
        device_train_period_predictions,
        device_test_period_predictions,
        device_train_period_metrics,
        device_test_period_metrics,
    )
