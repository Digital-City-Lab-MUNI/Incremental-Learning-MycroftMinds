"""Sequential (walk-forward) training / evaluation utilities.

This module provides helpers to perform incremental or periodic retraining
over time-ordered splits (walk-forward validation) as well as final test set
evaluation. Two update paradigms are supported:

* Batch models (standard scikit-learn estimators) retrained at a specified
    ``update_freq``.
* Incremental models (supporting ``partial_fit`` or River's ``learn_one``)
    updated sample-by-sample after a ``forecast_horizon`` of look-ahead.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, List, Dict, Any, Type
from tqdm.auto import tqdm
from sklearn.base import BaseEstimator
import time

# Import from our project's modules
from src.evaluation.metrics import calculate_metrics
from src.utils.memory import calculate_memory_usage
from src.utils.utils import as_timestamp, device_label, ensure_2d_array, get_device_data, time_index
from src.data.feature_engineering import filter_features_for_forecast_horizon


def _init_model_with_params(model_class: Type[BaseEstimator], model_params: Dict[str, Any]) -> BaseEstimator:
    """Instantiate a model class with parameters.

    Parameters
    ----------
    model_class : Type[BaseEstimator]
        Estimator class (not an instance) to be created.
    model_params : dict
        Keyword arguments forwarded to the class constructor.

    Returns
    -------
    BaseEstimator
        Instantiated estimator.

    Raises
    ------
    ValueError
        If model initialization fails.
    """
    try:
        # Attempt to initialize the model with the provided parameters
        return model_class(**model_params)

    except Exception:
        print(f"[ERROR] Failed to initialize model '{model_class.__name__}' with parameters {model_params}.")
        raise


def _partial_fit_model(model: BaseEstimator, X: np.ndarray, Y: pd.Series) -> None:
    """Update model incrementally using ``partial_fit`` or River's ``learn_one``.

    Parameters
    ----------
    model : BaseEstimator
        Estimator supporting either ``partial_fit`` or (River) ``learn_one``.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.
    Y : pandas.Series of shape (n_samples,)
        Target values aligned with ``X`` rows.
    """
    X = ensure_2d_array(X)
    if hasattr(model, "partial_fit"):
        model.partial_fit(X, Y)
    else:
        # River-style models take one dict per sample; build the dicts from the rows
        # directly (pd.DataFrame(X).to_dict("records") per call is far slower, and the
        # pipeline calls this once per single row)
        cols = list(range(X.shape[1]))
        for row, y in zip(X.tolist(), Y, strict=True):
            model.learn_one(dict(zip(cols, row)), y)


def _predict_model(model: BaseEstimator, X: np.ndarray) -> np.ndarray:
    """Predict using standard ``predict`` or River's ``predict_one``.

    Parameters
    ----------
    model : BaseEstimator
        Estimator supporting ``predict`` or (River) ``predict_one``.
    X : ndarray of shape (n_samples, n_features)
        Feature matrix.

    Returns
    -------
    ndarray of shape (n_samples,)
        Model predictions in original target scale.
    """
    X = ensure_2d_array(X)
    if hasattr(model, "predict"):
        return model.predict(X)
    else:
        # River-style models take one dict per sample (see _partial_fit_model)
        cols = list(range(X.shape[1]))
        return np.array([model.predict_one(dict(zip(cols, row))) for row in X.tolist()])


def seq_fit_predict_batch_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    forecast_horizon: str,
    update_freq: str = "14 days",
    features: Optional[List[str]] = None,
    predict_train: bool = True,
    index_cols: int = 2,
    verbose: int = 0,
) -> Tuple[Optional[pd.Series], pd.Series]:
    """Sequential walk-forward prediction for batch models.

    The estimator is fully retrained every ``update_freq`` using all data
    observed up to (but not including) the prediction horizon boundary. Optionally
    performs feature filtering to exclude lag features violating the
    ``forecast_horizon`` availability.

    Parameters
    ----------
    model : BaseEstimator
        Batch learning model with ``fit`` and ``predict``.
    X_train : ndarray of shape (n_train_samples, n_features)
        Initial training feature matrix.
    y_train : pandas.Series of shape (n_train_samples,)
        Initial training target series.
    X_valid : ndarray of shape (n_valid_samples, n_features)
        Validation feature matrix to predict sequentially.
    y_valid : pandas.Series of shape (n_valid_samples,)
        Validation target series to predict sequentially.
    forecast_horizon : str
        Time horizon string (pandas offset) specifying the minimum prediction
        horizon for feature filtering and model update delays.
    update_freq : str, default='14 days'
        Time interval between full model retrains (pandas Timedelta parsable string).
    features : list of str, optional
        Full feature name list of length n_features including index columns.
        Excludes the lag features violating the forecast horizon. If provided,
        must include ALL features available in the dataset.
    predict_train : bool, default=True
        Whether to generate in-sample predictions on the training data.
    index_cols : int, default=2
        Number of leading columns in feature matrices that contain non-feature
        metadata (e.g., timestamp, device_id).
    verbose : int, default=0
        Verbosity level for output (0=silent, 1=progress info, 2=detailed).

    Returns
    -------
    y_pred_train : pandas.Series or None
        Training predictions if ``predict_train`` is True, otherwise None.
    y_pred_valid : pandas.Series
        Validation predictions.
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
        valid_indices = [index_cols + i for i, feat in enumerate(features[index_cols:]) if feat in valid_features]

        if verbose >= 2:
            print(f"Filtering features for forecast horizon {forecast_horizon}")
            print(f"Original feature count: {len(features) - index_cols}")
            print(f"Filtered feature count: {len(valid_indices)}")
            print(f"Excluded lag features: {', '.join(excluded_features) if excluded_features else 'None'}")

    # Use filtered indices if available, otherwise use all features
    feature_indices = valid_indices if valid_indices else slice(index_cols, None)

    update_freq = pd.Timedelta(update_freq)
    update_freq_s = update_freq.total_seconds()
    valid_times = X_valid[:, 0]
    numeric_time = valid_times.dtype != object  # numeric layout: epoch seconds

    # Grow the training set chunk-wise and materialize it with a single vstack per
    # retrain, instead of re-copying the whole matrix on every chunk (quadratic).
    x_chunks = [X_train]
    y_parts = [y_train]

    # Iterate through the validation set in chunks based on update frequency
    current_chunk_start = 0
    while current_chunk_start < len(X_valid):
        # Determine the end of the current chunk using timestamp manipulation
        if numeric_time:
            chunk_end_time = valid_times[current_chunk_start] + update_freq_s
        else:
            chunk_end_time = pd.Timestamp(valid_times[current_chunk_start]) + update_freq
        chunk_end = np.searchsorted(valid_times, chunk_end_time, side="right")

        X_fit = x_chunks[0] if len(x_chunks) == 1 else np.vstack(x_chunks)
        y_fit = y_parts[0] if len(y_parts) == 1 else pd.concat(y_parts)

        # Retrain the model with the updated training set
        if verbose >= 1:
            start = time.time()
            print(f"\nRetraining model on data from {as_timestamp(X_fit[0, 0])} to {as_timestamp(X_fit[-1, 0])}")
            model.fit(X_fit[:, feature_indices], y_fit)
            print(f"\nTime: {time.time() - start:.2f} seconds")
        else:
            model.fit(X_fit[:, feature_indices], y_fit)

        # Predict for the initial training set if required
        if predict_train and y_train.shape[0] == y_fit.shape[0]:
            y_pred_train = model.predict(X_fit[:, feature_indices])
            y_pred_train = pd.Series(y_pred_train, index=y_train.index)
            if verbose == 2:
                print("Got predictions on training set")

        # Predict for the current chunk
        if verbose >= 1:
            print(
                f"Predicting for chunk from {as_timestamp(X_valid[current_chunk_start, 0])} "
                f"to {as_timestamp(X_valid[chunk_end - 1, 0])}"
            )
            print()
            print("-" * 25)
        y_pred_valid_chunk = model.predict(X_valid[current_chunk_start:chunk_end, feature_indices])
        y_pred_valid.extend(y_pred_valid_chunk)

        # Add the current chunk of validation data to the training set
        x_chunks.append(X_valid[current_chunk_start:chunk_end])
        y_parts.append(
            pd.Series(
                y_valid.iloc[current_chunk_start:chunk_end],
                index=y_valid.index[current_chunk_start:chunk_end],
            )
        )

        # Move to the next chunk
        current_chunk_start = chunk_end

    if verbose == 2:
        print("Validation predictions completed.")

    # Print memory usage after retraining (expensive pympler walk; debug level only,
    # the benchmark measures model size once per model via calculate_memory_usage)
    if verbose >= 2:
        memory_usage = calculate_memory_usage(model)
        print(f"Model's memory usage: {memory_usage:.6f} MB")

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)

    return y_pred_train, y_pred_valid


def seq_fit_predict_incremental_model(
    model: BaseEstimator,
    X_train: np.ndarray,
    y_train: pd.Series,
    X_valid: np.ndarray,
    y_valid: pd.Series,
    forecast_horizon: str,
    features: Optional[List[str]] = None,
    predict_train: bool = True,
    index_cols: int = 2,
    verbose: int = 0,
) -> Tuple[Optional[pd.Series], pd.Series]:
    """Sequential prediction for incremental models with horizon delay.

    After initial fitting, the model predicts each validation sample then
    updates only once the elapsed time since the last update exceeds the
    specified ``forecast_horizon``. Optional feature filtering mirrors the
    batch variant.

    Parameters
    ----------
    model : BaseEstimator
        Incremental estimator supporting ``partial_fit`` / ``learn_one`` and
        prediction.
    X_train : ndarray of shape (n_train_samples, n_features)
        Initial training feature matrix.
    y_train : pandas.Series of shape (n_train_samples,)
        Initial training target series.
    X_valid : ndarray of shape (n_valid_samples, n_features)
        Validation feature matrix to predict sequentially.
    y_valid : pandas.Series of shape (n_valid_samples,)
        Validation target series to predict sequentially.
    forecast_horizon : str
        Time horizon string (pandas offset) specifying the minimum prediction
        horizon for feature filtering and model update delays.
    features : list of str, optional
        Full feature name list of length n_features including index columns.
        Excludes the lag features violating the forecast horizon. If provided,
        must include ALL features available in the dataset.
    predict_train : bool, default=True
        Whether to generate in-sample predictions on the training data.
    index_cols : int, default=2
        Number of leading columns in feature matrices that contain non-feature
        metadata (e.g., timestamp, device_id).
    verbose : int, default=0
        Verbosity level for output (0=silent, 1=progress info, 2=detailed).

    Returns
    -------
    y_pred_train : pandas.Series or None
        Training predictions if ``predict_train`` is True, otherwise None.
    y_pred_valid : pandas.Series
        Validation predictions.
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
        valid_indices = [index_cols + i for i, feat in enumerate(features[index_cols:]) if feat in valid_features]

        if verbose >= 1:
            print(f"Filtering features for forecast horizon {forecast_horizon}")
            print(f"Original feature count: {len(features) - index_cols}")
            print(f"Filtered feature count: {len(valid_indices)}")
            print(f"Excluded lag features: {', '.join(excluded_features) if excluded_features else 'None'}")

    # Use filtered indices if available, otherwise use all features
    feature_indices = valid_indices if valid_indices else slice(index_cols, None)

    forecast_horizon_td = pd.Timedelta(forecast_horizon)

    # Slice the feature block once up front; slicing per row inside the loop below
    # copies a matrix row on every sample.
    X_train_f = X_train[:, feature_indices]
    X_valid_f = X_valid[:, feature_indices]
    valid_times = time_index(X_valid[:, 0])

    # Fit the model on the initial training data
    if verbose >= 1:
        print(f"Initial training on data from {as_timestamp(X_train[0, 0])} to {as_timestamp(X_train[-1, 0])}")

    _partial_fit_model(model, X_train_f, y_train)

    # Predict on the training set if required
    if predict_train:
        y_pred_train = _predict_model(model, X_train_f)
        y_pred_train = pd.Series(y_pred_train, index=y_train.index)

        if verbose >= 2:
            print("Train predictions completed.")

    # Initialize last_update_time with the last timestamp from training data
    last_update_time = as_timestamp(X_train[-1, 0])

    # Iterate through the validation set
    for i in range(len(X_valid)):
        # Predict the current validation point
        y_pred = _predict_model(model, X_valid_f[i : i + 1])
        y_pred_valid.extend(y_pred)

        # Update the model with the current validation point only after forecast_horizon has passed
        current_time = valid_times[i]

        # Check if enough time has passed since the last update
        if current_time - last_update_time >= forecast_horizon_td:
            if verbose == 2:
                print(f"Updating model at {current_time} (forecast horizon: {forecast_horizon})")
            _partial_fit_model(model, X_valid_f[i : i + 1], y_valid.iloc[i : i + 1])
            last_update_time = current_time

    if verbose == 2:
        print("Validation predictions completed.")

    # Print memory usage after retraining (expensive pympler walk; debug level only,
    # the benchmark measures model size once per model via calculate_memory_usage)
    if verbose >= 2:
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
    use_residuals: bool,
    incremental: bool,
    data_freq: str,
    forecast_horizon: str,
    features: Optional[List[str]] = None,
    update_freq: str = "14D",
    spike_prominence: float = 10.0,
    normalize_robust: bool = True,
    start_split: int = 0,
    end_split: Optional[int] = None,
    predict_train: bool = True,
    index_cols: int = 2,
    verbose: int = 1,
    tqdm_on: bool = True,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Walk-forward evaluation across validation splits per device.

    For each split and device the model is (re)initialised, residuals are
    modeled via 1-step differencing, and sequential predictions are produced
    using either batch retraining or incremental updates. Predictions and
    metrics are stored per device & period.

    Parameters
    ----------
    model_class : Type[BaseEstimator]
        Estimator class to instantiate per device per split.
    model_params : dict
        Constructor parameters for the estimator.
    train : list of tuple
        List of (X, y) training data tuples representing sequential folds, where
        X is ndarray of shape (n_samples, n_features) and y is pandas.Series
        of shape (n_samples,).
    valid : list of tuple
        List of (X, y) validation data tuples representing sequential folds, where
        X is ndarray of shape (n_samples, n_features) and y is pandas.Series
        of shape (n_samples,).
    use_residuals : bool
        If True, model the residuals (1-step differencing) instead of the original target.
    incremental : bool
        If True use incremental mode, else batch mode.
    data_freq : str
        Original data frequency (passed to metric aggregation).
    forecast_horizon : str
        Time horizon string (pandas offset) specifying the minimum prediction
        horizon for feature filtering and model update delays.
    features : list of str, optional
        Full feature name list of length n_features including index columns.
        Excludes the lag features violating the forecast horizon. If provided,
        must include ALL features available in the dataset.
    update_freq : str, default='14D'
        Time interval between full model retrains (pandas Timedelta parsable string).
    spike_prominence, normalize_robust : see metrics module
        Parameters forwarded to spike/drop evaluation.
    start_split, end_split : int, optional
        Subset range of splits (``end_split`` exclusive) to process.
    predict_train : bool, default=True
        Whether to generate in-sample predictions on the training data.
    index_cols : int, default=2
        Number of leading columns in feature matrices that contain non-feature
        metadata (e.g., timestamp, device_id).
    verbose : int, default=1
        Verbosity level for output (0=silent, 1=progress info, 2=detailed).
    tqdm_on : bool, default=True
        If False, disable progress bars.

    Returns
    -------
    device_train_period_predictions : dict
        Training predictions per device and period.
    device_valid_period_predictions : dict
        Validation predictions per device and period.
    device_train_period_metrics : dict
        Training metrics per device and period.
    device_valid_period_metrics : dict
        Validation metrics per device and period.
    """
    device_train_period_metrics = {}
    device_train_period_predictions = {}
    device_valid_period_metrics = {}
    device_valid_period_predictions = {}
    end_split = len(train) if not end_split else end_split

    if tqdm_on:
        split_iter = tqdm(range(start_split, end_split), desc="Processing splits", position=0)
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
            # numeric layout stores device ids as codes; report the original label
            device_key = device_label(device_id)

            if X_valid.shape[0] == 0:
                print(f"[! WARNING !] No validation data for device {device_key} in split {split}. Skipping.")
                continue

            if X_train.shape[1] != len(features):
                raise ValueError(
                    f"Feature count mismatch for device {device_id} in split {split}: "
                    f"X has {X_train.shape[1]} features, but features list has {len(features)} entries."
                )

            if X_valid.shape[1] != X_train.shape[1]:
                raise ValueError(
                    f"Feature count mismatch for device {device_id} in split {split}: "
                    f"train has {X_train.shape[1]} features, valid has {X_valid.shape[1]} features."
                )

            if use_residuals:
                # Model the residuals, not the original target
                y_train_shifted = y_train.shift(1)  # use 1-step lag for differencing
                d_train = y_train - y_train_shifted
                d_train = d_train[1:]  # remove the first row to match the shifted target
                X_train = X_train[1:]
                y_train = y_train[1:]

                # Prepend last training target value to the validation Series to align differencing
                y_valid = pd.concat([y_train.iloc[[-1]], y_valid])
                y_valid_shifted = y_valid.shift(1)  # use 1-step lag for differencing
                d_valid = y_valid - y_valid_shifted
                d_valid = d_valid[1:]  # remove the first row to match the shifted target
                y_valid = y_valid[1:]
                # X_valid remains unchanged as it is already aligned with y_valid

            model = _init_model_with_params(model_class, model_params)

            if incremental:
                y_pred_train, y_pred_valid = seq_fit_predict_incremental_model(
                    model,
                    X_train,
                    d_train if use_residuals else y_train,
                    X_valid,
                    d_valid if use_residuals else y_valid,
                    forecast_horizon,
                    features,
                    predict_train,
                    index_cols,
                    verbose,
                )
            else:
                y_pred_train, y_pred_valid = seq_fit_predict_batch_model(
                    model,
                    X_train,
                    d_train if use_residuals else y_train,
                    X_valid,
                    d_valid if use_residuals else y_valid,
                    forecast_horizon,
                    update_freq,
                    features,
                    predict_train,
                    index_cols,
                    verbose,
                )

            train_period = f"{as_timestamp(X_train[0, 0])} - {as_timestamp(X_train[-1, 0])}"
            valid_period = f"{as_timestamp(X_valid[0, 0])} - {as_timestamp(X_valid[-1, 0])}"

            if use_residuals:
                # Adjust predictions to match the original y_train and y_valid
                y_pred_train = y_pred_train + y_train_shifted[1:] if predict_train else None
                y_pred_valid = y_pred_valid + y_valid_shifted[1:]

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
                    device_key,
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
                device_key,
                valid_period,
                data_freq=data_freq,
                spike_prominence=spike_prominence,
                normalize_robust=normalize_robust,
                model=model,  # Pass model for memory calculation
                verbose=verbose,
            )

            if device_key not in device_train_period_predictions:
                device_train_period_predictions[device_key] = []
            device_train_period_predictions[device_key].append(device_train_predictions)

            if device_key not in device_valid_period_predictions:
                device_valid_period_predictions[device_key] = []
            device_valid_period_predictions[device_key].append(device_valid_predictions)

            if device_key not in device_train_period_metrics:
                device_train_period_metrics[device_key] = []
            device_train_period_metrics[device_key].append(device_train_metrics)

            if device_key not in device_valid_period_metrics:
                device_valid_period_metrics[device_key] = []
            device_valid_period_metrics[device_key].append(device_valid_metrics)

    # report missing values
    for device_id in device_train_period_predictions:
        for period in device_train_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} "
                    f"during the training period {period['period']}!"
                )
            if period["y_pred"] is not None and period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} "
                    f"during the training period {period['period']}!"
                )

    for device_id in device_valid_period_predictions:
        for period in device_valid_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} "
                    f"during the validation period {period['period']}!"
                )
            if period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} "
                    f"during the validation period {period['period']}!"
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
    use_residuals: bool,
    incremental: bool,
    data_freq: str,
    forecast_horizon: str,
    features: Optional[List[str]] = None,
    update_freq: str = "14D",
    spike_prominence: float = 10.0,
    normalize_robust: bool = True,
    predict_train: bool = False,
    index_cols: int = 2,
    verbose: int = 1,
) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """Train on final combined (train+valid) data and evaluate on test set.

    The last training and validation splits are concatenated per device (where
    available) to form the fitting window, then sequential predictions are
    produced on the test split using either batch or incremental strategy.

    Parameters
    ----------
    model_class : Type[BaseEstimator]
        Estimator class to instantiate per device per split.
    model_params : dict
        Constructor parameters for the estimator.
    train : list of tuple
        List of (X, y) training data tuples representing sequential folds, where
        X is ndarray of shape (n_samples, n_features) and y is pandas.Series
        of shape (n_samples,).
    valid : list of tuple
        List of (X, y) validation data tuples representing sequential folds, where
        X is ndarray of shape (n_samples, n_features) and y is pandas.Series
        of shape (n_samples,).
    test : list of tuple
        List of (X, y) test data tuples, where X is ndarray of shape (n_samples,
        n_features) and y is pandas.Series of shape (n_samples,).
    use_residuals : bool
        If True, model the residuals (1-step differencing) instead of the original target.
    incremental : bool
        If True use incremental mode, else batch mode.
    data_freq : str
        Original data frequency (passed to metric aggregation).
    forecast_horizon : str
        Time horizon string (pandas offset) specifying the minimum prediction
        horizon for feature filtering and model update delays.
    features : list of str, optional
        Full feature name list of length n_features including index columns.
        Excludes the lag features violating the forecast horizon. If provided,
        must include ALL features available in the dataset.
    update_freq : str, default='14D'
        Time interval between full model retrains (pandas Timedelta parsable string).
    spike_prominence : float, default=10.0
        Minimum prominence of spikes to detect.
    normalize_robust : bool, default=True
        Whether to apply robust normalization to the ``y_true`` and ``y_pred``
        for spike detection.
    predict_train : bool, default=False
        Whether to generate in-sample predictions on the training data.
    index_cols : int, default=2
        Number of leading columns in feature matrices that contain non-feature
        metadata (e.g., timestamp, device_id).
    verbose : int, default=1
        Verbosity level for output (0=silent, 1=progress info, 2=detailed).

    Returns
    -------
    device_train_period_predictions : dict
        Training predictions per device and period.
    device_test_period_predictions : dict
        Test predictions per device and period.
    device_train_period_metrics : dict
        Training metrics per device and period.
    device_test_period_metrics : dict
        Test metrics per device and period.
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
        # numeric layout stores device ids as codes; report the original label
        device_key = device_label(device_id)
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
        X_test, y_test = get_device_data(test, device_id=device_id)  # Assuming test has only one split

        # Validate feature dimensions
        if X_combined.shape[1] != X_test.shape[1]:
            raise ValueError(
                f"Feature count mismatch for device {device_id}: "
                f"train+valid has {X_combined.shape[1]} features, test has {X_test.shape[1]} features."
            )
        if X_combined.shape[1] != len(features):
            raise ValueError(
                f"Feature count mismatch for device {device_id}: "
                f"X has {X_combined.shape[1]} features, but features list has {len(features)} entries."
            )

        # Skip if device is not in test set
        if len(X_test) == 0:
            continue

        if use_residuals:
            # Model the residuals, not the original target
            y_combined_shifted = y_combined.shift(1)  # use 1-step lag for differencing
            d_combined = y_combined - y_combined_shifted
            d_combined = d_combined[1:]  # remove the first row to match the shifted target
            X_combined = X_combined[1:]
            y_combined = y_combined[1:]

            # Prepend last combined target value to the test Series to align differencing
            y_test = pd.concat([y_combined.iloc[[-1]], y_test])
            y_test_shifted = y_test.shift(1)  # use 1-step lag for differencing
            d_test = y_test - y_test_shifted
            d_test = d_test[1:]  # remove the first row to match the shifted target
            y_test = y_test[1:]
            # X_test remains unchanged as it is already aligned with y_test

        model = _init_model_with_params(model_class, model_params)

        # Train model and predict on test data
        if incremental:
            y_pred_combined, y_pred_test = seq_fit_predict_incremental_model(
                model,
                X_combined,
                d_combined if use_residuals else y_combined,
                X_test,
                d_test if use_residuals else y_test,
                forecast_horizon,
                features,
                predict_train,
                index_cols,
                verbose,
            )
        else:
            y_pred_combined, y_pred_test = seq_fit_predict_batch_model(
                model,
                X_combined,
                d_combined if use_residuals else y_combined,
                X_test,
                d_test if use_residuals else y_test,
                forecast_horizon,
                update_freq,
                features,
                predict_train,
                index_cols,
                verbose,
            )

        # Store periods info
        train_period = f"{as_timestamp(X_combined[0, 0])} - {as_timestamp(X_combined[-1, 0])}"
        test_period = f"{as_timestamp(X_test[0, 0])} - {as_timestamp(X_test[-1, 0])}"

        if use_residuals:
            # Adjust predictions to match the original y_combined and y_test
            y_pred_combined = y_pred_combined + y_combined_shifted[1:] if predict_train else None
            y_pred_test = y_pred_test + y_test_shifted[1:]

        # Store predictions
        device_train_predictions = {
            "period": train_period,
            "y_true": y_combined,
            "y_pred": y_pred_combined,
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
                y_pred_combined,
                device_key,
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
            device_key,
            test_period,
            data_freq=data_freq,
            spike_prominence=spike_prominence,
            normalize_robust=normalize_robust,
            model=model,  # Pass model for memory calculation
            verbose=verbose,
        )

        # Store results
        if device_key not in device_train_period_predictions:
            device_train_period_predictions[device_key] = []
        device_train_period_predictions[device_key].append(device_train_predictions)

        if device_key not in device_test_period_predictions:
            device_test_period_predictions[device_key] = []
        device_test_period_predictions[device_key].append(device_test_predictions)

        if device_key not in device_train_period_metrics:
            device_train_period_metrics[device_key] = []
        if device_train_metrics is not None:  # Only append metrics if they were calculated
            device_train_period_metrics[device_key].append(device_train_metrics)

        if device_key not in device_test_period_metrics:
            device_test_period_metrics[device_key] = []
        device_test_period_metrics[device_key].append(device_test_metrics)

    # Report missing values
    for device_id in device_train_period_predictions:
        for period in device_train_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} "
                    f"during the training period {period['period']}!"
                )
            if period["y_pred"] is not None and period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} "
                    f"during the training period {period['period']}!"
                )

    for device_id in device_test_period_predictions:
        for period in device_test_period_predictions[device_id]:
            if period["y_true"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_true for device {device_id} "
                    f"during the test period {period['period']}!"
                )
            if period["y_pred"].isnull().any():
                print(
                    f"[! WARNING !] There are missing values in the y_pred for device {device_id} "
                    f"during the test period {period['period']}!"
                )

    return (
        device_train_period_predictions,
        device_test_period_predictions,
        device_train_period_metrics,
        device_test_period_metrics,
    )
