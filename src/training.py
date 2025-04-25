import pandas as pd
import numpy as np
from typing import Tuple, Optional
from sklearn.base import BaseEstimator
import time

# Import from our project's modules
from src.memory import calculate_memory_usage

def seq_fit_predict_batch_model(model: BaseEstimator, 
                              X_train: np.array, 
                              y_train: pd.Series, 
                              X_valid: np.array, 
                              y_valid: pd.Series,
                              update_freq: str = "14 days",
                              predict_train: bool = True,
                              index_cols=2) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using regular batch-learning models by updating 
    the training data with each new point. You may also specify the time-frequency at which to update the model.

    Args:
        model (BaseEstimator): The regular batch learning model to use for predictions. Must have `.fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        update_freq (str, optional): Frequency at which to update the model (e.g., "28 days", "14 days", "1 hour"). Default is "14 days".
        predict_train (bool, optional): Whether to predict on the initial training set. Default is True.
        index_cols (int, optional): Number of columns in `X_train` and `X_valid` that are not features (e.g., timestamps, device IDs). Default is 2.

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
        start = time.time()
        print(f"\nRetraining model on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}")
        model.fit(X_train_seq[:, index_cols:], y_train_seq)
        print(f"\nTime: {time.time() - start:.2f} seconds")
        # Print memory usage after retraining
        memory_usage = calculate_memory_usage(model)
        print(f"Memory usage after retraining: {memory_usage:.4f} MB")

        # Predict for the initial training set if required
        if predict_train and y_train.shape[0] == y_train_seq.shape[0]:
            y_pred_train = model.predict(X_train_seq[:, index_cols:])
            y_pred_train = pd.Series(y_pred_train, index=y_train.index)
            print("Got predictions on training set")

        # Predict for the current chunk
        print(f"Predicting for chunk from {X_valid[current_chunk_start, 0]} to {X_valid[chunk_end-1, 0]}")
        print()
        print("-" * 25)
        y_pred__valid_chunk = model.predict(X_valid[current_chunk_start:chunk_end, index_cols:])
        y_pred_valid.extend(y_pred__valid_chunk)

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
                                    update_freq: str = "14 days",
                                    predict_train: bool = True,
                                    index_cols=2) -> Tuple[Optional[pd.Series], pd.Series]:
    """
    Perform sequential predictions for time series data using incremental learning models by updating 
    the model with each new point. You may also specify the time-frequency at which to update the model.

    Args:
        model (BaseEstimator): The incremental learning model to use for predictions. Must have `.partial_fit` and `.predict` methods.
        X_train (np.ndarray): Training features, where the first `index_cols` columns are not features.
        y_train (pd.Series): Training labels.
        X_valid (np.ndarray): Validation features, where the first `index_cols` columns are not features.
        y_valid (pd.Series): Validation labels.
        update_freq (str, optional): Frequency at which to update the model (e.g., "28 days", "14 days", "1 hour"). Default is "14 days".
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
    print(f"Initial training on data from {X_train_seq[0, 0]} to {X_train_seq[-1, 0]}")
    model.partial_fit(X_train_seq[:, index_cols:], y_train_seq)

    # Predict on the training set if required
    if predict_train:
        y_pred_train = model.predict(X_train_seq[:, index_cols:])
        y_pred_train = pd.Series(y_pred_train, index=y_train.index)
        print("Train predictions completed.")

    # Iterate through the validation set
    for i in range(len(X_valid)):
        # Predict the current validation point
        y_pred = model.predict(X_valid[i:i+1, index_cols:])[0]
        y_pred_valid.append(y_pred)

        # Update the model with the current validation point
        if i + 1 < len(X_valid):  # Ensure we don't go out of bounds
            current_time = X_valid[i, 0]  # Assuming the first column in X_valid is the timestamp
            next_time = X_valid[i + 1, 0]
            if pd.Timestamp(next_time) - pd.Timestamp(current_time) >= update_freq:
                #print(f"Updating model with data up to {next_time}")
                model.partial_fit(X_valid[i:i+1, index_cols:], y_valid.iloc[i:i+1])

    # Convert predictions list to pandas Series
    y_pred_valid = pd.Series(y_pred_valid, index=y_valid.index)

    return y_pred_train, y_pred_valid
