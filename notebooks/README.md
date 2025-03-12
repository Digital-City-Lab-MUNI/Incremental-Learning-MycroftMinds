Explaining meter_data_processing.ipynb 

Data Preprocessing
Input Data:
The code assumes that you have a list of complete, preprocessed time series DataFrames (named processed_time_series), where each DataFrame is already sorted by time.

Feature Selection & Cleaning:
Non-informative columns such as time, device_id, and serial_number are excluded from the input features. Rows containing missing values are dropped to ensure each sample is complete.

Training/Test Split:
The first few series (defined by n_series_train) are used for full training. The last series is held out for incremental prediction updates. Only a small seed (defined by test_training_ratio) from the test series is used initially to warm-start the model.

Sliding Window Data Preparation
Sliding Window Technique:
The function create_supervised_data_from_series converts a time series DataFrame into supervised learning samples:

Input (X): For each time step t (starting at window_size), it flattens the values of the specified input features from times t - window_size to t - 1.
Target (y): The value of the target feature at time t is used as the prediction target.
Why This Works:
This method ensures that each prediction is based solely on past observations, maintaining temporal causality and preventing data leakage. The window_size parameter determines how many past observations are used, effectively controlling the size of each input sample (if there are f features, each sample is of size window_size × f).

