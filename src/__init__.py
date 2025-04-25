# This file makes the src directory a Python package

# Core dataloader functionality
from src.data.dataloader import get_train_test_sets

# I/O functions
from src.data.io import (
    save_dataset, 
    load_dataset, 
    save_data_numpy,
    load_data_numpy,
    save_data_pickle,
    load_data_pickle
)

# Memory functions
from src.utils.memory import (
    calculate_memory_usage, 
    print_memory_stats,
    optimize_memory_usage
)

# Preprocessing functions
from src.data.preprocessing import (
    preprocess_df,
    concat_dataframes_by_time,
    get_complete_time_series,
    create_supervised_data_from_series,
    clean_data
)

# Feature engineering functions
from src.data.feature_engineering import (
    add_lag_features, 
    add_cyclical_time_features
)

# Metrics and evaluation functions
from src.evaluation.metrics import (
    calculate_metrics,
    all_time_aggregation_metrics
)

# Visualization functions
from src.evaluation.visualization import (
    plot_predictions,
    plot_hourly_predictions
)

# Training functions
from src.evaluation.training import (
    seq_fit_predict_batch_model,
    seq_fit_predict_incremental_model
)

# Utility functions
from src.utils.utils import aggregate_values
