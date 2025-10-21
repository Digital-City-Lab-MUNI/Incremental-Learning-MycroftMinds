import numpy as np
from unittest import TestCase
from src.data.io import load_dataset

from src.data.io import load_dataset
from src.utils.memory import calculate_memory_usage
from src.utils.utils import get_device_data
from src.evaluation.metrics import calculate_metrics, all_time_aggregation_metrics
from src.evaluation.visualization import plot_part_predictions, plot_hourly_predictions
from src.evaluation.training import all_valid_splits_seq_fit_predict, test_set_seq_fit_predict

from gym_pybullet_swarms.simulator.Simulator import Simulator
from gym_pybullet_swarms.drones.baseDrone import Drone

from river.forest import ARFRegressor
from river.tree import HoeffdingTreeRegressor
from src.evaluation.baseline import PreviousValueRegressor, SeasonalMeanRegressor
from src.trees import fastodt
from src.trees import xfastodt
from src.bagging import ARFRegressorFODT
from src.boosting import IncubationBoost
from src.ensemble_methods import elasticGBDT

data_dir = '../../data/first_batch'

train, valid, test = load_dataset(
    data_dir=data_dir, 
    prefix='dev_4_1h_wfv', 
    use_numpy=True,
    max_length=100
)

ALL_FEATURES = ['time', 'device_id', 'hour', 'weekday', 'month', 
                'L1_act_power_lag_1h', 'L1_act_power_lag_24h', 'L1_app_power_lag_1h', 'L1_app_power_lag_24h', 
                'L1_current_lag_1h', 'L1_current_lag_24h', 'L1_react_power_lag_1h', 'L1_react_power_lag_24h', 
                'L1_voltage_lag_1h', 'L1_voltage_lag_24h', 
                'L2_act_power_lag_1h', 'L2_act_power_lag_24h', 'L2_app_power_lag_1h', 'L2_app_power_lag_24h', 
                'L2_current_lag_1h', 'L2_current_lag_24h', 'L2_react_power_lag_1h', 'L2_react_power_lag_24h', 
                'L2_voltage_lag_1h', 'L2_voltage_lag_24h', 
                'L3_act_power_lag_1h', 'L3_act_power_lag_24h', 'L3_app_power_lag_1h', 'L3_app_power_lag_24h', 
                'L3_current_lag_1h', 'L3_current_lag_24h', 'L3_react_power_lag_1h', 'L3_react_power_lag_24h', 
                'L3_voltage_lag_1h', 'L3_voltage_lag_24h', 
                'grid_freq_lag_1h', 'grid_freq_lag_24h', 
                'powerfactor1_lag_1h', 'powerfactor1_lag_24h', 
                'powerfactor2_lag_1h', 'powerfactor2_lag_24h', 
                'powerfactor3_lag_1h', 'powerfactor3_lag_24h']

class TestARF(TestCase):

    def test_ARF(self):
        DATA_FREQUENCY = 'H' # '15min'
        USE_RESIDUAL = True  # If True, model the residuals (1-step differencing) instead of the original target.
        FORECAST_HORIZON = '1h'
        SPIKE_PROMINENCE = 10

        update_freq = "1h"  # for incremental models put the same as the data frequency to recreate real world scenario
        arf_hf = ARFRegressor
        arf_hf_params = dict(n_models=10, 
                            max_features=0.8, 
                            aggregation_method='median', 
                            grace_period=10, 
                            max_depth=10, 
                            delta=0.5, 
                            tau=0.01, 
                            seed=42)
        arf_hf_all_train_preds, arf_hf_all_valid_preds, \
        arf_hf_all_train_metrics, arf_hf_all_valid_metrics = all_valid_splits_seq_fit_predict(
            model_class=arf_hf,
            model_params=arf_hf_params,
            train=train,
            valid=valid,
            use_residuals=USE_RESIDUAL,
            incremental=True,
            data_freq=DATA_FREQUENCY,
            forecast_horizon=FORECAST_HORIZON,
            features=ALL_FEATURES,
            update_freq=update_freq,
            spike_prominence=SPIKE_PROMINENCE,
            start_split=0, 
            #end_split=4,  # Use only the third and fourth splits
            verbose=True
        )
        print("\n\nTrain period metrics:")
        arf_hf_all_time_train_metrics = all_time_aggregation_metrics(arf_hf_all_train_metrics, data_freq=DATA_FREQUENCY, verbose=True)
        print("\n\Valid period metrics:")
        arf_hf_all_time_valid_metrics = all_time_aggregation_metrics(arf_hf_all_valid_metrics, data_freq=DATA_FREQUENCY, verbose=True)

        # Define model classes and params for test-set evaluation
        models_classes = [arf_hf]
        models_params = [
            bl_prev_params,
            bl_mean_params,
            gbdt_params,
            rf_params,
            hoef_params,
            arf_hf_params,
            fodt_params,
            xfodt_params,
            arf_params,
            incb_params,
        ]

        model_test_results = {}
        for model_class, params in zip(models_classes, models_params):
            if model_class in [gbdt, rf, bl_mean]:
                incremental = False
                update_freq = "14D"
                print(f"Model: {model_class.__name__} - update_freq: {update_freq}")
            else:
                incremental = True
                update_freq = "1h"
                print(f"Model: {model_class.__name__} - update_freq: {update_freq}")

            _, test_preds, \
            _, test_metrics = test_set_seq_fit_predict(
                model_class=model_class,
                model_params=params,
                train=train,
                valid=valid,
                test=test,
                use_residuals=USE_RESIDUAL,
                incremental=incremental,
                data_freq=DATA_FREQUENCY,
                forecast_horizon=FORECAST_HORIZON,
                features=ALL_FEATURES,
                update_freq=update_freq,
                spike_prominence=SPIKE_PROMINENCE,
                normalize_robust=True,
                predict_train=False,
                verbose=True
            )
            model_test_results[model_class.__name__] = {
                "test_preds": test_preds,
                "test_metrics": test_metrics
            }