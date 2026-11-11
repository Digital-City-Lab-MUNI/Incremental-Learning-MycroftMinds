"""Check protocol equivalence, fresh workers, and outdated benchmark rejection."""

import json
import os
import tempfile
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pandas as pd
import yaml

from experiments import prepare_results, run_benchmark
from experiments.bootstrap import git_revision
from experiments.registry import load_config, resolve_model
from src.evaluation.metrics import all_time_aggregation_metrics
from src.evaluation.training import test_set_seq_fit_predict


def make_split(start: int, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Two hourly devices with different target scales and occasional zeros."""
    features, targets = [], []
    for device in range(2):
        hours = np.arange(start, start + count)
        values = 4 + device * 3 + np.sin(hours / 5)
        values[hours % 37 == 0] = 0
        lag = 4 + device * 3 + np.sin((hours - 1) / 5)
        rows = np.empty((count, 4), dtype=object)
        rows[:, 0] = pd.to_datetime(1577836800 + hours * 3600, unit="s", utc=True)
        rows[:, 1] = device
        rows[:, 2] = hours % 24
        rows[:, 3] = lag
        features.append(rows)
        targets.append(values)
    return np.concatenate(features), np.concatenate(targets)


def direct_test(
    model_class: Any, params: dict, train: list, valid: list, test: tuple, incremental: bool = False
) -> tuple:
    """Run the uninstrumented test entry point used by run_evaluation."""
    np.random.seed(42)
    return test_set_seq_fit_predict(
        model_class=model_class,
        model_params=params,
        train=train,
        valid=valid,
        test=test,
        use_residuals=True,
        incremental=incremental,
        data_freq="H",
        forecast_horizon="1h",
        features=["time", "device_id", "hour", "power_lag_1h"],
        update_freq="1D",
        spike_prominence=1,
        normalize_robust=True,
        predict_train=False,
        verbose=0,
    )


class ExperimentBenchmarkTests(unittest.TestCase):
    """Exercise actual test pipelines and subprocess benchmark orchestration."""

    def setUp(self) -> None:
        """Write tiny train/validation/test archives in a temporary directory."""
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.train = [make_split(0, 96)]
        self.valid = [make_split(96, 48)]
        self.test = make_split(144, 96)
        data_dir = self.directory / "data"
        data_dir.mkdir()
        for split, pairs in (("train", self.train), ("valid", self.valid)):
            np.savez(
                data_dir / f"fixture_{split}.npz",
                X_0=pairs[0][0],
                y_0=pairs[0][1],
                num_splits=1,
                data_type="split_tuples",
            )
        np.savez(data_dir / "fixture_test.npz", X=self.test[0], y=self.test[1], data_type="single_tuple")
        config = {
            "matched_models": {"xFastODT": "FastODT"},
            "protocol": {
                "forecast_horizon": "1h",
                "agg_level": "hourly",
                "update_freq_batch": "1D",
                "normalize_robust": True,
                "predict_train_test": False,
            },
            "datasets": {
                "fixture": {
                    "data_dir": "data",
                    "prefix": "fixture",
                    "data_freq": "H",
                    "use_residuals": True,
                    "spike_prominence": 1,
                    "features": ["time", "device_id", "hour", "power_lag_1h"],
                    "models": ["GBDT", "RF", "FastODT", "HoeffdingTree"],
                    "model_params": {
                        "GBDT": {"n_estimators": 3, "max_depth": 2, "random_state": 42},
                        "RF": {"n_estimators": 3, "max_depth": 2, "random_state": 42, "n_jobs": 1},
                        "FastODT": {"depth": 2, "grace_period": 10, "random_state": 42},
                        "HoeffdingTree": {"max_depth": 2, "grace_period": 10, "leaf_prediction": "mean"},
                    },
                }
            },
        }
        self.config_path = self.directory / "config.yaml"
        self.config_path.write_text(yaml.safe_dump(config))
        self.arguments = [
            "--config",
            str(self.config_path),
            "--dataset",
            "fixture",
            "--data-root",
            str(self.directory),
            "--results-dir",
            str(self.directory / "raw"),
            "--predict-samples",
            "2",
            "--models",
            "GBDT",
            "RF",
            "FastODT",
            "xFastODT",
            "HoeffdingTree",
        ]

    def test_instrumentation_preserves_reconstructed_full_test_metrics(self) -> None:
        """Instrumentation preserves predictions and includes periodic batch refits."""
        cfg = load_config(self.config_path)
        model_class, params, _, _ = resolve_model(cfg, "fixture", "GBDT")
        expected = direct_test(model_class, params, self.train, self.valid, self.test)
        measured = run_benchmark.MeasuredModelFactory(model_class, 2)
        actual = direct_test(measured, params, self.train, self.valid, self.test)
        measured.finish_device()
        for device in expected[1]:
            np.testing.assert_allclose(expected[1][device][0]["y_pred"], actual[1][device][0]["y_pred"])
        self.assertEqual(len(measured.devices), 2)
        self.assertGreater(sum(device["fit_calls"] for device in measured.devices), 2)
        self.assertEqual(sum(device["samples"] for device in measured.devices), 4)

    def test_fresh_workers_score_all_rows_and_reject_legacy_resume(self) -> None:
        """Fresh processes score all test rows; resume reuses only matching outputs."""
        self.assertEqual(run_benchmark.main(self.arguments), 0)
        results = []
        cfg = load_config(self.config_path)
        for name in ("GBDT", "RF", "FastODT", "xFastODT", "HoeffdingTree"):
            path = self.directory / "raw" / "fixture" / f"{name}_benchmark.json"
            result = json.loads(path.read_text())
            results.append(result)
            self.assertNotEqual(result["worker_pid"], os.getpid())
            self.assertEqual(result["test_samples"], 192)
            self.assertEqual(result["pred_batch_samples"], 4)
            self.assertEqual(result["n_devices"], 2)
            self.assertGreater(result["peak_rss_mb"], 0)
            model_class, params, incremental, _ = resolve_model(cfg, "fixture", name)
            _, _, _, metrics = direct_test(model_class, params, self.train, self.valid, self.test, incremental)
            aggregated = all_time_aggregation_metrics(metrics, data_freq="H", verbose=False)
            for metric in ("rmse", "mae", "mape"):
                self.assertAlmostEqual(
                    result[f"test_{metric}"], run_benchmark.mean_device_metric(aggregated, "hourly", metric)
                )
        self.assertNotEqual(results[0]["worker_pid"], results[1]["worker_pid"])
        revision = git_revision(Path(run_benchmark.__file__).resolve().parents[1])
        with (
            patch("experiments.run_benchmark.subprocess.run") as launch,
            patch("experiments.bootstrap.git_revision", return_value=revision),
        ):
            self.assertEqual(run_benchmark.main(self.arguments + ["--skip-existing"]), 0)
            launch.assert_not_called()
        legacy = self.directory / "raw" / "fixture" / "GBDT_benchmark.json"
        legacy.write_text(json.dumps({"test_mape": 1.0}))
        with (
            patch("experiments.run_benchmark.subprocess.run") as launch,
            patch("experiments.bootstrap.git_revision", return_value=revision),
        ):
            launch.return_value.returncode = 0
            run_benchmark.main(self.arguments + ["--skip-existing"])
            self.assertEqual(launch.call_count, 1)

    def test_xfastodt_uses_matched_fastodt_parameters(self) -> None:
        """Both implementations receive equal but independently copied parameters."""
        cfg = load_config(self.config_path)
        dataset = cfg["datasets"]["fixture"]
        self.assertIn("xFastODT", dataset["models"])
        self.assertEqual(dataset["model_params"]["FastODT"], dataset["model_params"]["xFastODT"])
        dataset["model_params"]["xFastODT"]["depth"] = 99
        self.assertEqual(dataset["model_params"]["FastODT"]["depth"], 2)

    def test_tables_exclude_legacy_benchmarks(self) -> None:
        """An old accuracy protocol cannot silently reappear in resource tables."""
        dataset_dir = self.directory / "fixture"
        dataset_dir.mkdir()
        (dataset_dir / "FastODT_benchmark.json").write_text(json.dumps({"test_mape": 1.0}))
        (dataset_dir / "xFastODT_benchmark.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "test_mape": 0.2,
                    "environment": {"ilm_git_revision": "test-revision"},
                }
            )
        )
        _, benchmarks, _ = prepare_results.collect_dataset(self.directory, "fixture", "hourly")
        self.assertEqual(list(benchmarks["model"]), ["xFastODT"])
        self.assertEqual(benchmarks.iloc[0]["ilm_git_revision"], "test-revision")


if __name__ == "__main__":
    unittest.main()
