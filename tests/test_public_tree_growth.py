"""Validation-only selection of nontrivial public-data tree configurations."""

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import yaml

from experiments.registry import load_config, resolve_model
from experiments.run_public_trees import evaluate_candidate, main, rerun_config, root_diagnostics, select_trials
from src.data.io import load_dataset
from src.trees.fastodt import FastODT
from tests.test_experiment_benchmarks import make_split


def fixture_config() -> dict:
    """Configure a short noisy two-device stream without production datasets."""
    return {
        "matched_models": {"xFastODT": "FastODT", "xFastODT_C": "FastODT"},
        "protocol": {
            "seed": 42,
            "forecast_horizon": "1h",
            "agg_level": "hourly",
            "update_freq_batch": "14D",
            "normalize_robust": True,
            "predict_train_test": False,
        },
        "datasets": {
            "od_aq_1h": {
                "data_dir": "data",
                "prefix": "fixture",
                "data_freq": "H",
                "use_residuals": True,
                "spike_prominence": 1,
                "features": ["time", "device_id", "hour", "power_lag_1h"],
                "models": ["FastODT", "BL_Prev"],
                "model_params": {
                    "FastODT": {
                        "depth": 2,
                        "grace_period": 0,
                        "split_confidence": 0.001,
                        "tie_threshold": 0.01,
                        "max_bins": 4,
                    },
                    "BL_Prev": {"is_residual": True},
                },
            }
        },
    }


def write_fixture(directory: Path) -> Path:
    """Persist the synthetic folds in the production archive layout."""
    data_dir = directory / "data"
    data_dir.mkdir()
    for split, start, count in (("train", 0, 96), ("valid", 96, 96)):
        features, targets = make_split(start, count)
        np.savez(data_dir / f"fixture_{split}.npz", X_0=features, y_0=targets, num_splits=1, data_type="split_tuples")
    features, targets = make_split(192, 96)
    np.savez(data_dir / "fixture_test.npz", X=features, y=targets, data_type="single_tuple")
    source = directory / "config.yaml"
    source.write_text(yaml.safe_dump(fixture_config()))
    return source


class PublicTreeGrowthTests(unittest.TestCase):
    """Tree-growth constraints are explicit and do not select on test errors."""

    def test_selection_retains_unconstrained_winner_and_reports_cost(self) -> None:
        """A more accurate unsplit model stays visible rather than silently disappearing."""
        trials = [
            {"valid_rmse": 0.8, "params": {"tie_threshold": 0.01}, "all_validation_fits_split": False},
            {"valid_rmse": 1.2, "params": {"tie_threshold": 0.05}, "all_validation_fits_split": True},
            {"valid_rmse": 1.1, "params": {"tie_threshold": 0.1}, "all_validation_fits_split": True},
        ]
        result = select_trials(trials)
        self.assertEqual(result["best_unconstrained_params"]["tie_threshold"], 0.01)
        self.assertEqual(result["selected_params"]["tie_threshold"], 0.1)
        self.assertAlmostEqual(result["growth_constraint_valid_rmse_cost"], 0.3)
        with self.assertRaisesRegex(ValueError, "No candidate split"):
            select_trials(trials[:1])

    def test_root_diagnostics_are_read_only_and_explain_degenerate_targets(self) -> None:
        """A constant target has no positive candidate even when the tie gate is open."""
        model = FastODT(depth=2, grace_period=0, split_confidence=1, tie_threshold=0.1, random_state=42)
        features = np.arange(40.0).reshape(-1, 1)
        model.partial_fit(features, np.ones(40))
        before = copy.deepcopy(model.root.__dict__)
        diagnostics = root_diagnostics(model)
        self.assertEqual(diagnostics["routing_depth"], 0)
        self.assertTrue(diagnostics["tie_break_available_at_snapshot"])
        self.assertEqual(diagnostics["positive_candidate_count_at_snapshot"], 0)
        self.assertEqual(model.root.samples_seen, before["samples_seen"])
        self.assertEqual(model.root.grace_period_offset, before["grace_period_offset"])
        self.assertEqual(model.root.rng.bit_generator.state, before["rng"].bit_generator.state)

    def test_validation_candidate_collects_structure_without_mutating_config(self) -> None:
        """A permissive trial grows real trees while source parameters remain intact."""
        cfg = fixture_config()
        original = copy.deepcopy(cfg)
        trial = evaluate_candidate(cfg, "od_aq_1h", [make_split(0, 96)], [make_split(96, 96)], 1.0)
        self.assertTrue(trial["all_validation_fits_split"])
        self.assertEqual(trial["validation_fits"], 2)
        self.assertTrue(np.isfinite(trial["valid_rmse"]))
        self.assertTrue(all(entry["routing_depth"] >= 1 for entry in trial["root_diagnostics"]))
        self.assertEqual(cfg, original)

    def test_tune_only_does_not_load_test_targets_and_preserves_original_control(self) -> None:
        """The generated config shares selected parameters; selection reads train/valid only."""
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write_fixture(directory)
            results = directory / "results"
            with patch("src.data.io.load_dataset", wraps=load_dataset) as load:
                status = main(
                    [
                        "--config",
                        str(source),
                        "--datasets",
                        "od_aq_1h",
                        "--data-root",
                        str(directory),
                        "--results-dir",
                        str(results),
                        "--tie-thresholds",
                        "1",
                        "--tune-only",
                    ]
                )
            self.assertEqual(status, 0)
            load.assert_called_once()
            self.assertIs(load.call_args.kwargs["test"], False)
            report = json.loads((results / "tuning/od_aq_1h/tuning.json").read_text())
            self.assertFalse(report["test_targets_used_for_selection"])
            generated = load_config(results / "selected_config.yaml")
            params = generated["datasets"]["od_aq_1h"]["model_params"]
            self.assertEqual(params["FastODT_Original"]["tie_threshold"], 0.01)
            self.assertEqual(params["FastODT"], params["xFastODT"])
            self.assertEqual(params["FastODT"], params["xFastODT_C"])
            self.assertEqual(params["FastODT"], report["selected_params"])
            _, original_params, _, _ = resolve_model(generated, "od_aq_1h", "FastODT_Original")
            self.assertEqual(original_params["tie_threshold"], 0.01)

    def test_full_target_writes_growing_models_and_controls(self) -> None:
        """Exercise tuning, evaluation, isolated benchmarks, tables and the growth check."""
        root = Path(__file__).resolve().parents[1]
        if not (root / "src/trees/_native/libxfastodt.so").exists():
            self.skipTest("run make native to verify the complete public-tree rerun")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            source = write_fixture(directory)
            results = directory / "results"
            status = main(
                [
                    "--config",
                    str(source),
                    "--datasets",
                    "od_aq_1h",
                    "--data-root",
                    str(directory),
                    "--results-dir",
                    str(results),
                    "--tie-thresholds",
                    "1",
                ]
            )
            self.assertEqual(status, 0)
            self.assertTrue((results / "tables/all_benchmark.csv").exists())
            for name in ("FastODT_Original", "BL_Prev", "FastODT", "xFastODT", "xFastODT_C"):
                path = results / "raw/od_aq_1h" / f"{name}_benchmark.json"
                benchmark = json.loads(path.read_text())
                self.assertEqual(benchmark["seed"], 42)
                if name in ("FastODT", "xFastODT", "xFastODT_C"):
                    self.assertTrue(all(entry["routing_depth"] > 0 for entry in benchmark["model_structure"]))

    def test_rerun_config_preserves_unaffected_datasets(self) -> None:
        """Public-data selection must not overwrite parameters for other datasets."""
        cfg = fixture_config()
        cfg["datasets"]["untouched"] = copy.deepcopy(cfg["datasets"]["od_aq_1h"])
        selected = dict(cfg["datasets"]["od_aq_1h"]["model_params"]["FastODT"], tie_threshold=0.05)
        result = rerun_config(
            cfg,
            {
                "od_aq_1h": {
                    "selected_params": selected,
                    "selected_valid_rmse": 1.1,
                    "best_unconstrained_valid_rmse": 1.0,
                    "growth_constraint_valid_rmse_cost": 0.1,
                }
            },
        )
        self.assertEqual(result["datasets"]["untouched"], cfg["datasets"]["untouched"])
        self.assertEqual(cfg["datasets"]["od_aq_1h"]["model_params"]["FastODT"]["tie_threshold"], 0.01)


if __name__ == "__main__":
    unittest.main()
