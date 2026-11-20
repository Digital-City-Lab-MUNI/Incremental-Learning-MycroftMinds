"""Select nontrivial FastODT configurations on validation, then rerun public data.

The unconstrained best trial is retained in the tuning report. The routing-profile
selection requires a split in every validation fit and minimizes validation RMSE.
Test target values are loaded only by the subsequent evaluation/benchmark stage.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .bootstrap import ensure_ilm_on_path, run_fingerprint
from .registry import experiment_seed, load_config, resolve_model

PUBLIC_DATASETS = ("od_aq_1h", "od_friedman_1h", "od_electricity_1h")
RERUN_MODELS = ("FastODT_Original", "BL_Prev", "FastODT", "xFastODT", "xFastODT_C")


def root_diagnostics(model: Any) -> dict:
    """Read current root split candidates without mutating the learner."""
    node = model.root
    best = second = 0.0
    best_feature = best_threshold = None
    candidates = positive = representatives = 0
    for feature, histogram in (node.histograms or {}).items():
        representatives += len(histogram.bin_edges)
        for threshold, left, right in histogram.iter_candidate_stats():
            candidates += 1
            gain = float(node.split_criterion(left, right, node.samples_seen, node.sum_y, node.sum_y_squared))
            positive += int(gain > 0)
            if gain > best:
                second, best = best, gain
                best_feature, best_threshold = int(feature), float(threshold)
            elif second < gain < best:
                second = gain
    epsilon = None
    if node.samples_seen:
        radius = float(node._hoeffding_R())
        epsilon = math.sqrt(radius**2 * math.log(1 / node.delta) / (2 * node.samples_seen))
    return {
        "routing_depth": sum(split is not None for split in model.splits),
        "root_observations": int(node.samples_seen),
        "last_attempt_observations": int(node.grace_period_offset),
        "candidate_count_at_snapshot": candidates,
        "positive_candidate_count_at_snapshot": positive,
        "histogram_representatives": representatives,
        "best_feature_at_snapshot": best_feature,
        "best_threshold_at_snapshot": best_threshold,
        "best_gain_at_snapshot": best,
        "second_gain_at_snapshot": second,
        "gain_gap_at_snapshot": best - second,
        "epsilon_at_snapshot": epsilon,
        "tie_threshold": float(node.tie_threshold),
        "tie_break_available_at_snapshot": epsilon is not None and node.tie_threshold > epsilon,
    }


class DiagnosticFactory:
    """Collect each validation model's final structure and root diagnostics."""

    def __init__(self, model_class: type) -> None:
        self.model_class = model_class
        self.__name__ = model_class.__name__
        self.current = None
        self.models: list[dict] = []

    def __call__(self, **params: Any) -> Any:
        """Release the preceding fit after recording its diagnostic snapshot."""
        self.finish()
        self.current = self.model_class(**params)
        return self.current

    def finish(self) -> None:
        """Capture the last fit after validation completes."""
        if self.current is not None:
            self.models.append(root_diagnostics(self.current))
            self.current = None


def evaluate_candidate(cfg: dict, dataset: str, train: list, valid: list, tau: float) -> dict:
    """Score a threshold over all validation folds; never access the test split."""
    from src.evaluation.metrics import all_time_aggregation_metrics
    from src.evaluation.training import all_valid_splits_seq_fit_predict

    trial_cfg = copy.deepcopy(cfg)
    ds = trial_cfg["datasets"][dataset]
    ds["model_params"]["FastODT"]["tie_threshold"] = tau
    model_class, params, incremental, update_freq = resolve_model(trial_cfg, dataset, "FastODT")
    seed = experiment_seed(trial_cfg)
    random.seed(seed)
    np.random.seed(seed)
    factory = DiagnosticFactory(model_class)
    protocol = trial_cfg["protocol"]
    _, _, _, metrics = all_valid_splits_seq_fit_predict(
        model_class=factory,
        model_params=params,
        train=train,
        valid=valid,
        use_residuals=ds["use_residuals"],
        incremental=incremental,
        data_freq=ds["data_freq"],
        forecast_horizon=protocol["forecast_horizon"],
        features=ds["features"],
        update_freq=update_freq,
        spike_prominence=ds["spike_prominence"],
        normalize_robust=protocol["normalize_robust"],
        predict_train=False,
        verbose=0,
        tqdm_on=False,
    )
    factory.finish()
    aggregated = all_time_aggregation_metrics(metrics, data_freq=ds["data_freq"], verbose=False)
    errors = {}
    for metric in ("rmse", "mae", "mape"):
        values = [float(levels[protocol["agg_level"]][metric]["mean"]) for levels in aggregated.values()]
        errors[f"valid_{metric}"] = float(np.mean(values)) if values else float("nan")
    return {
        "params": params,
        **errors,
        "validation_fits": len(factory.models),
        "split_validation_fits": sum(snapshot["routing_depth"] > 0 for snapshot in factory.models),
        "all_validation_fits_split": bool(factory.models)
        and all(snapshot["routing_depth"] > 0 for snapshot in factory.models),
        "root_diagnostics": factory.models,
    }


def validation_rmse(trial: dict) -> tuple[float, float]:
    """Use validation RMSE, then the smaller threshold as a deterministic tie-break."""
    return trial["valid_rmse"], trial["params"]["tie_threshold"]


def select_trials(trials: list[dict]) -> dict:
    """Retain the unconstrained winner and select the best growing-tree trial."""
    finite = [trial for trial in trials if math.isfinite(trial.get("valid_rmse", float("nan")))]
    if not finite:
        raise ValueError("No finite validation scores; inspect tuning.json")
    best = min(finite, key=validation_rmse)
    growing = [trial for trial in finite if trial["all_validation_fits_split"]]
    if not growing:
        raise ValueError("No candidate split in every validation fit; inspect root diagnostics before rerunning")
    selected = min(growing, key=validation_rmse)
    return {
        "best_unconstrained_params": best["params"],
        "best_unconstrained_valid_rmse": best["valid_rmse"],
        "selected_params": selected["params"],
        "selected_valid_rmse": selected["valid_rmse"],
        "growth_constraint_valid_rmse_cost": selected["valid_rmse"] - best["valid_rmse"],
    }


def tune_dataset(
    cfg: dict,
    dataset: str,
    config_path: str,
    root: Path,
    data_root: Path,
    out_dir: Path,
    thresholds: list[float],
    max_length: int,
    skip_existing: bool,
) -> dict:
    """Checkpoint validation-only tuning for one affected public dataset."""
    from src.data.io import load_dataset

    ds = cfg["datasets"][dataset]
    data_dir = data_root / ds["data_dir"]
    candidates = sorted(set([float(ds["model_params"]["FastODT"]["tie_threshold"]), *thresholds]))
    fingerprint = run_fingerprint(
        config_path,
        root,
        data_dir,
        ds["prefix"],
        {
            "selection": "validation-rmse-with-all-fits-split-v1",
            "thresholds": candidates,
            "seed": experiment_seed(cfg),
            "max_length": max_length,
        },
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "tuning.json"
    if skip_existing and report_path.exists():
        previous = json.loads(report_path.read_text())
        if previous.get("fingerprint") == fingerprint and previous.get("selected_params"):
            print(f"[SKIP] matching public-tree tuning {dataset}")
            return previous
    cap = max_length if max_length > 0 else 10**12
    train, valid, _ = load_dataset(str(data_dir), ds["prefix"], max_length=cap, test=False)
    if not train or not valid:
        raise ValueError(f"Missing train/validation folds for {dataset}")
    report = {
        "dataset": dataset,
        "fingerprint": fingerprint,
        "selection": "min validation RMSE among candidates splitting in every validation fit",
        "test_targets_used_for_selection": False,
        "trials": [],
    }
    for tau in candidates:
        trial = evaluate_candidate(cfg, dataset, train, valid, tau)
        report["trials"].append(trial)
        report_path.write_text(json.dumps(report, indent=2))
        print(
            f"[TUNE] {dataset} tau={tau:g}: valid RMSE={trial['valid_rmse']:.6g}, "
            f"split fits={trial['split_validation_fits']}/{trial['validation_fits']}"
        )
    report.update(select_trials(report["trials"]))
    report_path.write_text(json.dumps(report, indent=2))
    return report


def rerun_config(cfg: dict, selections: dict[str, dict]) -> dict:
    """Preserve original parameters as a labelled control; share selected tree params."""
    generated = copy.deepcopy(cfg)
    generated["public_tree_growth_selection"] = {}
    for dataset, report in selections.items():
        ds = generated["datasets"][dataset]
        ds["model_params"]["FastODT_Original"] = copy.deepcopy(ds["model_params"]["FastODT"])
        if "FastODT_Original" not in ds["models"]:
            ds["models"].append("FastODT_Original")
        ds["model_params"]["FastODT"] = copy.deepcopy(report["selected_params"])
        for name in ("xFastODT", "xFastODT_C"):
            if name not in ds["models"]:
                ds["models"].append(name)
            ds["model_params"][name] = copy.deepcopy(report["selected_params"])
        generated["public_tree_growth_selection"][dataset] = {
            key: report[key]
            for key in ("selected_valid_rmse", "best_unconstrained_valid_rmse", "growth_constraint_valid_rmse_cost")
        }
    return generated


def check_results(results: Path, datasets: list[str]) -> int:
    """Fail if any selected FastODT variant has a root-only final test model."""
    for dataset in datasets:
        for model in ("FastODT", "xFastODT", "xFastODT_C"):
            path = results / "raw" / dataset / f"{model}_benchmark.json"
            if not path.exists():
                print(f"[FAIL] {dataset}/{model}: missing {path}", file=sys.stderr)
                return 1
            benchmark = json.loads(path.read_text())
            structures = benchmark.get("model_structure", [])
            if not structures or any(entry.get("routing_depth", 0) < 1 for entry in structures):
                print(
                    f"[FAIL] {dataset}/{model}: unsplit final test model; inspect tuning diagnostics", file=sys.stderr
                )
                return 1
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse targeted tuning/rerun options."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--datasets", nargs="+", choices=PUBLIC_DATASETS, default=list(PUBLIC_DATASETS))
    parser.add_argument("--tie-thresholds", nargs="+", type=float, default=[0.01, 0.02, 0.05, 0.1])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-dir", default="results/rerun/public_trees")
    parser.add_argument("--max-length", type=int, default=-1)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--model-jobs", type=int, default=1)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--tune-only", action="store_true")
    parser.add_argument("--check-results", action="store_true")
    args = parser.parse_args(argv)
    if any(not math.isfinite(value) or not 0 < value <= 1 for value in args.tie_thresholds):
        parser.error("--tie-thresholds must be finite values in (0, 1]")
    return args


def main(argv: list[str] | None = None) -> int:
    """Tune on validation, write the selected config, and rerun controls plus tree variants."""
    from . import run_matrix

    args = parse_args(argv)
    results = Path(args.results_dir).resolve()
    if args.check_results:
        return check_results(results, args.datasets)
    root = ensure_ilm_on_path()
    cfg = load_config(args.config)
    data_root = Path(args.data_root or os.environ.get("FASTODT_DATA_ROOT") or root / "data").expanduser().resolve()
    selections = {}
    for dataset in args.datasets:
        selections[dataset] = tune_dataset(
            cfg,
            dataset,
            args.config,
            root,
            data_root,
            results / "tuning" / dataset,
            args.tie_thresholds,
            args.max_length,
            args.skip_existing,
        )
    selected_path = results / "selected_config.yaml"
    selected_path.write_text(yaml.safe_dump(rerun_config(cfg, selections), sort_keys=False))
    print(f"[DONE] selected validation configurations: {selected_path}")
    if args.tune_only:
        return 0
    command = [
        "--config",
        str(selected_path),
        "--datasets",
        *args.datasets,
        "--models",
        *RERUN_MODELS,
        "--data-root",
        str(data_root),
        "--results-dir",
        str(results),
        "--max-length",
        str(args.max_length),
        "--jobs",
        str(args.jobs),
        "--model-jobs",
        str(args.model_jobs),
    ]
    if args.skip_existing:
        command.append("--skip-existing")
    status = run_matrix.main(command)
    if status:
        return status
    return check_results(results, args.datasets)


if __name__ == "__main__":
    sys.exit(main())
