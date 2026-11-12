"""Benchmark the full evaluation test protocol in a fresh process per model.

Uses last train + last validation split, delayed updates, periodic batch refits,
residual reconstruction and the same hourly/device aggregation as run_evaluation.
Only inference probes are capped; fitting and accuracy use every device/test row.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import collect_environment, ensure_ilm_on_path, peak_rss_mb, run_fingerprint
from .registry import experiment_seed, load_config, resolve_dataset, resolve_model

SCHEMA_VERSION = 2
THREAD_KEYS = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse coordinator and isolated-worker arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/experiments.yaml")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--models", nargs="*", default=None)
    parser.add_argument("--ilm-path", default=None)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--results-dir", default="results/rerun/raw")
    parser.add_argument("--max-length", type=int, default=-1, help="Cap validation folds, not test rows")
    parser.add_argument("--predict-samples", type=int, default=2000, help="Inference probes per device")
    parser.add_argument("--skip-existing", action="store_true", help="Reuse only matching v2 benchmark fingerprints")
    parser.add_argument(
        "--jobs", type=int, default=1, help="Accepted for matrix compatibility; benchmarks run serially"
    )
    parser.add_argument("--worker-model", default=None, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.predict_samples < 1:
        parser.error("--predict-samples must be at least 1")
    return args


def benchmark_fingerprint(args: argparse.Namespace, root: Path, data_dir: Path, prefix: str, seed: int) -> str:
    """Hash code/config, dataset file identities, and benchmark settings for resuming."""
    return run_fingerprint(
        args.config,
        root,
        data_dir,
        prefix,
        {
            "schema_version": SCHEMA_VERSION,
            "max_length": args.max_length,
            "predict_samples": args.predict_samples,
            "threads": 1,
            "seed": seed,
        },
    )


class MeasuredEstimator:
    """Time fitting calls without changing pipeline inputs or predictions."""

    def __init__(self, model: Any, predict_samples: int) -> None:
        self.model = model
        self.predict_samples = predict_samples
        self.train_time_s = 0.0
        self.fit_calls = 0
        self.probe_rows = []
        self.probe_count = 0

    def fit(self, features: np.ndarray, targets: Any) -> MeasuredEstimator:
        """Time a complete batch fit, including periodic refits."""
        started = time.perf_counter()
        try:
            self.model.fit(features, targets)
        finally:
            self.train_time_s += time.perf_counter() - started
            self.fit_calls += 1
        return self

    def partial_fit(self, features: np.ndarray, targets: Any) -> MeasuredEstimator:
        """Time an incremental update through the pipeline's sklearn/River adapter."""
        from src.evaluation.training import _partial_fit_model

        started = time.perf_counter()
        try:
            _partial_fit_model(self.model, features, targets)
        finally:
            self.train_time_s += time.perf_counter() - started
            self.fit_calls += 1
        return self

    def predict(self, features: np.ndarray) -> np.ndarray:
        """Return unchanged predictions and retain bounded inference-probe inputs."""
        from src.evaluation.training import _predict_model

        predictions = _predict_model(self.model, features)
        remaining = self.predict_samples - self.probe_count
        if remaining > 0:
            rows = np.asarray(features)[:remaining].copy()
            self.probe_rows.append(rows)
            self.probe_count += len(rows)
        return predictions


class MeasuredModelFactory:
    """Finalize one device model before constructing the next device's model."""

    def __init__(self, model_class: Any, predict_samples: int) -> None:
        self.model_class = model_class
        self.__name__ = model_class.__name__
        self.predict_samples = predict_samples
        self.current = None
        self.devices = []

    def __call__(self, **params: Any) -> MeasuredEstimator:
        """Construct a device estimator after measuring the previous estimator."""
        self.finish_device()
        self.current = MeasuredEstimator(self.model_class(**params), self.predict_samples)
        return self.current

    def finish_device(self) -> None:
        """Measure fitted-model latency and size, then release the device model."""
        from src.evaluation.training import _predict_model
        from src.utils.memory import model_memory_breakdown

        measured = self.current
        if measured is None:
            return
        self.current = None
        if not measured.probe_rows:
            return
        features = np.concatenate(measured.probe_rows)
        _predict_model(measured.model, features[:1])  # Warm up before timed probes.
        single_us = []
        for row in features:
            started = time.perf_counter()
            _predict_model(measured.model, row.reshape(1, -1))
            single_us.append((time.perf_counter() - started) * 1e6)
        started = time.perf_counter()
        _predict_model(measured.model, features)
        batch_s = time.perf_counter() - started
        memory = model_memory_breakdown(measured.model)
        containers = getattr(measured.model, "depths_containers", None)
        node_count = getattr(measured.model, "n_nodes_", None)
        if containers is not None:
            node_count = sum(node is not None for level in containers for node in level.values())
        elif hasattr(measured.model, "root"):
            pending = [measured.model.root]
            node_count = 0
            while pending:
                node = pending.pop()
                node_count += 1
                if node.left is not None:
                    pending.extend([node.left, node.right])
        routing_depth = getattr(measured.model, "current_depth", None)
        if routing_depth is None and hasattr(measured.model, "splits"):
            routing_depth = sum(split is not None for split in measured.model.splits)
        self.devices.append(
            {
                "train_time_s": measured.train_time_s,
                "fit_calls": measured.fit_calls,
                "single_us": single_us,
                "batch_s": batch_s,
                "samples": len(features),
                "model_size_mb": memory["total_bytes"] / 1024**2,
                "python_model_size_mb": memory["python_bytes"] / 1024**2,
                "native_model_size_mb": memory["native_bytes"] / 1024**2,
                "native_peak_model_size_mb": memory["native_peak_bytes"] / 1024**2,
                "node_count": node_count,
                "routing_depth": routing_depth,
                "native_build_info": getattr(measured.model, "native_build_info", None),
            }
        )


def mean_device_metric(aggregated: dict, level: str, metric: str) -> float:
    """Use the same mean of device means as prepare_results."""
    values = [float(levels[level][metric]["mean"]) for levels in aggregated.values()]
    return float(np.mean(values))


def run_worker(args: argparse.Namespace, root: Path, cfg: dict, ds: dict, data_dir: Path) -> dict:
    """Load and benchmark exactly one model in this fresh interpreter."""
    from src.data.io import load_dataset
    from src.evaluation.metrics import all_time_aggregation_metrics
    from src.evaluation.training import test_set_seq_fit_predict

    seed = experiment_seed(cfg)
    np.random.seed(seed)
    random.seed(seed)
    model_class, params, incremental, update_freq = resolve_model(cfg, args.dataset, args.worker_model)
    # n_jobs is a resource setting, not a learned-model hyperparameter.
    params = dict(params)
    if "n_jobs" in params:
        params["n_jobs"] = 1
    protocol = cfg["protocol"]
    max_length = args.max_length if args.max_length > 0 else 10**12
    train, valid, test = load_dataset(str(data_dir), ds["prefix"], max_length=max_length)
    baseline_rss = peak_rss_mb()
    factory = MeasuredModelFactory(model_class, args.predict_samples)
    started = time.perf_counter()
    _, predictions, _, metrics = test_set_seq_fit_predict(
        model_class=factory,
        model_params=params,
        train=train,
        valid=valid,
        test=test,
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
    )
    factory.finish_device()
    pipeline_s = time.perf_counter() - started
    aggregated = all_time_aggregation_metrics(metrics, data_freq=ds["data_freq"], verbose=False)
    if not factory.devices or not aggregated:
        raise ValueError(f"No evaluated devices for {args.dataset}/{args.worker_model}")
    latency = np.concatenate([device["single_us"] for device in factory.devices])
    batch_s = sum(device["batch_s"] for device in factory.devices)
    batch_n = sum(device["samples"] for device in factory.devices)
    largest_model = factory.devices[0]
    for device in factory.devices[1:]:
        if device["model_size_mb"] > largest_model["model_size_mb"]:
            largest_model = device
    return {
        "schema_version": SCHEMA_VERSION,
        "worker_pid": os.getpid(),
        "seed": seed,
        "fingerprint": benchmark_fingerprint(args, root, data_dir, ds["prefix"], seed),
        "dataset": args.dataset,
        "model": args.worker_model,
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "params": params,
        "incremental": incremental,
        "update_freq": update_freq,
        "protocol": {**protocol, "use_residuals": ds["use_residuals"], "data_freq": ds["data_freq"]},
        "accuracy_scope": "full test; last train+valid; mean of per-device aggregated metrics",
        "n_devices": len(aggregated),
        "device_ids": [str(device) for device in aggregated],
        "test_samples": sum(len(period["y_true"]) for periods in predictions.values() for period in periods),
        "total_train_time_s": sum(device["train_time_s"] for device in factory.devices),
        "fit_calls": sum(device["fit_calls"] for device in factory.devices),
        "pipeline_time_s": pipeline_s,
        "pred_single_us_mean": float(np.mean(latency)),
        "pred_single_us_p50": float(np.percentile(latency, 50)),
        "pred_single_us_p99": float(np.percentile(latency, 99)),
        "pred_batch_s": batch_s,
        "pred_batch_samples": batch_n,
        "pred_batch_samples_per_s": batch_n / batch_s,
        "baseline_peak_rss_mb": baseline_rss,
        "peak_rss_mb": peak_rss_mb(),
        "memory_scope": (
            "fresh process per dataset/model; includes imports, all loaded splits, evaluation, probes, pympler"
        ),
        "model_size_mb": largest_model["model_size_mb"],
        "python_model_size_mb": largest_model["python_model_size_mb"],
        "native_model_size_mb": largest_model["native_model_size_mb"],
        "native_peak_model_size_mb": max(device["native_peak_model_size_mb"] for device in factory.devices),
        "native_build_info": factory.devices[0]["native_build_info"],
        "model_structure": [
            {"node_count": device["node_count"], "routing_depth": device["routing_depth"]} for device in factory.devices
        ],
        "model_size_scope": "maximum final model size across devices; not sum or peak during fitting",
        "test_rmse": mean_device_metric(aggregated, protocol["agg_level"], "rmse"),
        "test_mae": mean_device_metric(aggregated, protocol["agg_level"], "mae"),
        "test_mape": mean_device_metric(aggregated, protocol["agg_level"], "mape"),
        "environment": collect_environment(root),
    }


def main(argv: list[str] | None = None) -> int:
    """Run fresh, serial model processes and reject outdated resume artifacts."""
    args = parse_args(argv)
    root = ensure_ilm_on_path(args.ilm_path)
    cfg = load_config(args.config)
    ds = resolve_dataset(cfg, args.dataset)
    data_root = Path(args.data_root or os.environ.get("FASTODT_DATA_ROOT") or root / "data").resolve()
    data_dir = data_root / ds["data_dir"]
    missing = [
        str(data_dir / f"{ds['prefix']}_{split}.npz")
        for split in ("train", "valid", "test")
        if not (data_dir / f"{ds['prefix']}_{split}.npz").exists()
    ]
    if missing:
        print(f"[FAIL] missing dataset files: {missing}", file=sys.stderr)
        return 1
    out_dir = Path(args.results_dir).resolve() / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.worker_model:
        result = run_worker(args, root, cfg, ds, data_dir)
        path = out_dir / f"{args.worker_model}_benchmark.json"
        temporary = path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(result, indent=1, default=str), encoding="utf-8")
        temporary.replace(path)
        print(f"[DONE] {args.dataset}/{args.worker_model}: wrote {path}")
        return 0
    fingerprint = benchmark_fingerprint(args, root, data_dir, ds["prefix"], experiment_seed(cfg))
    failed = False
    for name in args.models or ds["models"]:
        path = out_dir / f"{name}_benchmark.json"
        if args.skip_existing and path.exists():
            try:
                previous = json.loads(path.read_text())
                if previous.get("schema_version") == SCHEMA_VERSION and previous.get("fingerprint") == fingerprint:
                    print(f"[SKIP] matching benchmark {args.dataset}/{name}")
                    continue
            except (ValueError, OSError):
                pass
        command = [
            sys.executable,
            "-m",
            "experiments.run_benchmark",
            "--config",
            str(Path(args.config).resolve()),
            "--dataset",
            args.dataset,
            "--data-root",
            str(data_root),
            "--results-dir",
            str(out_dir.parent),
            "--max-length",
            str(args.max_length),
            "--predict-samples",
            str(args.predict_samples),
            "--worker-model",
            name,
            "--ilm-path",
            str(root),
        ]
        environment = dict(os.environ)
        environment.update({key: "1" for key in THREAD_KEYS})
        environment["PYTHONHASHSEED"] = str(experiment_seed(cfg))
        completed = subprocess.run(command, cwd=root, env=environment, check=False)
        failed = failed or completed.returncode != 0
    return int(failed)


if __name__ == "__main__":
    sys.exit(main())
