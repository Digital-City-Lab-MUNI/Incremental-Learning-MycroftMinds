"""Computational benchmark for one dataset x a set of models (paper.todo.md checklist).

For each model and each of up to ``--max-devices`` devices this measures:

* ``total_train_time_s`` — wall time of the pipeline's sequential fit+predict run
  over the test stream; for batch models this includes every periodic refit
  (``update_freq_batch``), for incremental models every online update;
* ``pred_single_us_*`` — per-sample prediction latency (one prediction call per
  sample, as the pipeline does), mean / p50 / p99 in microseconds;
* ``pred_batch_s`` — one batched prediction call over ``--predict-samples``
  samples (throughput);
* ``peak_rss_mb`` — process peak RSS (``ru_maxrss``), the consistent peak-memory
  measurement used for all models;
* ``model_size_mb`` — in-memory model size via ``calculate_memory_usage`` (pympler);
* ``test_rmse / test_mae / test_mape`` — accuracy of the same run (original scale).

Results are written to ``<results-dir>/<dataset>/<Model>_benchmark.json`` and are
merged into the summary tables by ``prepare_results.py``.

Example (from this repo's root)::

    python -m experiments.run_benchmark --dataset sb_ind_1h \
        --models FastODT xFastODT IncubationBoost --results-dir results/rerun/raw
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .bootstrap import collect_environment, ensure_ilm_on_path, peak_rss_mb, run_over_models
from .registry import load_config, resolve_dataset, resolve_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="*", default=None, help="Default: all models of the dataset")
    ap.add_argument("--ilm-path", default=None)
    ap.add_argument("--data-root", default=None)
    ap.add_argument("--results-dir", default="results/rerun/raw")
    ap.add_argument("--max-length", type=int, default=-1, help="Cap samples per split when loading")
    ap.add_argument("--max-devices", type=int, default=5, help="Benchmark at most this many devices")
    ap.add_argument("--max-train", type=int, default=20000, help="Cap training rows per device")
    ap.add_argument("--predict-samples", type=int, default=2000, help="Rows used for latency measurements")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run this many models in parallel worker processes (fork; loaded data is shared "
        "copy-on-write). Default 1 = sequential",
    )
    return ap.parse_args(argv)


def _bench_worker(name: str, ctx: dict) -> str:
    """Benchmark one model, write ``<Model>_benchmark.json``, return the ``[DONE]`` log line."""
    result = benchmark_model(
        ctx["cfg"], ctx["ds"], ctx["dataset"], name, ctx["train"], ctx["test"], ctx["args"], ctx["env"]
    )
    with open(ctx["out_dir"] / f"{name}_benchmark.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=1, default=str)
    return (
        f"[DONE] {ctx['dataset']}/{name}: train {result['total_train_time_s']:.1f}s, "
        f"single {result['pred_single_us_mean']:.1f}us, batch {result['pred_batch_s']:.3f}s/"
        f"{result['pred_batch_samples']}, peak RSS {result['peak_rss_mb']:.0f} MB, "
        f"size {result['model_size_mb']:.2f} MB, rmse {result['test_rmse']:.3f}"
    )


def _device_xy(splits: list | tuple, device_id: float, max_rows: int) -> tuple[np.ndarray, pd.Series]:
    """Concatenate all splits for one device into ``(X, y)`` (capped at ``max_rows``).

    ``splits`` may be a list of ``(X, y)`` tuples or a single ``(X, y)`` tuple.
    """
    from src.utils.utils import get_device_data

    if isinstance(splits, tuple):  # single (X, y) split
        splits = [splits]
    xs, ys = [], []
    for X, y in splits:
        X_d, y_d = get_device_data((X, y), device_id)
        if X_d is None or len(X_d) == 0:
            continue
        xs.append(X_d)
        ys.append(y_d)
    if not xs:
        return np.empty((0, 0)), pd.Series(dtype=float)
    X = np.vstack(xs)
    y = pd.concat(ys)
    if max_rows > 0 and len(X) > max_rows:
        X, y = X[:max_rows], y.iloc[:max_rows]
    return X, y


def _percentile(vals: list[float], q: float) -> float:
    return float(np.percentile(vals, q)) if vals else float("nan")


def benchmark_model(
    cfg: dict,
    ds: dict,
    dataset: str,
    name: str,
    train_splits: list,
    test_splits: list | tuple,
    args: argparse.Namespace,
    env: dict,
) -> dict:
    """Benchmark a single model on up to ``args.max_devices`` devices."""
    from src.data.feature_engineering import filter_features_for_forecast_horizon
    from src.evaluation.training import _predict_model, seq_fit_predict_batch_model, seq_fit_predict_incremental_model
    from src.utils.memory import calculate_memory_usage

    model_class, params, incremental, update_freq = resolve_model(cfg, dataset, name)
    protocol = cfg["protocol"]
    horizon = protocol["forecast_horizon"]

    # feature columns after forecast-horizon filtering (same as the pipeline)
    valid_features, _ = filter_features_for_forecast_horizon(ds["features"][2:], horizon)
    feature_indices = [2 + i for i, feat in enumerate(ds["features"][2:]) if feat in valid_features]

    device_ids = sorted({d for X, _ in train_splits for d in np.unique(X[:, 1])})[: args.max_devices]

    total_train_time = 0.0
    single_lat_us: list[float] = []
    batch_time_s = 0.0
    batch_n = 0
    all_true: list[float] = []
    all_pred: list[float] = []
    model_size_mb = float("nan")

    for device_id in device_ids:
        X_train, y_train = _device_xy(train_splits, device_id, args.max_train)
        X_eval, y_eval = _device_xy(test_splits, device_id, -1)
        if len(X_train) == 0 or len(X_eval) == 0:
            continue
        model = model_class(**params)

        if incremental:
            fit_fn = seq_fit_predict_incremental_model
            kwargs = {}
        else:
            fit_fn = seq_fit_predict_batch_model
            kwargs = {"update_freq": update_freq}

        t0 = time.time()
        _, y_pred = fit_fn(
            model,
            X_train,
            y_train,
            X_eval,
            y_eval,
            horizon,
            features=ds["features"],
            predict_train=False,
            verbose=0,
            **kwargs,
        )
        total_train_time += time.time() - t0

        n = min(args.predict_samples, len(X_eval))
        Xe = X_eval[:n, feature_indices]
        # single-sample latency: one prediction call per sample (pipeline style)
        for i in range(n):
            t0 = time.perf_counter()
            _predict_model(model, Xe[i : i + 1])
            single_lat_us.append((time.perf_counter() - t0) * 1e6)
        # batched prediction: one call for the whole block
        t0 = time.perf_counter()
        _predict_model(model, Xe)
        batch_time_s += time.perf_counter() - t0
        batch_n += n

        yp = np.asarray(y_pred.iloc[:n], dtype=float)
        yt = np.asarray(y_eval.iloc[:n], dtype=float)
        all_true.extend(yt.tolist())
        all_pred.extend(yp.tolist())
        model_size_mb = calculate_memory_usage(model)

    yt = np.asarray(all_true, dtype=float)
    yp = np.asarray(all_pred, dtype=float)
    err = yp - yt
    rmse = float(np.sqrt(np.mean(err**2))) if len(err) else float("nan")
    mae = float(np.mean(np.abs(err))) if len(err) else float("nan")
    mape = float(np.mean(np.abs(err / np.where(yt == 0, np.nan, yt)))) if len(err) else float("nan")

    return {
        "dataset": dataset,
        "model": name,
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "params": params,
        "incremental": incremental,
        "update_freq": update_freq,
        "n_devices": len(device_ids),
        "total_train_time_s": total_train_time,
        "pred_single_us_mean": float(np.mean(single_lat_us)) if single_lat_us else float("nan"),
        "pred_single_us_p50": _percentile(single_lat_us, 50),
        "pred_single_us_p99": _percentile(single_lat_us, 99),
        "pred_batch_s": batch_time_s,
        "pred_batch_samples": batch_n,
        "peak_rss_mb": peak_rss_mb(),
        "model_size_mb": model_size_mb,
        "test_rmse": rmse,
        "test_mae": mae,
        "test_mape": mape,
        "environment": env,
    }


def main(argv: list[str] | None = None) -> int:
    """Run the computational benchmark."""
    args = parse_args(argv)
    ilm_path = ensure_ilm_on_path(args.ilm_path)
    cfg = load_config(args.config)
    ds = resolve_dataset(cfg, args.dataset)

    data_root = (
        Path(args.data_root or os.environ.get("FASTODT_DATA_ROOT") or (ilm_path / "data")).expanduser().resolve()
    )
    data_dir = data_root / ds["data_dir"]

    if not (data_dir / f"{ds['prefix']}_train.npz").exists():
        print(
            f"[SKIP] dataset={args.dataset}: no '{ds['prefix']}_train.npz' under {data_dir} "
            f"(data root {data_root}); dataset not available, skipping"
        )
        return 0

    from src.data.io import load_dataset

    max_length = args.max_length if args.max_length and args.max_length > 0 else 10**12
    train, _, test = load_dataset(
        data_dir=str(data_dir), prefix=ds["prefix"], use_numpy=True, max_length=max_length, valid=False
    )
    env = collect_environment(ilm_path)
    out_dir = Path(args.results_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or ds["models"]
    ctx = dict(cfg=cfg, ds=ds, dataset=args.dataset, train=train, test=test, args=args, env=env, out_dir=out_dir)
    for line in run_over_models(args.jobs, models, _bench_worker, ctx):
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
