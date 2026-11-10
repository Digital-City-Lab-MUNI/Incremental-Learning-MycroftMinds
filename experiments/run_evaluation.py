"""Rerun the notebook evaluation protocol for one dataset x a set of models.

For each (dataset, model) this script mirrors what the evaluation notebooks
(``notebooks/*batch/*_evaluation.ipynb`` and ``notebooks/other_data/*_evaluation.ipynb``) do:

1. walk-forward evaluation over the validation splits
   (``all_valid_splits_seq_fit_predict``; batch models are refit every
   ``update_freq_batch`` inside the pipeline, so the measured wall time is the
   total training time *including* periodic refits),
2. final test-set evaluation (``test_set_seq_fit_predict``),
3. aggregation of per-period metrics (``all_time_aggregation_metrics``).

Raw outputs are written to ``<results-dir>/<dataset>/``:

* ``<Model>_valid.pkl`` / ``<Model>_test.pkl`` — predictions, per-period metrics
  and aggregated metrics (pickle),
* ``<Model>_meta.json`` — params, timing, peak RSS, environment / git provenance.

Example (from this repo's root)::

    python -m experiments.run_evaluation --dataset sb_ind_1h --results-dir results/rerun/raw
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

from .bootstrap import collect_environment, ensure_ilm_on_path, peak_rss_mb, run_over_models
from .registry import load_config, resolve_dataset, resolve_model


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml")
    ap.add_argument("--dataset", required=True, help="Dataset key, e.g. sb_ind_1h")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of model names; default: all in the dataset")
    ap.add_argument(
        "--ilm-path", default=None, help="Path to this repo (or ILM_PATH env); default: auto-detected repo root"
    )
    ap.add_argument(
        "--data-root",
        default=None,
        help="Root containing the data dirs (or FASTODT_DATA_ROOT env); default <ilm-path>/data",
    )
    ap.add_argument("--results-dir", default="results/rerun/raw", help="Output dir for raw results")
    ap.add_argument("--max-length", type=int, default=-1, help="Cap samples per split (passed to load_dataset)")
    ap.add_argument("--start-split", type=int, default=0, help="First validation split to process")
    ap.add_argument("--end-split", type=int, default=None, help="End split (exclusive)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run this many models in parallel worker processes (fork; loaded data is shared "
        "copy-on-write). Default 1 = sequential",
    )
    return ap.parse_args(argv)


def _eval_model(name: str, ctx: dict) -> str:
    """Run the valid-splits walk-forward + test evaluation for one model.

    Writes ``<Model>_valid.pkl`` / ``<Model>_test.pkl`` / ``<Model>_meta.json`` into
    ``ctx['out_dir']`` and returns the ``[DONE]`` log line.
    """
    cfg, ds, protocol = ctx["cfg"], ctx["ds"], ctx["protocol"]
    model_class, params, incremental, update_freq = resolve_model(cfg, ctx["dataset"], name)
    common = ctx["common"]

    t0 = time.time()
    tr_preds, va_preds, tr_metrics, va_metrics = all_valid_splits_seq_fit_predict(
        model_class=model_class,
        model_params=params,
        train=ctx["train"],
        valid=ctx["valid"],
        start_split=ctx["start_split"],
        end_split=ctx["end_split"],
        tqdm_on=ctx["tqdm_on"],
        **common,
    )
    valid_time_s = time.time() - t0
    va_agg = all_time_aggregation_metrics(va_metrics, data_freq=ds["data_freq"], verbose=False)

    t0 = time.time()
    tr_preds_t, te_preds, tr_metrics_t, te_metrics = test_set_seq_fit_predict(
        model_class=model_class,
        model_params=params,
        train=ctx["train"],
        valid=ctx["valid"],
        test=ctx["test"],
        normalize_robust=protocol["normalize_robust"],
        predict_train=protocol["predict_train_test"],
        **common,
    )
    test_time_s = time.time() - t0
    te_agg = all_time_aggregation_metrics(te_metrics, data_freq=ds["data_freq"], verbose=False)

    out_dir = ctx["out_dir"]
    with open(out_dir / f"{name}_valid.pkl", "wb") as f:
        pickle.dump(
            {
                "predictions_train": tr_preds,
                "predictions_valid": va_preds,
                "metrics_train": tr_metrics,
                "metrics_valid": va_metrics,
                "aggregated_valid": va_agg,
            },
            f,
        )
    with open(out_dir / f"{name}_test.pkl", "wb") as f:
        pickle.dump(
            {
                "predictions_train": tr_preds_t,
                "predictions_test": te_preds,
                "metrics_train": tr_metrics_t,
                "metrics_test": te_metrics,
                "aggregated_test": te_agg,
            },
            f,
        )
    meta = {
        "dataset": ctx["dataset"],
        "model": name,
        "model_class": f"{model_class.__module__}.{model_class.__name__}",
        "params": params,
        "incremental": incremental,
        "update_freq": update_freq,
        "data_dir": str(ctx["data_dir"]),
        "prefix": ds["prefix"],
        "valid_time_s": valid_time_s,
        "test_time_s": test_time_s,
        "peak_rss_mb": peak_rss_mb(),
        "environment": ctx["env"],
    }
    with open(out_dir / f"{name}_meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=1, default=str)

    return (
        f"[DONE] {ctx['dataset']}/{name}: valid {valid_time_s:.1f}s, test {test_time_s:.1f}s, "
        f"peak RSS {meta['peak_rss_mb']:.0f} MB, ilm_rev {ctx['env']['ilm_git_revision']}"
    )


def main(argv: list[str] | None = None) -> int:
    """Run the evaluation for the requested dataset and models."""
    args = parse_args(argv)
    ilm_path = ensure_ilm_on_path(args.ilm_path)
    cfg = load_config(args.config)
    ds = resolve_dataset(cfg, args.dataset)
    protocol = cfg["protocol"]

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

    # imported only after the model repo is on sys.path; bound as module globals so
    # the (possibly forked) per-model worker can see them too
    global load_dataset, all_time_aggregation_metrics, all_valid_splits_seq_fit_predict, test_set_seq_fit_predict
    from src.data.io import load_dataset
    from src.evaluation.metrics import all_time_aggregation_metrics
    from src.evaluation.training import all_valid_splits_seq_fit_predict, test_set_seq_fit_predict

    # load_dataset slices result[:max_length], so -1 ("no cap") would drop the
    # last split; translate it to a huge cap instead.
    max_length = args.max_length if args.max_length and args.max_length > 0 else 10**12
    print(f"[INFO] dataset={args.dataset} data_dir={data_dir} prefix={ds['prefix']}")
    train, valid, test = load_dataset(
        data_dir=str(data_dir), prefix=ds["prefix"], use_numpy=True, max_length=max_length
    )

    env = collect_environment(ilm_path)
    out_dir = Path(args.results_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    models = args.models or ds["models"]
    # verbose=0: per-device/split metric printing and per-retrain pympler walks only
    # slow the run down; timings and peak RSS are still recorded in <Model>_meta.json.
    common = dict(
        use_residuals=ds["use_residuals"],
        data_freq=ds["data_freq"],
        forecast_horizon=protocol["forecast_horizon"],
        features=ds["features"],
        spike_prominence=ds["spike_prominence"],
        verbose=0,
    )
    ctx = dict(
        cfg=cfg,
        dataset=args.dataset,
        ds=ds,
        protocol=protocol,
        data_dir=data_dir,
        train=train,
        valid=valid,
        test=test,
        out_dir=out_dir,
        common=common,
        start_split=args.start_split,
        end_split=args.end_split,
        env=env,
        tqdm_on=args.jobs <= 1,
    )

    # update_freq differs per model (incremental vs batch), so resolve it inside the worker
    def _worker(name: str, c: dict) -> str:
        model_class, params, incremental, update_freq = resolve_model(c["cfg"], c["dataset"], name)
        c = dict(c)
        c["common"] = dict(c["common"], incremental=incremental, update_freq=update_freq)
        return _eval_model(name, c)

    for line in run_over_models(args.jobs, models, _worker, ctx):
        if line:
            print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
