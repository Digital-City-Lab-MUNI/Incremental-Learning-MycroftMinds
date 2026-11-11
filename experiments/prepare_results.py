"""Aggregate raw experiment results into paper-ready tables.

Reads ``<results-dir>/<dataset>/`` produced by ``run_evaluation.py`` and
``run_benchmark.py`` and writes to ``<out>``:

* ``<dataset>_valid.csv`` / ``<dataset>_test.csv`` — per-model metrics at the
  configured aggregation level (``hourly``), averaged over devices
  (mean of per-device means, plus across-device std), with provenance columns;
* ``<dataset>_benchmark.csv`` — computational benchmark per model (total
  training time incl. batch refits, single-sample and batched prediction,
  peak RSS, model size, accuracy);
* ``summary.md`` — markdown tables for all datasets + a provenance section
  (model-repo git revision, environment, timestamps);
* ``comparison.csv`` — only with ``--baseline``: metric deltas vs a previous
  raw-results directory (same layout), to compare corrected vs prior results.

Use ``--datasets`` / ``--models`` to tabulate only a subset of the raw results.

Example::

    python -m experiments.prepare_results --results-dir results/rerun/raw --out results/rerun/tables \
        --baseline /path/to/previous/rerun/raw
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# metric keys produced by src.evaluation.metrics (see evaluation/constants.py)
METRICS = [
    "rmse",
    "mae",
    "mape",
    "dir_accuracy",
    "first_diff_corr",
    "spikes_immediate.f1",
    "spikes_delayed.f1",
    "drops_immediate.f1",
    "drops_delayed.f1",
]
PERC_METRICS = {
    "mape",
    "dir_accuracy",
    "first_diff_corr",
    "spikes_immediate.f1",
    "spikes_delayed.f1",
    "drops_immediate.f1",
    "drops_delayed.f1",
}
BENCH_COLS = [
    "total_train_time_s",
    "pred_single_us_mean",
    "pred_single_us_p50",
    "pred_single_us_p99",
    "pred_batch_s",
    "pred_batch_samples",
    "peak_rss_mb",
    "model_size_mb",
    "test_rmse",
    "test_mae",
    "test_mape",
]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--results-dir", default="results/rerun/raw", help="Raw results root (per-dataset subdirs)")
    ap.add_argument("--out", default="results/rerun/tables", help="Output dir for tables")
    ap.add_argument("--agg-level", default="hourly", help="Aggregation level to tabulate")
    ap.add_argument("--datasets", nargs="*", default=None, help="Subset of dataset dirs to tabulate")
    ap.add_argument("--models", nargs="*", default=None, help="Subset of models to tabulate")
    ap.add_argument("--baseline", default=None, help="Optional previous raw-results dir for delta columns")
    return ap.parse_args(argv)


def _agg_rows_for_split(pkl_path: Path, agg_level: str) -> dict[str, float]:
    """Average aggregated metrics over devices for one result pkl."""
    with open(pkl_path, "rb") as f:
        payload = pickle.load(f)
    key = "aggregated_valid" if pkl_path.name.endswith("_valid.pkl") else "aggregated_test"
    agg = payload.get(key, {})
    rows: dict[str, float] = {}
    for metric in METRICS:
        vals = []
        for _device_id, levels in agg.items():
            entry = levels.get(agg_level, {}).get(metric)
            if isinstance(entry, dict) and entry.get("mean") is not None:
                vals.append(float(entry["mean"]))
            elif isinstance(entry, (int, float)):
                vals.append(float(entry))
        rows[metric] = float(np.mean(vals)) if vals else float("nan")
        rows[metric + "_std"] = float(np.std(vals)) if vals else float("nan")
    return rows


def collect_dataset(
    results_dir: Path, dataset: str, agg_level: str, models: set[str] | None = None
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Build (metrics_df, bench_df, provenance) for one dataset directory."""
    ds_dir = results_dir / dataset
    metric_rows, bench_rows = [], []
    provenance: dict = {"dataset": dataset}
    for meta_path in sorted(ds_dir.glob("*_meta.json")):
        name = meta_path.name[: -len("_meta.json")]
        if models is not None and name not in models:
            continue
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        row = {"model": name, "model_class": meta.get("model_class", ""), "params": json.dumps(meta.get("params", {}))}
        for split in ("valid", "test"):
            pkl = ds_dir / f"{name}_{split}.pkl"
            if pkl.exists():
                for metric, val in _agg_rows_for_split(pkl, agg_level).items():
                    row[f"{split}_{metric}"] = val
        row["valid_time_s"] = meta.get("valid_time_s")
        row["test_time_s"] = meta.get("test_time_s")
        row["peak_rss_mb"] = meta.get("peak_rss_mb")
        env = meta.get("environment", {})
        row["ilm_git_revision"] = env.get("ilm_git_revision")
        row["created_utc"] = env.get("created_utc")
        row["hostname"] = env.get("hostname")
        metric_rows.append(row)
        provenance.setdefault("ilm_git_revision", row["ilm_git_revision"])
        provenance.setdefault("created_utc", row["created_utc"])
        provenance["environment"] = env
    for bench_path in sorted(ds_dir.glob("*_benchmark.json")):
        name = bench_path.name[: -len("_benchmark.json")]
        if models is not None and name not in models:
            continue
        with open(bench_path, encoding="utf-8") as f:
            b = json.load(f)
        brow = {"model": name}
        brow.update({k: b.get(k) for k in BENCH_COLS})
        brow["ilm_git_revision"] = (b.get("environment") or {}).get("ilm_git_revision")
        bench_rows.append(brow)
    metrics_df = pd.DataFrame(metric_rows)
    bench_df = pd.DataFrame(bench_rows)
    return metrics_df, bench_df, provenance


def add_deltas(
    df: pd.DataFrame, baseline_df: pd.DataFrame, key_cols: list[str], metric_prefixes: tuple[str, ...]
) -> pd.DataFrame:
    """Append ``<col>_delta`` columns = value - baseline value, matched on key_cols + model."""
    if baseline_df is None or baseline_df.empty:
        return df
    base = baseline_df.set_index(key_cols + ["model"])
    idx = df.set_index(key_cols + ["model"]).index
    out = df.copy()
    for col in df.columns:
        if col.startswith(metric_prefixes) and not col.endswith("_std") and col not in key_cols + ["model"]:
            if col in base.columns:
                out[col + "_delta"] = df[col].to_numpy(dtype=float) - base[col].reindex(idx).to_numpy(dtype=float)
    return out


def md_table(df: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    """Render a DataFrame as a GitHub-flavored markdown table."""
    if df.empty:
        return "_no results_"
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(floatfmt.format(v) if np.isfinite(v) else "")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """Aggregate raw results into CSV tables and a markdown summary."""
    args = parse_args(argv)
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.datasets:
        datasets = [d for d in args.datasets if (results_dir / d).is_dir()]
        missing = [d for d in args.datasets if not (results_dir / d).is_dir()]
        if missing:
            print(f"[WARN] missing dataset dirs under {results_dir}: {missing}", file=sys.stderr)
    else:
        datasets = sorted(p.name for p in results_dir.iterdir() if p.is_dir())
    if not datasets:
        print(f"[WARN] no dataset dirs under {results_dir}", file=sys.stderr)
        return 1

    model_filter = set(args.models) if args.models else None
    baseline_dir = Path(args.baseline) if args.baseline else None
    summary: list[str] = ["# FastODT rerun results", ""]
    all_metrics, all_bench = [], []

    for dataset in datasets:
        metrics_df, bench_df, prov = collect_dataset(results_dir, dataset, args.agg_level, models=model_filter)
        if baseline_dir and (baseline_dir / dataset).is_dir():
            b_metrics, _, _ = collect_dataset(
                baseline_dir, dataset, args.agg_level, models=model_filter
            )
            metrics_df = add_deltas(metrics_df, b_metrics, ["model"], ("valid_", "test_"))
        metrics_df.insert(0, "dataset", dataset)
        bench_df.insert(0, "dataset", dataset)
        metrics_df.to_csv(out_dir / f"{dataset}_metrics.csv", index=False)
        bench_df.to_csv(out_dir / f"{dataset}_benchmark.csv", index=False)
        all_metrics.append(metrics_df)
        all_bench.append(bench_df)

        summary.append(f"## Dataset `{dataset}` (agg: {args.agg_level})")
        summary.append("### Accuracy (valid/test, mean over devices)")
        show_cols = ["dataset", "model"] + [
            c for c in metrics_df.columns if c.startswith(("valid_", "test_")) and not c.endswith("_std")
        ]
        summary.append(md_table(metrics_df[show_cols]))
        summary.append("### Computational benchmark")
        summary.append(md_table(bench_df))
        env = prov.get("environment", {})
        summary.append("### Provenance")
        summary.append(f"- model repo git revision: `{prov.get('ilm_git_revision')}`")
        summary.append(f"- created (UTC): {prov.get('created_utc')} on `{env.get('hostname')}`")
        summary.append(
            f"- python {env.get('python')}, libs: "
            + ", ".join(f"{k}={v}" for k, v in (env.get("libraries") or {}).items())
        )
        summary.append("")

    if all_metrics:
        pd.concat(all_metrics).to_csv(out_dir / "all_metrics.csv", index=False)
    if all_bench:
        pd.concat(all_bench).to_csv(out_dir / "all_benchmark.csv", index=False)
    if baseline_dir:
        pd.concat(all_metrics).to_csv(out_dir / "comparison.csv", index=False)

    (out_dir / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")
    print(f"[DONE] tables written to {out_dir}/ (summary.md, *_metrics.csv, *_benchmark.csv, all_*.csv)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
