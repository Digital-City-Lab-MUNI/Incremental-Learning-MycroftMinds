"""Run the full experiment matrix (datasets x models) in parallel without SLURM.

Each dataset is one work unit: walk-forward valid evaluation + test evaluation
(``run_evaluation``) followed by the computational benchmark (``run_benchmark``),
then the summary tables are rebuilt once at the end (``prepare_results``).
Dataset units run in ``--jobs`` forked worker processes, so a plain server works::

    python -m experiments.run_matrix --jobs 4 --model-jobs 2 --skip-existing

This is the non-SLURM equivalent of ``jobs/job.sh`` (one array task per dataset
with ``JOBS=model_jobs``). Total concurrent processes are roughly
``jobs x model_jobs``; every dataset worker loads its own data, so on a
memory-limited machine keep ``--jobs`` modest (compare the SLURM job's 32G per
task). ``--skip-existing`` makes re-running interrupted batches a no-op for
already-finished models.

Example (from this repo's root)::

    python -m experiments.run_matrix --jobs 4 --datasets sb_ind_1h od_aq_1h \
        --models FastODT xFastODT --results-dir results/rerun
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from .bootstrap import ensure_ilm_on_path, run_in_parallel
from .registry import dataset_names, load_config


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse CLI arguments."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/experiments.yaml", help="Path to experiments.yaml")
    ap.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Subset of dataset keys; default: all datasets in the config",
    )
    ap.add_argument("--models", nargs="*", default=None, help="Subset of model names (forwarded to both stages)")
    ap.add_argument("--ilm-path", default=None, help="Path to this repo (or ILM_PATH env); default: auto-detected")
    ap.add_argument(
        "--data-root",
        default=None,
        help="Root containing the data dirs (or FASTODT_DATA_ROOT env); default <ilm-path>/data",
    )
    ap.add_argument(
        "--results-dir",
        default="results/rerun",
        help="Output root; raw results go to <results-dir>/raw, tables to <results-dir>/tables",
    )
    ap.add_argument("--max-length", type=int, default=-1, help="Cap samples per split (passed to load_dataset)")
    ap.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Run this many dataset workers in parallel (fork). Default 1 = sequential",
    )
    ap.add_argument(
        "--model-jobs",
        type=int,
        default=1,
        help="Model-level parallelism forwarded as --jobs to each per-dataset stage. Default 1",
    )
    ap.add_argument(
        "--skip-existing",
        action="store_true",
        help="Forward --skip-existing to both stages: models whose results already exist are skipped",
    )
    ap.add_argument("--no-benchmark", action="store_true", help="Skip the computational benchmark stage")
    return ap.parse_args(argv)


def _dataset_worker(dataset: str, ctx: dict) -> str:
    """Run evaluation (+ benchmark) for one dataset; return a ``[DONE]``/``[FAIL]`` line."""
    from . import run_benchmark, run_evaluation

    base = [
        "--config", ctx["config"],
        "--data-root", ctx["data_root"],
        "--results-dir", ctx["raw_dir"],
        "--max-length", str(ctx["max_length"]),
        "--jobs", str(ctx["model_jobs"]),
    ]
    if ctx["skip_existing"]:
        base.append("--skip-existing")
    if ctx["models"]:
        base += ["--models", *ctx["models"]]
    base += ["--dataset", dataset]

    rc_eval = run_evaluation.main(base)
    rc_bench = 0 if ctx["no_benchmark"] else run_benchmark.main(base)
    if rc_eval == 0 and rc_bench == 0:
        return f"[DONE] {dataset}"
    return f"[FAIL] {dataset} (eval rc={rc_eval}, bench rc={rc_bench})"


def main(argv: list[str] | None = None) -> int:
    """Run the matrix: per-dataset stages in parallel, then rebuild the tables."""
    args = parse_args(argv)
    ilm_path = ensure_ilm_on_path(args.ilm_path)
    cfg = load_config(args.config)

    datasets = args.datasets or dataset_names(cfg)
    unknown = [d for d in datasets if d not in cfg["datasets"]]
    if unknown:
        print(f"[ERROR] unknown datasets {unknown}; available: {dataset_names(cfg)}")
        return 2

    data_root = (
        Path(args.data_root or os.environ.get("FASTODT_DATA_ROOT") or (ilm_path / "data")).expanduser().resolve()
    )
    results_dir = Path(args.results_dir).expanduser().resolve()
    raw_dir = results_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    ctx = dict(
        config=args.config,
        data_root=str(data_root),
        raw_dir=str(raw_dir),
        max_length=args.max_length,
        model_jobs=args.model_jobs,
        models=args.models,
        skip_existing=args.skip_existing,
        no_benchmark=args.no_benchmark,
    )
    print(
        f"[INFO] matrix: {len(datasets)} datasets x --jobs {args.jobs} dataset workers "
        f"x --model-jobs {args.model_jobs}, results {results_dir}"
    )
    failed = 0
    for line in run_in_parallel(args.jobs, datasets, _dataset_worker, ctx):
        if line:
            print(line)
            if line.startswith("[FAIL]"):
                failed += 1

    from . import prepare_results

    prepare_results.main(["--results-dir", str(raw_dir), "--out", str(results_dir / "tables")])
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
