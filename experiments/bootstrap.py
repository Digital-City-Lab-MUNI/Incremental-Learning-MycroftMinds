"""Shared helpers: locate the model repo on ``sys.path``, import classes, capture environment."""

from __future__ import annotations

import importlib
import hashlib
import json
import os
import platform
import socket
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def _repo_root() -> Path:
    """Return the repo root: this file lives at ``<repo>/experiments/bootstrap.py``."""
    return Path(__file__).resolve().parents[1]


def ensure_ilm_on_path(ilm_path: str | Path | None = None) -> Path:
    """Resolve the model repo (this repository) and put it on ``sys.path``.

    Resolution order: explicit argument, ``ILM_PATH`` env var, then the repository
    root auto-detected from this file's location (``<repo>/experiments/``).
    """
    raw = ilm_path or os.environ.get("ILM_PATH") or _repo_root()
    p = Path(raw).expanduser().resolve()
    if not (p / "src").is_dir():
        raise FileNotFoundError(
            f"Model repository not found at {p} (expected a 'src' package inside). Pass --ilm-path or set ILM_PATH."
        )
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
    return p


def import_class(dotted: str) -> Any:
    """Import ``module.Class`` from a dotted path."""
    module, _, attr = dotted.rpartition(".")
    if not module:
        raise ValueError(f"Not a dotted class path: {dotted!r}")
    return getattr(importlib.import_module(module), attr)


def git_revision(path: Path) -> str:
    """Return the git revision of ``path`` (``<sha>-dirty`` if the worktree is dirty), else ``unknown``."""
    try:
        rev = subprocess.run(
            ["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(path), "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def collect_environment(ilm_path: Path) -> dict:
    """Capture reproducibility details: host, Python, library versions, thread counts, model-repo revision."""
    libraries: dict[str, str | None] = {}
    for name in ("numpy", "pandas", "scipy", "sklearn", "river", "tqdm", "pympler", "yaml"):
        try:
            libraries[name] = str(importlib.import_module(name).__version__)
        except Exception:
            libraries[name] = None
    thread_env = {
        k: v
        for k, v in os.environ.items()
        if k
        in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
            "VECLIB_MAXIMUM_THREADS",
        )
    }
    return {
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "cpu_count": os.cpu_count(),
        "thread_env": thread_env,
        "libraries": libraries,
        "ilm_path": str(ilm_path),
        "ilm_git_revision": git_revision(ilm_path),
    }


def peak_rss_mb() -> float:
    """Peak resident set size of this process in MB (Linux: ru_maxrss is KiB).

    Used as the consistent peak-memory measurement across all models.
    """
    import resource

    divisor = 1024.0**2 if sys.platform == "darwin" else 1024.0
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / divisor


def run_fingerprint(config: str, root: Path, data_dir: Path, prefix: str, settings: dict) -> str:
    """Identify code/config, dataset file stats, and settings for safe result reuse."""
    digest = hashlib.sha256()
    paths = sorted((root / "src").rglob("*.py")) + sorted((root / "experiments").glob("*.py"))
    for path in paths:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(Path(config).read_bytes())
    digest.update(git_revision(root).encode())
    for split in ("train", "valid", "test"):
        path = data_dir / f"{prefix}_{split}.npz"
        stat = path.stat()
        digest.update(f"{path.resolve()}:{stat.st_size}:{stat.st_mtime_ns}".encode())
    digest.update(json.dumps(settings, sort_keys=True).encode())
    return digest.hexdigest()


# Globals inherited by forked workers in run_in_parallel (avoids pickling the
# loaded dataset through the task queue).
_WORKER = None


def _worker_entry(item: Any) -> Any:
    fn, ctx = _WORKER
    try:
        return fn(item, ctx)
    except Exception as exc:  # one bad item must not kill the whole run
        traceback.print_exc()
        return f"[FAIL] {item}: {type(exc).__name__}: {exc}"


def run_in_parallel(jobs: int, items: list, fn: Callable[[Any, dict], Any], ctx: dict) -> list:
    """Run ``fn(item, ctx)`` for each item, optionally in parallel processes.

    With ``jobs > 1`` and more than one item, the work runs in ``jobs`` forked
    worker processes; the parent-loaded data inside ``ctx`` is shared with the
    workers copy-on-write (Linux fork), so it is never pickled through the task
    queue. Per-worker BLAS/OpenMP thread counts are capped to avoid
    oversubscription (e.g. RandomForest ``n_jobs=-1`` inside each worker).
    Falls back to sequential execution when fork is unavailable or jobs <= 1.
    Returns the list of per-item results in item order; an item that raises
    yields a ``[FAIL] <item>: ...`` string instead of aborting the run.
    """
    if jobs and jobs > 1 and len(items) > 1:
        import multiprocessing as mp

        global _WORKER
        try:
            mp_ctx = mp.get_context("fork")
        except ValueError:
            mp_ctx = None
        if mp_ctx is not None:
            _WORKER = (fn, ctx)
            n = min(jobs, len(items))
            per = max(1, (os.cpu_count() or n) // n)
            thread_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS")
            saved = {k: os.environ.get(k) for k in thread_keys}
            for k in thread_keys:
                os.environ[k] = str(per)
            try:
                with mp_ctx.Pool(processes=n) as pool:
                    return pool.map(_worker_entry, items)
            finally:
                _WORKER = None
                for k, v in saved.items():
                    if v is None:
                        os.environ.pop(k, None)
                    else:
                        os.environ[k] = v
    results = []
    for item in items:
        try:
            results.append(fn(item, ctx))
        except Exception as exc:  # one bad item must not kill the whole run
            traceback.print_exc()
            results.append(f"[FAIL] {item}: {type(exc).__name__}: {exc}")
    return results


def run_over_models(jobs: int, models: list, fn: Callable[[str, dict], Any], ctx: dict) -> list:
    """Run ``fn(name, ctx)`` for each model name, optionally in parallel processes.

    Thin wrapper around :func:`run_in_parallel`; see its docstring for details.
    """
    return run_in_parallel(jobs, models, fn, ctx)
