"""Shared helpers: locate the model repo on ``sys.path``, import classes, capture environment."""

from __future__ import annotations

import importlib
import os
import platform
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


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

    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
