"""Experiment configuration loading and model registry.

Model names map to classes in the Incremental-Learning-MycroftMinds ``src``
package (or sklearn / river). Per-dataset hyperparameters live in
``configs/experiments.yaml`` under ``datasets.<name>.model_params``.
"""

from __future__ import annotations

import copy
import inspect
from pathlib import Path
from typing import Any

import yaml

from .bootstrap import import_class

# name -> (dotted class path, incremental?)
MODEL_REGISTRY: dict[str, dict[str, Any]] = {
    "BL_Prev": {"class": "src.evaluation.baseline.PreviousValueRegressor", "incremental": True},
    "BL_Mean": {"class": "src.evaluation.baseline.SeasonalMeanRegressor", "incremental": False},
    "GBDT": {"class": "sklearn.ensemble.GradientBoostingRegressor", "incremental": False},
    "RF": {"class": "sklearn.ensemble.RandomForestRegressor", "incremental": False},
    "HoeffdingTree": {"class": "river.tree.HoeffdingTreeRegressor", "incremental": True},
    "ARF_Hoeffding": {"class": "river.forest.ARFRegressor", "incremental": True},
    "FastODT": {"class": "src.trees.fastodt.FastODT", "incremental": True},
    "xFastODT": {"class": "src.trees.xfastodt.xFastODT", "incremental": True},
    "xFastODT_C": {"class": "src.trees.native_xfastodt.NativeXFastODT", "incremental": True},
    "ARF_FastODT": {"class": "src.bagging.ARFRegressorFODT", "incremental": True},
    "ARF_FastODT_10x10": {"class": "src.bagging.ARFRegressorFODT", "incremental": True},
    "ARF_Hoeffding_10x10": {"class": "river.forest.ARFRegressor", "incremental": True},
    "IncubationBoost": {"class": "src.boosting.IncubationBoost", "incremental": True},
    "IncubationBoost_WR": {
        "class": "src.ensemble_methods.IncubationBoostWeightedRatio.IncubationBoostWeightedRatio",
        "incremental": True,
    },
    "IncubationBoost_Elastic": {
        "class": "src.ensemble_methods.IncubationBoostElastic.IncubationBoostElastic",
        "incremental": True,
    },
}

USE_RESIDUALS_SENTINEL = "$use_residuals"


def experiment_seed(cfg: dict) -> int:
    """Return the common experiment seed, restricted to NumPy's seed range."""
    seed = cfg["protocol"].get("seed", 42)
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError(f"protocol.seed must be an integer in [0, 2**32); got {seed!r}")
    return seed


def load_config(path: str | Path) -> dict:
    """Load ``configs/experiments.yaml``."""
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    for target, source in cfg.get("matched_models", {}).items():
        for dataset in cfg["datasets"].values():
            if source not in dataset["models"]:
                continue
            if target not in dataset["models"]:
                dataset["models"].append(target)
            dataset.setdefault("model_params", {})[target] = copy.deepcopy(dataset["model_params"][source])
    for target, variant in cfg.get("model_variants", {}).items():
        source = variant["source"]
        for dataset in cfg["datasets"].values():
            if source not in dataset["models"]:
                continue
            if target not in dataset["models"]:
                dataset["models"].append(target)
            params = copy.deepcopy(dataset["model_params"][source])
            params.update(copy.deepcopy(variant["params"]))
            dataset["model_params"][target] = params
    return cfg


def dataset_names(cfg: dict) -> list[str]:
    """Return the configured dataset keys."""
    return list(cfg["datasets"].keys())


def resolve_dataset(cfg: dict, dataset: str) -> dict:
    """Return the dataset section of the config, raising a helpful error if unknown."""
    if dataset not in cfg["datasets"]:
        raise KeyError(f"Unknown dataset {dataset!r}; available: {dataset_names(cfg)}")
    return cfg["datasets"][dataset]


def resolve_model(cfg: dict, dataset: str, name: str) -> tuple[type, dict, bool, str]:
    """Resolve a model name to ``(model_class, params, incremental, update_freq)`` for a dataset.

    ``$use_residuals`` in params is replaced by the dataset's ``use_residuals`` flag.
    ``update_freq`` is the dataset's ``data_freq`` for incremental models and the
    protocol's ``update_freq_batch`` (periodic refit interval) for batch models.
    """
    if name not in MODEL_REGISTRY:
        raise KeyError(f"Unknown model {name!r}; available: {sorted(MODEL_REGISTRY)}")
    ds = resolve_dataset(cfg, dataset)
    if name not in ds["models"]:
        raise KeyError(f"Model {name!r} is not part of dataset {dataset!r} (models: {ds['models']})")
    entry = MODEL_REGISTRY[name]
    params = copy.deepcopy(ds.get("model_params", {}).get(name, {}))

    def substitute(v: Any) -> Any:
        if isinstance(v, str) and v == USE_RESIDUALS_SENTINEL:
            return ds["use_residuals"]
        if isinstance(v, dict):
            return {k: substitute(x) for k, x in v.items()}
        if isinstance(v, list):
            return [substitute(x) for x in v]
        return v

    params = substitute(params)
    model_class = import_class(entry["class"])
    # Drop params the class constructor does not accept (config/class drift, e.g.
    # pruning_strategy on IncubationBoostElastic) instead of crashing the run.
    sig = inspect.signature(model_class.__init__)
    # Explicit estimator generators are independent of np.random.seed().
    # The protocol seed takes precedence over historical per-model seeds.
    seed = experiment_seed(cfg)
    for seed_param in ("random_state", "seed"):
        if seed_param in sig.parameters:
            params[seed_param] = seed
    if not any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        unsupported = [k for k in params if k not in sig.parameters]
        if unsupported:
            print(
                f"[WARN] {dataset}/{name}: ignoring params not accepted by "
                f"{model_class.__name__}.__init__: {unsupported}"
            )
            params = {k: v for k, v in params.items() if k in sig.parameters}
    incremental = bool(entry["incremental"])
    update_freq = (
        ds.get("update_freq_incremental", ds["data_freq"]) if incremental else cfg["protocol"]["update_freq_batch"]
    )
    return model_class, params, incremental, update_freq
