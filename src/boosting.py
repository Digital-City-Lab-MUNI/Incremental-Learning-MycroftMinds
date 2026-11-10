"""Incremental boosting ensembles with incubation for streaming regression.

This module provides two closely related ensemble variants that build a chain of
incremental base learners (e.g. :class:`FastODT`) trained on residuals, adding
new learners only once they are "hatched" (i.e., their initial structure / root
statistics are considered sufficiently filled). A simple pruning heuristic
similar to index-based pruning in efficient gradient boosting keeps only the
prefix of learners that continue to reduce error.

Classes
-------
IncubationBoost
    Specialized variant exposing common hyperparameters directly (tree-focused).
IncubationBoostAnyModel
    Generic variant that accepts a base learner class plus a parameter dict.

Both classes expose ``partial_fit`` for batch-style incremental updates and
``predict`` for single-sample prediction.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Type

from src.trees.fastodt import FastODT
import src.trees.constants as C
from sklearn.base import BaseEstimator
import tqdm
import numpy as np

BaseLearnerT = Type[BaseEstimator]


class IncubationBoost:
    """Incremental gradient-style boosting with a node incubation mechanism.

    Parameters
    ----------
    length : int, optional, default=5
        Maximum number of learners (ensemble size upper bound).
    learning_rate : float, optional, default=1.0
        Shrinkage factor applied to all learners except the first.
    depth : int, optional, default=15
        Maximum tree depth for the base learner. Forwarded to the base learner
        constructor.
    grace_period : int, optional, default=10
        Number of samples a leaf should observe between split attempts.
        Forwarded to the base learner.
    split_confidence : float, optional, default=0.01
        Hoeffding bound confidence (smaller values -> more eager splitting).
        Also known as ``delta``.
    tie_threshold : float, optional, default=0.05
        Threshold below which competing splits are forced to break ties. Forwarded to the base
        learner. Also known as ``tau``.
    max_feature : float, optional, default=1.0
        Fraction of features to consider per split (e.g., random subspace).
        Forwarded to the base learner.
    grace_period_depth : int, optional, default=-1
        Depth at which to change the grace period behavior (implementation
        specific). Forwarded to the base learner.
    eagerness : float, optional, default=0.0
        Controls eagerness of splitting (implementation specific). Forwarded to
        the base learner.
    bin_type : str, optional, default='adaptive'
        Binning strategy for histograms (e.g., ``"adaptive"``). Forwarded to
        the base learner.
    max_bins : int, optional, default=32
        Maximum number of bins per feature for histogram-based splits.
        Forwarded to the base learner.
    split_criterion : str, optional, default='vr_rel'
        Split criterion identifier (e.g., ``"vr_rel"``). Forwarded to the base
        learner.
    random_state : int, optional, default=None
        Random state for reproducibility. Created RNG is forwarded to the base learner.
        It has effect only when `max_feature` < 1.0. If None, the global NumPy
        RNG is used.
    verbose : bool, optional, default=True
        If True, display a progress bar during ``partial_fit``.

    Notes
    -----
    All base-learner-related parameters are forwarded to the base learner when
    learners are created for the chain and for the incubating node.

    Workflow per mini-batch sample::

        1. Predict sequentially with the existing chain, accumulating the
           strong prediction (all predictions captured before training on the
           current sample).
        2. A still-growing learner is trained on the residual of the prefix
           *preceding* it (``y - strong_prediction`` before adding its own
           contribution), so the first tree learns ``y`` itself.
        3. Greedy prefix pruning (eGBDT-style): scan mature learners in order
           and stop at the first prefix whose rolling error fails to improve;
           truncate the chain to the last improving prefix.
        4. Incubate a new learner on the residual of the retained chain
           (``y - retained_prefix_prediction``), excluding any rejected trees,
           until it becomes filled, then append if the length budget allows.
    """

    def __init__(
        self,
        length: int = 5,
        learning_rate: float = 1.0,
        depth: int = 15,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        grace_period_depth: int = -1,
        eagerness: float = 0.0,
        bin_type: str = C.HistogramType.ADAPTIVE,
        max_bins: int = 32,
        split_criterion: str = C.SplitCriterion.VR_REL,
        random_state: int = None,
        verbose: bool = True,
    ) -> None:
        self.chain: List[FastODT] = []
        self.length = length
        self.learning_rate = learning_rate
        self.base_learner = FastODT
        self.depth = depth
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.max_feature = max_feature
        self.grace_period_depth = grace_period_depth
        self.tie_threshold = tie_threshold
        self.eagerness = eagerness
        self.bin_type = bin_type
        self.max_bins = max_bins
        self.split_criterion = split_criterion
        self.incubating_node: Optional[FastODT] = None
        # Initialize local RNG
        self.rng = np.random.default_rng(random_state)
        self.verbose = verbose

        try:  # environment-adaptive progress bar
            from IPython import get_ipython  # type: ignore

            ipython = get_ipython()
            if ipython is not None and "IPKernelApp" in getattr(ipython, "config", {}):
                self.tqdm = tqdm.notebook.tqdm  # type: ignore[attr-defined]
            else:
                self.tqdm = tqdm.tqdm
        except Exception:  # pragma: no cover - fallback
            self.tqdm = tqdm.tqdm

    def partial_fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
        """Incrementally fit on a batch of samples."""
        iterator = range(len(X)) if not self.verbose else self.tqdm(range(len(X)))
        for i in iterator:
            self._partial_fit(X[i], y[i])

    def _partial_fit(self, X: Sequence[float], y: float) -> None:
        """Perform internal single-sample update (one residual boosting step)."""
        residual = y
        y_pred_strong = 0.0
        pruning_index: Optional[int] = None
        prev_pruning_error_rate = float("inf")
        # Prediction of the retained prefix, captured before any training on
        # the current sample (used for the incubation target after pruning).
        retained_prefix_prediction = 0.0

        if len(self.chain) == 0:
            new_node = self.base_learner(
                depth=self.depth,
                grace_period=self.grace_period,
                grace_period_depth=self.grace_period_depth,
                split_confidence=self.split_confidence,
                max_feature=self.max_feature,
                tie_threshold=self.tie_threshold,
                bin_type=self.bin_type,
                max_bins=self.max_bins,
                split_criterion=self.split_criterion,
                random_state=self.rng,
            )
            self.chain.append(new_node)

        # Index-based greedy prefix pruning (eGBDT-style): stop at the first
        # mature prefix that fails to improve the rolling error.
        for n, node in enumerate(self.chain):
            if node.is_filled():
                y_pred = node.predict(X) * (
                    1.0 if n == 0 else self.learning_rate
                )  # first node is not multiplied by learning rate
                y_pred_strong += y_pred
                residual = y - y_pred_strong

                error_rate = node.measure_error(y, y_pred_strong)
                if error_rate < prev_pruning_error_rate:
                    pruning_index = n
                    prev_pruning_error_rate = error_rate
                    retained_prefix_prediction = y_pred_strong
                else:
                    break
            else:
                # Growing tree: capture the training target BEFORE adding the
                # current tree's own contribution, so the first tree learns y
                # itself and every tree learns what the preceding prefix missed.
                # All predictions are taken before training on this sample.
                training_target = y - y_pred_strong
                y_pred = node.predict(X) * (
                    1.0 if n == 0 else self.learning_rate
                )  # first node is not multiplied by learning rate
                y_pred_strong += y_pred
                residual = y - y_pred_strong
                # learn_one is the single-sample update; partial_fit([X], ...) would
                # re-wrap and re-convert the row on every call (fall back for base
                # learners that only implement partial_fit)
                if hasattr(node, "learn_one"):
                    node.learn_one(X, training_target)
                else:
                    node.partial_fit([X], training_target)

        if pruning_index is not None:
            self.chain = self.chain[: pruning_index + 1]
            # Incubation target: residual of the chain actually retained.
            # The loop residual would still contain the rejected tree's
            # contribution, so recompute it from the retained prefix prediction.
            incubation_target = y - retained_prefix_prediction
        else:
            incubation_target = residual

        hatched_node = self.incubate(X, incubation_target)
        if hatched_node is not None and len(self.chain) < self.length:
            self.chain.append(hatched_node)

    def incubate(self, X: Sequence[float], residual: float) -> Optional[FastODT]:
        """Update (or create) an incubating learner on residual and hatch it.

        Parameters
        ----------
        X : sequence of float
            Feature vector of the current sample.
        residual : float
            Current residual target value.

        Returns
        -------
        base learner or None
            The hatched learner ready to append, else ``None`` while still incubating.
        """
        if self.incubating_node is None:
            self.incubating_node = self.base_learner(
                depth=self.depth,
                grace_period=self.grace_period,
                grace_period_depth=self.grace_period_depth,
                split_confidence=self.split_confidence,
                max_feature=self.max_feature,
                tie_threshold=self.tie_threshold,
                bin_type=self.bin_type,
                max_bins=self.max_bins,
                split_criterion=self.split_criterion,
                random_state=self.rng,
            )
            if hasattr(self.incubating_node, "learn_one"):
                self.incubating_node.learn_one(X, residual)
            else:
                self.incubating_node.partial_fit([X], residual)
            return None

        if hasattr(self.incubating_node, "learn_one"):
            self.incubating_node.learn_one(X, residual)
        else:
            self.incubating_node.partial_fit([X], residual)
        if self.incubating_node.is_filled():
            hatched = self.incubating_node
            self.incubating_node = None
            return hatched
        return None

    def predict(self, X: Sequence[float]) -> float:
        """Predict a single target value for a single feature vector."""
        y_pred = 0.0
        for n, node in enumerate(self.chain):
            y_pred += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
        return y_pred


# class IncubationBoostAnyModel:
#     """Generic incubation boosting variant accepting any incremental learner.

#     Parameters
#     ----------
#     length : int, default=5
#         Maximum number of learners retained.
#     learning_rate : float, default=1.0
#         Shrinkage factor for all but the first learner.
#     base_learner : type, default=FastODT
#         Incremental model class (must implement required interface).
#     base_learner_params : dict, optional
#         Parameter dictionary forwarded to the base learner constructor.
#     verbose : bool, default=True
#         If True display progress bar during ``partial_fit``.
#     """

#     def __init__(
#         self,
#         length: int = 5,
#         learning_rate: float = 1.0,
#         base_learner: Type[BaseLearnerT] = FastODT,
#         base_learner_params: dict[str, Any] | None = None,
#         verbose: bool = True,
#     ) -> None:
#         if base_learner_params is None:
#             base_learner_params = {
#                 "depth": 15,
#                 "grace_period": 10,
#                 "split_confidence": 0.01,
#                 "max_feature": 1.0,
#                 "grace_period_depth": -1,
#                 "eagerness": 0.0,
#                 "bin_type": "adaptive",
#                 "max_bins": 32,
#                 "split_criterion": "vr_rel",
#             }
#         self.chain: List[BaseLearnerT] = []
#         self.length = length
#         self.learning_rate = learning_rate
#         self.base_learner = base_learner
#         self.base_learner_params = base_learner_params
#         self.incubating_node: Optional[BaseLearnerT] = None
#         self.verbose = verbose
#         try:  # environment-adaptive progress bar
#             from IPython import get_ipython  # type: ignore

#             ipython = get_ipython()
#             if ipython is not None and "IPKernelApp" in getattr(ipython, "config", {}):
#                 self.tqdm = tqdm.notebook.tqdm  # type: ignore[attr-defined]
#             else:
#                 self.tqdm = tqdm.tqdm
#         except Exception:  # pragma: no cover
#             self.tqdm = tqdm.tqdm

#     def partial_fit(self, X: Sequence[Sequence[float]], y: Sequence[float]) -> None:
#         """Incrementally fit on a batch of samples (generic variant)."""
#         iterator = range(len(X)) if not self.verbose else self.tqdm(range(len(X)))
#         for i in iterator:
#             self._partial_fit(X[i], y[i])

#     def _partial_fit(self, X: Sequence[float], y: float) -> None:
#         """Internal single-sample update for generic variant."""
#         residual = y
#         y_pred_strong = 0.0
#         pruning_index: Optional[int] = None
#         prev_pruning_error_rate = float("inf")

#         if len(self.chain) == 0:
#             new_node = self.base_learner(**self.base_learner_params)
#             self.chain.append(new_node)

#         # Index-based pruning (eGBDT)
#         for n, node in enumerate(self.chain):
#             y_pred = node.predict(X) * (
#                 1.0 if n == 0 else self.learning_rate
#             )  # first node is not multiplied by learning rate
#             y_pred_strong += y_pred
#             residual = y - y_pred_strong

#             if node.is_filled():
#                 error_rate = node.measure_error(y, y_pred_strong)
#                 if error_rate < prev_pruning_error_rate:
#                     pruning_index = n
#                     prev_pruning_error_rate = error_rate
#                 else:
#                     break
#             else:
#                 node.partial_fit([X], residual)

#         if pruning_index is not None:
#             self.chain = self.chain[: pruning_index + 1]

#         hatched_node = self.incubate(X, residual)
#         if hatched_node is not None and len(self.chain) < self.length:
#             self.chain.append(hatched_node)

#     def incubate(self, X: Sequence[float], residual: float) -> Optional[BaseLearnerT]:
#         """Update / create incubating learner for generic variant."""
#         if self.incubating_node is None:
#             self.incubating_node = self.base_learner(**self.base_learner_params)
#             self.incubating_node.partial_fit([X], residual)
#             return None
#         self.incubating_node.partial_fit([X], residual)
#         if self.incubating_node.is_filled():
#             hatched = self.incubating_node
#             self.incubating_node = None
#             return hatched
#         return None

#     def predict(self, X: Sequence[float]) -> float:
#         """Predict a target value (generic variant)."""
#         y_pred = 0.0
#         for n, node in enumerate(self.chain):
#             y_pred += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
#         return y_pred
