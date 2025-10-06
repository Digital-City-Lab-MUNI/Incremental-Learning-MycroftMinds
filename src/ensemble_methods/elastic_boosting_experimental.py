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

class IncubationBoost_experimental:
    """Incremental gradient-style boosting with a node incubation mechanism.

    Parameters
    ----------
    length : int, default=5
        Maximum number of learners (ensemble size upper bound). Default is 5.
    learning_rate : float, default=1.0
        Shrinkage factor applied to all learners except the first.
    depth : int, default=15
        Maximum tree depth for the base learner.
    grace_period : int, default=10
        Number of samples a leaf should observe between split attempts.
    split_confidence : float, default=0.01
        Hoeffding bound confidence (smaller values -> more eager splitting).
    tie_threshold : float, default=0.05
        Threshold below which competing splits are forced to break ties.
    max_feature : float, default=1.0
        Fraction of features to consider per split.
    grace_period_depth : int, default=-1
        Depth at which to change grace period behavior.
    eagerness : float, default=0.0
        Controls eagerness of splitting (implementation specific).
    bin_type : str, default="adaptive"
        Binning strategy for histograms.
    max_bins : int, default=32
        Maximum number of bins per feature for histogram-based splits.
    split_criterion : str, default="vr_rel"
        Split criterion identifier.
    random_state : int, optional
        Random seed for reproducibility.
    pruning_strategy : {"old_egbdt", "new_egbdt", "weighted_ratio"}, default="weighted_ratio"
        Strategy for online pruning.
    verbose : bool, default=True
        If True, display a progress bar during ``partial_fit``.

    Weighted-Ratio Controls
    -----------------------
    wr_fixed_multiplier : float, default=1.0
        Multiplier κ converting weighted ratio to removal count.
    wr_alpha, wr_beta, wr_gamma : float
        Weights for recent error ratios: α·e(t)/e(t-1) + β·e(t-2)/e(t-3) + γ·e(t-3)/e(t-4)
    wr_baseline : float, default=0.0
        Additive offset to the removal count.
    wr_min_keep : int, default=1
        Minimum number of trees to keep after pruning.
    wr_error_agg : {"ema", "raw"}, default="ema"
        Error aggregation method.
    wr_ema_decay : float, default=0.90
        EMA smoothing factor (closer to 1.0 = smoother).
    wr_gate_eps : float, default=0.02
        Only prune if weighted ratio > 1 + eps.
    wr_cooldown : int, default=32
        Minimum number of samples between pruning events.
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
        random_state: Optional[int] = None,
        pruning_strategy: str = "old_egbdt",
        verbose: bool = True,
        wr_fixed_multiplier: float = 1.0,
        wr_alpha: float = 1.0,
        wr_beta: float = 0.0,
        wr_gamma: float = 0.0,
        wr_baseline: float = 0.0,
        wr_min_keep: int = 1,
        wr_error_agg: str = "ema",
        wr_ema_decay: float = 0.90,
        wr_gate_eps: float = 0.02,
        wr_cooldown: int = 32,
    ) -> None:

        # ---- base parameters ----
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
        self.pruning_strategy = pruning_strategy
        self.random_state = random_state
        self.verbose = verbose

        # ---- weighted-ratio pruning parameters ----
        self.wr_fixed_multiplier = float(wr_fixed_multiplier)
        self.wr_alpha = float(wr_alpha)
        self.wr_beta = float(wr_beta)
        self.wr_gamma = float(wr_gamma)
        self.wr_baseline = float(wr_baseline)
        self.wr_min_keep = int(max(1, wr_min_keep))
        self.wr_error_agg = wr_error_agg
        self.wr_ema_decay = float(wr_ema_decay)
        self.wr_gate_eps = float(wr_gate_eps)
        self.wr_cooldown = int(wr_cooldown)

        # ---- internal state for weighted-ratio logic ----
        self._wr_err_hist: list[float] = []  # last 4 aggregated errors
        self._wr_ema: Optional[float] = None
        self._wr_step: int = 0
        self._wr_last_prune_step: int = -10**9

        # ---- adaptive progress bar ----
        try:
            from IPython import get_ipython  # type: ignore
            ipython = get_ipython()
            if ipython is not None and "IPKernelApp" in getattr(ipython, "config", {}):
                self.tqdm = tqdm.notebook.tqdm  # type: ignore[attr-defined]
            else:
                self.tqdm = tqdm.tqdm
        except Exception:  # fallback if tqdm or IPython unavailable
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
                random_state=self.random_state,
            )
            self.chain.append(new_node)

        if self.pruning_strategy == "old_egbdt":   # Index-based pruning (eGBDT)
            for n, node in enumerate(self.chain):
                y_pred = node.predict(X) * (
                    1.0 if n == 0 else self.learning_rate
                )  # first node is not multiplied by learning rate
                y_pred_strong += y_pred
                residual = y - y_pred_strong

                if node.is_filled():
                    error_rate = node.measure_error(y, y_pred_strong)
                    if error_rate < prev_pruning_error_rate:
                        pruning_index = n
                        prev_pruning_error_rate = error_rate
                    else:
                        break
                else:
                    node.partial_fit([X], residual)

            if pruning_index is not None:
                self.chain = self.chain[: pruning_index + 1]
        
        elif self.pruning_strategy == "new_egbdt":  # New index-based pruning (eGBDT)
            best_index = None
            best_error = float("inf")

            for n, node in enumerate(self.chain):
                # Predict contribution
                y_pred = node.predict(X) * (1.0 if n == 0 else self.learning_rate)
                y_pred_strong += y_pred
                residual = y - y_pred_strong

                if node.is_filled():
                    error_rate = node.measure_error(y, y_pred_strong)

                    # Track the *best prefix index* over the whole chain
                    if error_rate < best_error:
                        best_error = error_rate
                        best_index = n
                else:
                    # If not filled, still update learner incrementally
                    node.partial_fit([X], residual)
            
            # After scanning full chain, prune to the index with minimum error
            if best_index is not None:
                self.chain = self.chain[: best_index + 1]
        
        elif self.pruning_strategy == "weighted_ratio":
            # NEW: weighted-ratio + contribution pruning (and incubation inside)
            self._update_weighted_ratio_contrib(X, y)
            return  # already incubated/updated inside



        hatched_node = self.incubate(X, residual)
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
                random_state=self.random_state,
            )
            self.incubating_node.partial_fit([X], residual)
            return None

        self.incubating_node.partial_fit([X], residual)
        if self.incubating_node.is_filled():
            hatched = self.incubating_node
            self.incubating_node = None
            return hatched
        return None
    
    def _loss(self, y_true: float, y_pred: float) -> float:
            return abs(np.sqrt(y_true**2 - y_pred**2))
    
    def _update_weighted_ratio_contrib(self, X: Sequence[float], y: float) -> None:
        # ---- pre-update prediction and *instant* error ----
        y_pred_full = 0.0
        for n, node in enumerate(self.chain):
            y_pred_full += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
        e_inst = abs(y - y_pred_full)

        # ---- aggregate error (EMA or raw) ----
        if self.wr_error_agg == "ema":
            if self._wr_ema is None:
                self._wr_ema = e_inst
            else:
                d = float(self.wr_ema_decay)
                d = min(max(d, 0.0), 0.9999)  # safety
                self._wr_ema = d * self._wr_ema + (1.0 - d) * e_inst
            e_agg = self._wr_ema
        else:
            e_agg = e_inst

        # ---- decide removal from *aggregated* ratios with gating/cooldown ----
        removal = 0
        can_prune = (
            len(self._wr_err_hist) >= 3  # plus current makes 4 points
            and (self._wr_step - self._wr_last_prune_step) >= int(self.wr_cooldown)
        )

        if can_prune:
            e_t4, e_t3, e_t2 = self._wr_err_hist[-3], self._wr_err_hist[-2], self._wr_err_hist[-1]
            e_t1 = e_agg  # current aggregated error

            r1 = (e_t3 / e_t4) if e_t4 > 0 else 1.0  # e(t-3)/e(t-4)
            r2 = (e_t2 / e_t3) if e_t3 > 0 else 1.0  # e(t-2)/e(t-3)
            r3 = (e_t1 / e_t2) if e_t2 > 0 else 1.0  # e(t)/e(t-1)

            weighted_ratio = self.wr_gamma * r1 + self.wr_beta * r2 + self.wr_alpha * r3

            # gate: only act on genuine degradation
            if weighted_ratio > (1.0 + float(self.wr_gate_eps)):
                removal_val = self.wr_fixed_multiplier * weighted_ratio + self.wr_baseline
                removal = int(round(removal_val))
                max_removal = max(0, len(self.chain) - int(max(1, self.wr_min_keep)))
                removal = max(0, min(removal, max_removal))
            else:
                removal = 0

            if self.verbose:
                print(
                    f"[WR] agg ratios r1={r1:.4f}, r2={r2:.4f}, r3={r3:.4f}, "
                    f"weighted={weighted_ratio:.4f} → removal={removal}"
                )
        else:
            if self.verbose:
                reason = "history<4" if len(self._wr_err_hist) < 3 else f"cooldown<{self.wr_cooldown}"
                print(f"[WR] no prune ({reason})")

        # ---- prune worst-contribution trees on current sample (if any) ----
        if removal > 0 and len(self.chain) > 0:
            contribs: list[float] = []
            partial = 0.0
            err_before = abs(y - partial)
            for i, node in enumerate(self.chain):
                incr = node.predict(X) * (1.0 if i == 0 else self.learning_rate)
                new_pred = partial + incr
                new_err = abs(y - new_pred)
                contribs.append(err_before - new_err)
                partial, err_before = new_pred, new_err

            worst_idx = np.argsort(np.asarray(contribs))[:removal].tolist()
            for idx in sorted(worst_idx, reverse=True):
                del self.chain[idx]

            self._wr_last_prune_step = self._wr_step
            if self.verbose:
                print(f"[WR] pruned={worst_idx} | trees={len(self.chain)}")

        # ---- update not-filled nodes on residual & incubate ----
        y_pred_strong = 0.0
        for n, node in enumerate(self.chain):
            incr = node.predict(X) * (1.0 if n == 0 else self.learning_rate)
            y_pred_strong += incr
            residual = y - y_pred_strong
            if not node.is_filled():
                node.partial_fit([X], residual)

        residual_final = y - y_pred_strong
        hatched_node = self.incubate(X, residual_final)
        if hatched_node is not None and len(self.chain) < self.length:
            self.chain.append(hatched_node)

        # ---- push aggregated error into rolling history (keep last 4) ----
        self._wr_err_hist.append(e_agg)
        if len(self._wr_err_hist) > 4:
            self._wr_err_hist.pop(0)

        self._wr_step += 1



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
