"""Elastic/prunable Gradient Boosting with three internal update strategies.

This estimator wraps scikit-learn's GradientBoostingRegressor and supports
online pruning updates driven by one of three strategies selected via the
``update_strategy`` hyperparameter:

- "index":        find the best index (stage) minimizing error; prune after it
- "contribution": prune trees with contributions below a threshold
- "ratio":        convert error ratio to a removal count (with optional adapt)

It provides a scikit-learn-like API so it can be used by the project's
training/tuning framework:
- ``fit``: initial batch training
- ``partial_fit``: incremental updates during walk-forward evaluation
- ``predict``: predictions through the current (possibly pruned) ensemble
- ``get_params`` / ``set_params``: hyperparameter tuning support

TODO notes:
- We might need a rolling forecast (e.g., always keep a forecast of 10 datapoints).
- Decide whether new predictions overwrite the oldest, or if we keep a sliding window that
  drops the earliest prediction as a new one arrives.
- This is particularly relevant for short forecast when the batch is too smalle to do the update
- This problem is known as "tuning the batch size" in online learning literature.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor  # add this



# ----------------------- helpers -----------------------


def _default_error_metric(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """Return MAE between y_true and y_pred."""
    y_true = np.asarray(list(y_true), dtype=float)
    y_pred = np.asarray(list(y_pred), dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


# -------------------- main estimator --------------------


class ElasticGBDT(BaseEstimator, RegressorMixin):
    """
    Gradient Boosting with elastic (prunable) online update.

    Parameters
    ----------
    # Base GBDT (initial fit)
    n_estimators : int, default=150
        Number of trees for the underlying GBDT at initial fit.
    learning_rate : float, default=0.05
        Learning rate for the underlying GBDT.
    max_depth : int, default=3
        Max depth for base trees.
    subsample : float, default=1.0
        Subsample for GBDT.
    random_state : int, default=42
        RNG seed forwarded to GBDT.

    # Elastic controls (apply after initial fit)
    alpha : float or None, default=None
        Optional scaling applied to each tree's contribution in our accumulator.
        If ``None``, uses ``learning_rate``.
    m_threshold : int, default=40
        Minimum number of trees to keep after pruning. If pruning would reduce
        below this value, the update falls back to a full retraining.
    update_strategy : {"index", "contribution", "ratio"}, default="ratio"
        Strategy used by ``partial_fit`` for online updates.
    # Strategy-specific knobs
    contrib_threshold : float, default=-5.0
        Contribution threshold for "contribution" strategy. Trees with marginal
        improvement < threshold are pruned (improvement measured as
        previous_error - new_error on the mini-batch).
    fixed_multiplier : float, default=1.0
        Multiplier used to convert the error ratio to a removal count for
        "ratio" strategy.
    adapt : bool, default=False
        If True and the error ratio < 1.0 (improvement), "ratio" strategy
        removes 0 trees (adaptive safeguard).
    error_metric : Callable or None, default=None
        Error function to drive updates. Defaults to MAE.

    Notes
    -----
    The project's incremental loop calls ``partial_fit``:
    - First on the *whole* training batch (we call ``fit`` there).
    - Then on single validation samples or small slices (we prune).
    """

    def __init__(
        self,
        # base GBDT
        n_estimators: int = 150,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        subsample: float = 1.0,
        random_state: int = 42,
        *,
        # elastic controls
        alpha: Optional[float] = None,
        m_threshold: int = 40,
        update_strategy: str = "ratio",
        # strategy-specific
        contrib_threshold: float = -5.0,
        fixed_multiplier: float = 1.0,
        adapt: bool = False,
        error_metric: Optional[Callable[[Iterable[float], Iterable[float]], float]] = None,
        prune_on_fit: bool = True,
        prune_holdout_frac: float = 0.1,
        ratio_history: int = 1,
        ratio_decay: float = 0.5,
        grow_back: bool = True,
        backfill_estimator=None,                # class (e.g., DecisionTreeRegressor)
        backfill_params: Optional[dict] = None,
        preserve_ensemble: bool = False,
        verbose: int = 0,

    ) -> None:
        # base params
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.subsample = subsample
        self.random_state = random_state

        # elastic params
        self.alpha = alpha
        self.m_threshold = m_threshold
        self.update_strategy = update_strategy
        self.contrib_threshold = contrib_threshold
        self.fixed_multiplier = fixed_multiplier
        self.adapt = adapt
        self.error_metric = error_metric or _default_error_metric
        self.ratio_history = ratio_history  # how many ratios to include
        self.ratio_decay = ratio_decay      # decay multiplier

        self.grow_back = grow_back
        self.backfill_estimator = backfill_estimator
        self.backfill_params = backfill_params or {}

        # learned state
        self._gbdt: Optional[GradientBoostingRegressor] = None
        self._trees: list[Any] = []
        self._initial_pred: float = 0.0
        self._last_error: Optional[float] = None
        self._error_history: list[float] = []

        # small buffer so single-sample partial_fit can batch if desired
        self._buffer_X: list[np.ndarray] = []
        self._buffer_y: list[float] = []
        self._buffer_max: int = 1  # keep 1 to align with horizon-driven updates
        
        self.prune_on_fit = prune_on_fit
        self.prune_holdout_frac = prune_holdout_frac
        self.preserve_ensemble = preserve_ensemble

         # ---- runtime stats (for debugging/monitoring) ----
        self._last_pre_err: Optional[float] = None
        self._last_post_err: Optional[float] = None
        self._last_removed: int = 0
        self._last_grown: int = 0
        self._last_ratios: list[float] = []
        self._total_pruned: int = 0
        self._total_grown: int = 0
        self.verbose = verbose


    # small helper
    def _log(self, msg: str) -> None:
        if getattr(self, "verbose", 0):
            print(f"[ElasticGBDT] {msg}")

    # ------------------- sklearn API -------------------

    def fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[float]) -> "ElasticGBDT":
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(list(y), dtype=float)

        # -------- Case A: no model yet → full initial build --------
        if self._gbdt is None or len(self._trees) == 0:
            self._log("fit: Case A (initial full train)")
            self._gbdt = GradientBoostingRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                subsample=self.subsample,
                random_state=self.random_state,
            )
            self._gbdt.fit(X_arr, y_arr)
            self._extract_from_gbdt()

            if self.prune_on_fit:
                self._log("fit: prune_on_fit=True → applying update strategy on full batch")
                self._apply_update_strategy(X_arr, y_arr)

            y_pred = self.predict(X_arr)
            self._last_error = self.error_metric(y_arr, y_pred)
            self._log(f"fit: done | trees={len(self._trees)}")
            return self

        # -------- Case B: model exists --------
        if self.preserve_ensemble:
            # treat as in-place update (no fresh GBDT rebuild)
            self._log("fit: Case B (preserve_ensemble=True → in-place prune/grow on full batch)")
            self._apply_update_strategy(X_arr, y_arr)
            y_pred = self.predict(X_arr)
            self._last_error = self.error_metric(y_arr, y_pred)
            self._log(f"fit: done | trees={len(self._trees)}")
            return self

        # -------- Case C: stateless batch mode (fresh rebuild every fit) --------
        self._log("fit: Case C (preserve_ensemble=False → fresh GBDT rebuild)")
        self._gbdt = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._gbdt.fit(X_arr, y_arr)
        self._extract_from_gbdt()

        if self.prune_on_fit:
            self._log("fit: prune_on_fit=True → applying update strategy on full batch")
            self._apply_update_strategy(X_arr, y_arr)

        y_pred = self.predict(X_arr)
        self._last_error = self.error_metric(y_arr, y_pred)
        self._log(f"fit: done | trees={len(self._trees)}")
        return self



    def partial_fit(self, X: pd.DataFrame | np.ndarray, y: Iterable[float]) -> "ElasticGBDT":
        """
        Incrementally update the model using the selected strategy.

        In the framework, the first call during a split is the full training
        batch (we call ``fit``). Subsequent calls are usually single samples or
        small slices; we apply pruning according to ``update_strategy``.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            New samples.
        y : Iterable of float
            Targets for the new samples.

        Returns
        -------
        ElasticGBDT
            Updated instance.
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        y_arr = np.asarray(list(y), dtype=float)

        # First call: no state yet => do a full fit.
        if self._gbdt is None or len(self._trees) == 0:
            return self.fit(X_arr, y_arr)

        # Buffering (kept at 1 to match horizon updates).
        for i in range(X_arr.shape[0]):
            self._buffer_X.append(X_arr[i : i + 1])
            self._buffer_y.append(float(y_arr[i]))
            if len(self._buffer_X) >= self._buffer_max:
                Xb = np.vstack(self._buffer_X)
                yb = np.asarray(self._buffer_y, dtype=float)
                self._apply_update_strategy(Xb, yb)
                self._buffer_X.clear()
                self._buffer_y.clear()

        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions using the current pruned ensemble.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix to predict on.

        Returns
        -------
        numpy.ndarray
            Predicted values (n_samples,).
        """
        X_arr = X.values if isinstance(X, pd.DataFrame) else X
        self._ensure_fitted()

        lr = self.alpha if self.alpha is not None else self.learning_rate
        pred = np.full(X_arr.shape[0], self._initial_pred, dtype=float)
        for tree in self._trees:
            pred += lr * tree.predict(X_arr)
        return pred

    # These two make it tuneable by name (sklearn convention).
    def get_params(self, deep: bool = True) -> dict[str, Any]:
        return {
            # base
            "n_estimators": self.n_estimators,
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "random_state": self.random_state,
            # elastic
            "alpha": self.alpha,
            "m_threshold": self.m_threshold,
            "update_strategy": self.update_strategy,
            # per-strategy
            "contrib_threshold": self.contrib_threshold,
            "fixed_multiplier": self.fixed_multiplier,
            "adapt": self.adapt,
            "error_metric": self.error_metric,
            "prune_on_fit": self.prune_on_fit,
            "prune_holdout_frac": self.prune_holdout_frac,
            "grow_back": self.grow_back,
            "backfill_estimator": self.backfill_estimator,
            "backfill_params": self.backfill_params,
            "ratio_history": self.ratio_history,
            "ratio_decay": self.ratio_decay,
    }
        

    def set_params(self, **params: Any) -> "ElasticGBDT":
        for k, v in params.items():
            setattr(self, k, v)
        return self

    # ------------------- internal logic -------------------

    def _ensure_fitted(self) -> None:
        """Raise if predict/update is called before initial fit."""
        if self._gbdt is None or len(self._trees) == 0:
            msg = "ElasticGBDT is not fitted yet. Call fit/partial_fit first."
            raise RuntimeError(msg)

    def _extract_from_gbdt(self) -> None:
        """Extract initial prediction and trees from the fitted GBDT."""
        assert self._gbdt is not None

        # Bias term:
        try:
            init_const = float(np.ravel(self._gbdt.init_.constant_)[0])
        except Exception:
            init_const = 0.0
        self._initial_pred = init_const

        # Stages (estimators_ shape is (n_stages, 1) for regression)
        self._trees = [est[0] for est in self._gbdt.estimators_]

    def _apply_update_strategy(self, X: np.ndarray, y: np.ndarray) -> None:
        """Compute pre-batch error, prune by strategy, (optionally) grow back, then log post error."""
        self._ensure_fitted()
        lr = self.alpha if self.alpha is not None else self.learning_rate

        # ---- PRE-BATCH ERROR (history driver) ----
        pred_pre = np.full(y.shape, self._initial_pred, dtype=float)
        for t in self._trees:
            pred_pre += lr * t.predict(X)
        pre_err = self.error_metric(y, pred_pre)
        self._error_history.append(pre_err)
        self._last_pre_err = pre_err

        n_before = len(self._trees)

        # ---- STRATEGY ----
        strategy = self.update_strategy.lower()
        if strategy == "index":
            self._update_index(X, y)
        elif strategy == "contribution":
            self._update_contribution(X, y, threshold=self.contrib_threshold)
        elif strategy == "ratio":
            self._update_ratio(X, y)  # uses self._error_history (pre-only)
        else:
            raise ValueError("Unknown update_strategy. Use one of: 'index', 'contribution', 'ratio'.")

        pruned = max(0, n_before - len(self._trees))
        self._last_removed = pruned
        self._total_pruned += pruned

        grown = 0
        if self.grow_back and pruned > 0:
            grown = self._backfill_trees(X, y, pruned)
            self._last_grown = grown
            self._total_grown += grown

        # ---- POST-BATCH ERROR (monitoring only) ----
        pred_post = np.full(y.shape, self._initial_pred, dtype=float)
        for t in self._trees:
            pred_post += lr * t.predict(X)
        post_err = self.error_metric(y, pred_post)
        self._last_post_err = post_err

        # keep _last_error as the pre-batch error (driver)
        self._last_error = pre_err

        # ---- concise prints ----
        self._log(
            f"strategy={strategy} | pre_err={pre_err:.6f} -> post_err={post_err:.6f} | "
            f"pruned={pruned}, grown={grown}, trees: {n_before}→{len(self._trees)}"
        )


    # ------------- strategy: ratio (with adapt) -------------
    # ---------- strategy: index ----------
    def _update_index(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Prune to the best stage index (lowest error on this mini-batch).
        If the best index would leave fewer than `m_threshold` trees, retrain on (X, y).

        Side-effect free w.r.t. error history: does not read/write `_error_history` or `_last_error`.
        """
        self._ensure_fitted()
        lr = self.alpha if self.alpha is not None else self.learning_rate

        running = np.full(y.shape, self._initial_pred, dtype=float)
        best_err = self.error_metric(y, running)
        best_idx = -1

        for i, t in enumerate(self._trees):
            running = running + lr * t.predict(X)
            err = self.error_metric(y, running)
            if err <= best_err:
                best_err = err
                best_idx = i

        # If no improvement, do nothing (no pruning).
        if best_idx < 0:
            return

        keep_count = best_idx + 1
        if keep_count < max(1, self.m_threshold):
            # Too few trees would remain → safer to retrain on this batch.
            self._retrain(X, y)
            return

        # Keep up to the best index (inclusive).
        self._trees = self._trees[:keep_count]


    # ------ strategy: contribution threshold ------
    def _update_contribution(self, X: np.ndarray, y: np.ndarray, *, threshold: float) -> None:
        """
        Prune trees whose marginal improvement (prev_err - new_err) on this batch is < threshold.
        If pruning would leave fewer than `m_threshold` trees, retrain on (X, y).

        Side-effect free w.r.t. error history.
        """
        self._ensure_fitted()
        lr = self.alpha if self.alpha is not None else self.learning_rate

        contribs: list[float] = []
        running = np.full(y.shape, self._initial_pred, dtype=float)
        prev_err = self.error_metric(y, running)

        for t in self._trees:
            running = running + lr * t.predict(X)
            new_err = self.error_metric(y, running)
            contribs.append(prev_err - new_err)  # positive => helped
            prev_err = new_err

        keep_idx = [i for i, c in enumerate(contribs) if c >= threshold]

        # Always keep at least m_threshold trees; otherwise retrain.
        if len(keep_idx) < max(1, self.m_threshold):
            self._retrain(X, y)
            return

        self._trees = [self._trees[i] for i in keep_idx]


    # ------------- strategy: ratio (with adapt, weighted history) -------------
    def _update_ratio(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Prune count derived from weighted recent error ratios with guards:
        - clips/validates decay ∈ [0,1]
        - ignores non-positive or NaN/Inf ratios
        - enforces min-ensemble size (m_threshold)
        """
        self._ensure_fitted()

        if len(self._error_history) < 2:
            return

        # hyperparams & guards
        H = max(1, int(self.ratio_history))
        decay = float(self.ratio_decay)
        if not np.isfinite(decay) or decay < 0.0:
            decay = 0.0
        if decay > 1.0:
            decay = 1.0

        alpha = float(self.fixed_multiplier)
        if not np.isfinite(alpha):
            alpha = 1.0

        adapt = bool(self.adapt)

        # collect up to H most-recent ratios (newest first)
        ratios: list[float] = []
        n_hist = min(len(self._error_history) - 1, H)
        for k in range(1, n_hist + 1):
            err_curr = self._error_history[-k]       # e_{t-k+1}
            err_prev = self._error_history[-k - 1]   # e_{t-k}
            if err_prev <= 0:
                continue
            r = float(err_curr) / float(err_prev)
            if np.isfinite(r) and r >= 0:  # ignore NaN/Inf/negatives
                ratios.append(r)

        self._last_ratios = ratios[:]  # store for inspection

        if not ratios:
            # no valid ratios -> nothing to do
            return

        # adaptive safeguard on latest ratio
        latest_prev = self._error_history[-2]
        latest_curr = self._error_history[-1]
        if adapt and latest_prev > 0 and (latest_curr / latest_prev) < 1.0:
            self._last_removed = 0
            return

        # weighted sum: alpha * Σ decay^(k-1) * ratio_k
        removal_val = 0.0
        for idx, r in enumerate(ratios, start=1):
            removal_val += alpha * (decay ** (idx - 1)) * r

        # turn into integer count, then clamp to preserve min ensemble
        removal = int(round(removal_val))
        n = len(self._trees)
        min_keep = max(1, int(self.m_threshold))
        max_removal = max(0, n - min_keep)
        removal = max(0, min(removal, max_removal))

        if removal == 0:
            self._log(
                f"[ratio] no-prune: removal_val≈{removal_val:.3f}, "
                f"ratios={np.round(ratios, 4).tolist()}"
            )
            return

        # compute marginal contributions on this batch and drop the worst `removal`
        lr = self.alpha if self.alpha is not None else self.learning_rate
        contribs: list[float] = []
        running = np.full(y.shape, self._initial_pred, dtype=float)
        prev_err = self.error_metric(y, running)
        for t in self._trees:
            running = running + lr * t.predict(X)
            new_err = self.error_metric(y, running)
            contribs.append(prev_err - new_err)  # positive => helped
            prev_err = new_err

        contribs_np = np.asarray(contribs, dtype=float)
        # safety on weird numerics
        contribs_np = np.where(np.isfinite(contribs_np), contribs_np, -np.inf)

        worst_idx = np.argsort(contribs_np)[:removal].tolist()
        keep = [i for i in range(n) if i not in set(worst_idx)]
        keep_count = len(keep)

        # final safeguard before applying
        if keep_count < min_keep:
            self._log(
                f"[ratio] retrain triggered: requested removal={removal} would keep {keep_count} < m_threshold={min_keep}"
            )
            self._retrain(X, y)
            return

        # apply pruning
        self._trees = [self._trees[i] for i in keep]

        # brief stats print
        self._log(
            f"[ratio] removal={removal} (val≈{removal_val:.3f}, ratios={np.round(ratios,4).tolist()}, "
            f"decay={decay}, α={alpha})"
        )



    # ---------------- utility / maintenance ----------------

    def _make_backfill_estimator(self) -> BaseEstimator:
        est_cls = self.backfill_estimator or DecisionTreeRegressor
        default = {"max_depth": self.max_depth, "random_state": self.random_state}
        params = {**default, **(self.backfill_params or {})}
        return est_cls(**params)


    def _backfill_trees(self, X: np.ndarray, y: np.ndarray, k: int) -> int:
        if k <= 0:
            return 0
        lr = self.alpha if self.alpha is not None else self.learning_rate

        pred = np.full(y.shape, self._initial_pred, dtype=float)
        for t in self._trees:
            pred += lr * t.predict(X)

        grown = 0
        for _ in range(k):
            residual = y - pred
            base = self._make_backfill_estimator()
            base.fit(X, residual)
            self._trees.append(base)
            pred += lr * base.predict(X)
            grown += 1

        self._log(f"[grow-back] built {grown} tree(s)")
        return grown




    # ---------------- utility / maintenance ----------------

    def _retrain(self, X: np.ndarray, y: np.ndarray) -> None:
        self._gbdt = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            random_state=self.random_state,
        )
        self._gbdt.fit(X, y)
        self._extract_from_gbdt()
        y_pred = self.predict(X)
        err = self.error_metric(y, y_pred)
        self._error_history.append(err)
        self._last_error = err

    def get_number_of_trees(self) -> int:
        return len(self._trees)