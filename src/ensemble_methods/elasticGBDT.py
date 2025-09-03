"""Elastic/prunable Gradient Boosting utilities for incremental-like updates.

Wraps a scikit-learn GradientBoostingRegressor and provides several update
strategies (index-based, contribution-based, ratio-based, weighted-ratio, etc.).
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error  # Default error metric if none provided

"""
TODO notes:
- We might need a rolling forecast (e.g., always keep a forecast of 10 datapoints).
- Decide whether new predictions overwrite the oldest, or if we keep a sliding window that
  drops the earliest prediction as a new one arrives.
- This is particularly relevant for short forecast when the batch is too smalle to do the update
- This problem is known as "tuning the batch size" in online learning literature.
"""


class ElasticGBDT:
    """
    Elastic Gradient Boosting Decision Tree (eGBDT) using scikit-learn's GradientBoostingRegressor.

    This class supports two main pruning methods:
      - Index-Based pruning: Selects a subset of trees based on sequential error improvement.
      - Contribution-Based pruning: Removes trees that contribute little to error reduction.

    It also supports incremental tree addition.
    """

    def __init__(
        self,
        pretrained_model: GradientBoostingRegressor,
        y_train: Iterable[float],
        m_threshold: int = 40,
        alpha: Optional[float] = None,
        max_depth: int = 5,
        random_state: int = 42,
        error_metric: Optional[Callable[[Iterable[float], Iterable[float]], float]] = None,
        adapt: bool = False,
    ) -> None:
        # Set the threshold below which too few trees trigger full retraining.
        self.m_threshold = m_threshold
        self.max_depth = max_depth
        self.random_state = random_state
        self.adapt = adapt

        # Use provided error_metric or default to mean absolute error.
        self.error_metric = error_metric if error_metric is not None else mean_absolute_error

        # Use the pretrained model's learning rate as alpha (shrinkage) if not provided.
        self.alpha = pretrained_model.learning_rate if alpha is None else alpha

        # The initial prediction is the mean of the training target.
        self.initial_pred = np.mean(y_train)

        # Extract trees from the pretrained model.
        # We assume a single-output model, so we take the first column of the estimator array.
        self.trees = [estimator for estimator in pretrained_model.estimators_[:, 0]]
        print(f"Loaded {len(self.trees)} trees from the pretrained model. m_threshold={self.m_threshold}")

    def update_index_based(self, X_new: pd.DataFrame | np.ndarray, y_new: Iterable[float]) -> "ElasticGBDT":
        """
        Update using the index-based strategy.

        Finds the tree index that minimizes the error when trees are added sequentially.
        If this index is below `m_threshold`, performs full retraining; otherwise prunes
        trees after that index.

        Parameters
        ----------
        X_new : pandas.DataFrame or numpy.ndarray
            Feature matrix of the new batch.
        y_new : Iterable of float
            Targets for the new batch.

        Returns
        -------
        ElasticGBDT
            Updated instance.
        """
        n_trees = len(self.trees)
        if n_trees == 0:
            print("[Index-Based] No trees; retraining.")
            self._retrain(X_new, y_new)
            return self

        partial_pred = np.full(y_new.shape, self.initial_pred)
        best_error = self.error_metric(y_new, partial_pred)
        best_i = 0

        for i in range(n_trees):
            # Predict with each tree and update the cumulative prediction.
            partial_pred += self.alpha * self.trees[i].predict(X_new)
            current_error = self.error_metric(y_new, partial_pred)
            if current_error < best_error:
                best_error = current_error
                best_i = i

        print(f"[Index-Based] Best tree index = {best_i}, Error = {best_error:.4f}")

        if best_i < self.m_threshold:
            print(f"[Index-Based] {best_i} < {self.m_threshold} -> Full retraining.")
            self._retrain(X_new, y_new)
        else:
            pruned_count = n_trees - (best_i + 1)
            self.trees = self.trees[: best_i + 1]
            print(f"[Index-Based] Pruned {pruned_count} trees. Now {len(self.trees)} remain.")
        return self

    def update_contribution_based(
        self,
        X_new: pd.DataFrame | np.ndarray,
        y_new: Iterable[float],
        threshold_difference: float = -5.0,
    ) -> "ElasticGBDT":
        """
        Update using the contribution-based strategy.

        Computes the incremental improvement contributed by each tree and prunes
        trees whose improvement is below a threshold. Triggers full retraining if
        remaining tree count drops below `m_threshold`.

        Parameters
        ----------
        X_new : pandas.DataFrame or numpy.ndarray
            Feature matrix of the new batch.
        y_new : Iterable of float
            Targets for the new batch.
        threshold_difference : float, optional
            Minimum improvement required to keep a tree. Defaults to -5.0.

        Returns
        -------
        ElasticGBDT
            Updated instance.
        """
        n_trees = len(self.trees)
        if n_trees == 0:
            print("[Contribution-Based] No trees; retraining.")
            self._retrain(X_new, y_new)
            return self

        partial_pred = np.full(y_new.shape, self.initial_pred)
        old_error = self.error_metric(y_new, partial_pred)
        contributions = []

        for i in range(n_trees):
            new_pred = partial_pred + self.alpha * self.trees[i].predict(X_new)
            new_error = self.error_metric(y_new, new_pred)
            improvement = old_error - new_error
            contributions.append((improvement, i))
            partial_pred = new_pred
            old_error = new_error

        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for (imp, idx) in contributions if imp < threshold_difference]
        if not to_remove:
            print(f"[Contribution-Based] No tree below {threshold_difference}. No pruning.")
            return self

        print(f"[Contribution-Based] Removing {len(to_remove)} trees (improvement < {threshold_difference}).")
        for idx in sorted(to_remove, reverse=True):
            self.trees.pop(idx)

        if len(self.trees) < self.m_threshold:
            print(
                f"[Contribution-Based] Only {len(self.trees)} trees remain (< {self.m_threshold}) -> Full retraining."
            )
            self._retrain(X_new, y_new)
        else:
            print(f"[Contribution-Based] Now {len(self.trees)} trees remain.")
        return self

    def update_ratio(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[float],
        fixed_multiplier: float = 1.0,
        *,
        adapt: bool = False,
    ) -> "ElasticGBDT":
        """
        Update the ensemble using the ratio-based strategy.

        Computes the ratio between the current error and the last stored error, converts
        it to a removal count (scaled by ``fixed_multiplier``), then prunes the trees with
        the smallest incremental contributions. When ``adapt=True`` and the ratio is below
        1, no trees are removed (adaptive safeguard).

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix of the new batch.
        y : Iterable of float
            Targets for the new batch.
        fixed_multiplier : float, optional
            Multiplier applied to the error ratio to determine the removal count. Defaults to 1.0.
        adapt : bool, optional
            If ``True``, when ratio < 1, do not remove trees (i.e., keep the ensemble size).
            Defaults to ``False``.

        Returns
        -------
        ElasticGBDT
            Updated instance.

        Notes
        -----
        - Requires ``self.error_metric`` (callable) and ``self.error_history`` (list-like).
        - Assumes:
            * ``self.initial_pred`` : float baseline prediction
            * ``self.alpha``        : float learning rate / scaling for tree outputs
            * ``self.trees``        : list of fitted regressors with ``predict(X)``

        """
        # --- 1) Compute current error with the full ensemble ---
        # Start from the initial/bias prediction.
        y = np.asarray(list(y), dtype=float)
        partial_pred = np.full(y.shape, self.initial_pred, dtype=float)
        for tree in self.trees:
            partial_pred += self.alpha * tree.predict(X)

        final_error = self.error_metric(y, partial_pred)

        # --- 2) Compute error ratio versus last error ---
        # If no history yet, store and return early (nothing to compare).
        if not getattr(self, "error_history", None):
            self.error_history = [float(final_error)]
            if hasattr(self, "last_error"):
                self.last_error = float(final_error)
            return self

        last_error = float(self.error_history[-1])
        # Guard against divide-by-zero or missing last_error.
        if last_error <= 0.0:
            ratio = 1.0
        else:
            ratio = float(final_error) / last_error

        # --- 3) Translate ratio -> removal_count with optional adaptation ---
        removal_count = int(round(fixed_multiplier * ratio))
        if adapt and ratio < 1.0:
            # Adaptive safeguard: if we improved (ratio < 1), keep all trees.
            removal_count = 0

        # Bound the removal count to a safe range.
        n_trees = len(self.trees)
        removal_count = max(0, min(removal_count, max(0, n_trees - 1)))

        # --- 4) If nothing to remove, just record error and exit ---
        if removal_count == 0:
            # Record error for future ratios.
            self.error_history.append(float(final_error))
            if hasattr(self, "last_error"):
                self.last_error = float(final_error)
            else:
                self.last_error = float(final_error)
            # (Optional) debug print wrapped to satisfy E501
            print(
                "[Ratio Update] No pruning performed. "
                f"final_error={final_error:.3f}, ratio={ratio:.3f}, n_trees={n_trees}"
            )
            return self

        # --- 5) Compute per-tree incremental contributions and prune the smallest ---
        # Rebuild incremental predictions to measure each tree's marginal improvement.
        contribs: list[float] = []
        running_pred = np.full(y.shape, self.initial_pred, dtype=float)
        prev_err = self.error_metric(y, running_pred)
        for tree in self.trees:
            running_pred = running_pred + self.alpha * tree.predict(X)
            new_err = self.error_metric(y, running_pred)
            # Contribution as "improvement" (positive means error decreased).
            contribs.append(prev_err - new_err)
            prev_err = new_err

        # Identify indices of the k smallest contributions.
        # If contributions are improvements, "smallest" means least helpful (possibly negative).
        k = removal_count
        if k > 0:
            idx_sorted = np.argsort(contribs)  # ascending: worst first
            to_remove = set(idx_sorted[:k].tolist())
            # Keep trees not selected for removal (stable order).
            self.trees = [t for i, t in enumerate(self.trees) if i not in to_remove]

        # --- 6) Record error for history and exit ---
        self.error_history.append(float(final_error))
        self.last_error = float(final_error)

        print(
            "[Ratio Update] "
            f"final_error={final_error:.3f}, ratio={ratio:.3f}, "
            f"removed={removal_count}, remaining={len(self.trees)}"
        )
        return self

    def update_contribution_weighted_ratio(
        self,
        X: pd.DataFrame | np.ndarray,
        y: Iterable[float],
        fixed_multiplier: float,
        gamma_multiplier: float,
        beta_multiplier: float,
        alpha_multiplier: float,
        baseline: float = 0.0,
    ) -> "ElasticGBDT":
        """
        Update using the weighted-ratio strategy.

        Combines multiple historical error ratios using weights (gamma, beta, alpha),
        converts the weighted ratio to a removal count, and prunes least-contributing
        trees.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix of the new batch.
        y : Iterable of float
            Targets for the new batch.
        fixed_multiplier : float
            Global multiplier for the weighted ratio.
        gamma_multiplier : float
            Weight for the oldest ratio term.
        beta_multiplier : float
            Weight for the middle ratio term.
        alpha_multiplier : float
            Weight for the most recent ratio term.
        baseline : float, optional
            Baseline added to the weighted ratio before scaling.

        Returns
        -------
        ElasticGBDT
            Updated instance.
        """
        final_error = self.error_metric(y, self.predict(X))

        if not hasattr(self, "error_history"):
            self.error_history = []
        if len(self.error_history) < 4:
            self.error_history.append(final_error)
            print(f"[Weighted Ratio] History length {len(self.error_history)} < 4; no tree deletion.")
            return self

        e_t4 = self.error_history[-4]
        e_t3 = self.error_history[-3]
        e_t2 = self.error_history[-2]
        e_t1 = self.error_history[-1]

        ratio1 = e_t3 / e_t4 if e_t4 != 0 else 1.0
        ratio2 = e_t2 / e_t3 if e_t3 != 0 else 1.0
        ratio3 = final_error / e_t1 if e_t1 != 0 else 1.0

        weighted_ratio = gamma_multiplier * ratio1 + beta_multiplier * ratio2 + alpha_multiplier * ratio3
        removal_count = round(fixed_multiplier * weighted_ratio + baseline)
        removal_count = max(removal_count, 0)
        removal_count = min(removal_count, len(self.trees))

        print(
            "[Weighted Ratio] "
            f"e(t-4)={e_t4:.3f}, e(t-3)={e_t3:.3f}, e(t-2)={e_t2:.3f}, "
            f"e(t-1)={e_t1:.3f}, e(t)={final_error:.3f}"
        )
        print(
            "[Weighted Ratio Details] "
            f"ratio1={ratio1:.3f}, ratio2={ratio2:.3f}, ratio3={ratio3:.3f}, "
            f"weighted_ratio={weighted_ratio:.3f}, removal_count={removal_count}"
        )

        partial_pred = np.full(y.shape, self.initial_pred)
        contributions = []
        error_before = self.error_metric(y, partial_pred)
        for i, tree in enumerate(self.trees):
            tree_pred = tree.predict(X)
            new_pred = partial_pred + self.alpha * tree_pred
            new_error = self.error_metric(y, new_pred)
            improvement = error_before - new_error
            contributions.append((improvement, i))
            partial_pred = new_pred
            error_before = new_error

        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for _, idx in contributions[:removal_count]]
        for idx in sorted(to_remove, reverse=True):
            self.trees.pop(idx)

        # Update error history to keep a fixed length of 4.
        self.error_history.append(final_error)
        if len(self.error_history) > 4:
            self.error_history.pop(0)

        return self

    def build_incremental_trees(
        self,
        X_new: pd.DataFrame | np.ndarray,
        Y_residuals_left: np.ndarray,
        number_of_incremental_trees: int = 5,
    ) -> None:
        """
        Build additional trees incrementally.

        Trains `number_of_incremental_trees` on current residuals and appends
        them to the ensemble, updating residuals after each tree.

        Parameters
        ----------
        X_new : pandas.DataFrame or numpy.ndarray
            Feature matrix for training additional trees.
        Y_residuals_left : numpy.ndarray
            Residual vector to fit.
        number_of_incremental_trees : int, optional
            Number of trees to add. Defaults to 5.

        Returns
        -------
        None
        """
        from sklearn.tree import DecisionTreeRegressor

        for _ in range(number_of_incremental_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + len(self.trees))
            tree.fit(X_new, Y_residuals_left)
            self.trees.append(tree)
            # Update the residuals using the newly added tree.
            Y_residuals_left -= self.alpha * tree.predict(X_new)

    def _retrain(self, X: pd.DataFrame | np.ndarray, y: Iterable[float], num_trees: int = 40) -> None:
        """
        Retrain the underlying GradientBoostingRegressor.

        Resets the ensemble, refits a fresh model on `(X, y)`, updates the initial
        prediction, and rebuilds the internal tree list.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Full feature matrix for retraining.
        y : Iterable of float
            Target vector for retraining.
        num_trees : int, optional
            Number of estimators for the new model. Defaults to 40.

        Returns
        -------
        None
        """
        print("Retraining triggered.")

        # Reset the initial prediction.
        self.initial_pred = np.mean(y)

        # Train a new GradientBoostingRegressor with the desired number of trees.
        new_gbdt = GradientBoostingRegressor(
            n_estimators=num_trees, learning_rate=self.alpha, max_depth=self.max_depth, random_state=self.random_state
        )
        new_gbdt.fit(X, y)

        # Extract trees from the new model.
        self.trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        print(f"Retrained with {len(self.trees)} trees.")
        return self

    def get_number_of_trees(self) -> int:
        """Return the current number of trees."""
        return len(self.trees)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """
        Generate predictions.

        Starts from the initial prediction and adds each tree’s contribution,
        scaled by `alpha`.

        Parameters
        ----------
        X : pandas.DataFrame or numpy.ndarray
            Feature matrix to predict on.

        Returns
        -------
        numpy.ndarray
            Predicted values with shape (n_samples,).
        """
        predictions = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            predictions += self.alpha * tree.predict(X)
        return predictions
