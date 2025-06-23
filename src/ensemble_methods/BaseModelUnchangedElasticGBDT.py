import numpy as np
from sklearn.metrics import \
    mean_absolute_error  # Default error metric if none provided


class HybridElasticGBDT:
    """
    Hybrid Elastic Gradient Boosting Decision Tree (Hybrid eGBDT)

    This class encapsulates a fixed base model (trained on a large dataset)
    whose trees are stored in `base_trees` and remain unchanged. In addition,
    it maintains a set of extra (incremental) trees in `incremental_trees` that
    are built and updated during deployment to capture short-term concept drifts.

    The final prediction is computed as:
    
        prediction = base_prediction + incremental_prediction

    where:
      - base_prediction is computed using the base trees and a base initial prediction,
      - incremental_prediction is computed from the extra trees (with an initial value of 0).
      
    NEW: Instead of pre-building the incremental trees, we build them gradually.
    For example, during the first few batches, if no incremental trees exist,
    we add a fixed number (e.g. 2 trees) per update.
    """

    def __init__(self, base_model, y_train, m_threshold=40, alpha=None, 
                 max_depth=5, random_state=42, error_metric=None, warmup_trees_per_batch=2):
        # Set the threshold below which too few incremental trees trigger full retraining.
        self.m_threshold = m_threshold
        self.max_depth = max_depth
        self.random_state = random_state
        self.warmup_trees_per_batch = warmup_trees_per_batch

        # Use provided error_metric or default to mean absolute error.
        self.error_metric = error_metric if error_metric is not None else mean_absolute_error

        # Use the base model's learning rate as alpha if not provided.
        self.alpha = base_model.learning_rate if alpha is None else alpha

        # The base initial prediction is the mean of the training target.
        self.base_initial_pred = np.mean(y_train)
        # For the incremental part, we assume that it is learning residuals.
        self.incremental_initial_pred = 0.0

        # Extract and store the base trees from the pretrained model.
        self.base_trees = [estimator for estimator in base_model.estimators_[:, 0]]
        # Initially, no incremental trees are built.
        self.incremental_trees = []

        print(f"Loaded {len(self.base_trees)} base trees from the pretrained model. "
              f"m_threshold for incremental trees={self.m_threshold}")

    def _predict_base(self, X):
        predictions = np.full(X.shape[0], self.base_initial_pred)
        for tree in self.base_trees:
            predictions += self.alpha * tree.predict(X)
        return predictions

    def _predict_incremental(self, X):
        predictions = np.full(X.shape[0], self.incremental_initial_pred)
        for tree in self.incremental_trees:
            predictions += self.alpha * tree.predict(X)
        return predictions

    def predict(self, X):
        return self._predict_base(X) + self._predict_incremental(X)

    def get_total_number_of_trees(self):
        return len(self.base_trees) + len(self.incremental_trees)

    # ---------------------------
    # Modified Update Methods for Gradual Incremental Tree Building
    # ---------------------------
    def update_index_based(self, X_new, y_new):
        """
        Index-Based Update for incremental trees only.
        If no incremental trees exist (e.g. during the first few batches),
        add a fixed number (warmup_trees_per_batch) instead of full retraining.
        """
        base_pred = self._predict_base(X_new)
        residual = y_new - base_pred

        n_trees = len(self.incremental_trees)
        if n_trees == 0:
            print("[Index-Based] No incremental trees; adding warmup trees "
                  f"({self.warmup_trees_per_batch} trees).")
            self.build_incremental_trees(X_new, residual, number_of_incremental_trees=self.warmup_trees_per_batch)
            return self

        partial_pred = np.full(y_new.shape, self.incremental_initial_pred)
        best_error = self.error_metric(residual, partial_pred)
        best_i = 0

        for i in range(n_trees):
            partial_pred += self.alpha * self.incremental_trees[i].predict(X_new)
            current_error = self.error_metric(residual, partial_pred)
            if current_error < best_error:
                best_error = current_error
                best_i = i

        print(f"[Index-Based] Best incremental tree index = {best_i}, Error = {best_error:.4f}")
        if best_i < self.m_threshold:
            print(f"[Index-Based] {best_i} < {self.m_threshold} -> Full retraining of incremental trees.")
            self._retrain_incremental(X_new, residual)
        else:
            pruned_count = n_trees - (best_i + 1)
            self.incremental_trees = self.incremental_trees[:best_i + 1]
            print(f"[Index-Based] Pruned {pruned_count} incremental trees. Now {len(self.incremental_trees)} remain.")
        return self

    def update_contribution_based(self, X_new, y_new, threshold_difference=-5):
        """
        Contribution-Based Update for incremental trees only.
        If no incremental trees exist, add warmup trees.
        """
        base_pred = self._predict_base(X_new)
        residual = y_new - base_pred

        n_trees = len(self.incremental_trees)
        if n_trees == 0:
            print("[Contribution-Based] No incremental trees; adding warmup trees "
                  f"({self.warmup_trees_per_batch} trees).")
            self.build_incremental_trees(X_new, residual, number_of_incremental_trees=self.warmup_trees_per_batch)
            return self

        partial_pred = np.full(y_new.shape, self.incremental_initial_pred)
        old_error = self.error_metric(residual, partial_pred)
        contributions = []

        for i in range(n_trees):
            new_pred = partial_pred + self.alpha * self.incremental_trees[i].predict(X_new)
            new_error = self.error_metric(residual, new_pred)
            improvement = old_error - new_error
            contributions.append((improvement, i))
            partial_pred = new_pred
            old_error = new_error

        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for (imp, idx) in contributions if imp < threshold_difference]

        if not to_remove:
            print(f"[Contribution-Based] No incremental tree below {threshold_difference}. No pruning.")
            return self

        print(f"[Contribution-Based] Removing {len(to_remove)} incremental trees (improvement < {threshold_difference}).")
        for idx in sorted(to_remove, reverse=True):
            self.incremental_trees.pop(idx)

        if len(self.incremental_trees) < self.m_threshold:
            print(f"[Contribution-Based] Only {len(self.incremental_trees)} incremental trees remain (< {self.m_threshold}) -> Full retraining.")
            self._retrain_incremental(X_new, residual)
        else:
            print(f"[Contribution-Based] Now {len(self.incremental_trees)} incremental trees remain.")
        return self

    # (Other update methods such as update_contribution_ema, update_contribution_ratio_adapt, etc.,
    # can be similarly modified if needed to check for an empty incremental ensemble.)

    def build_incremental_trees(self, X_new, residual, number_of_incremental_trees=5):
        """
        Incrementally adds new trees to the incremental ensemble.
        """
        from sklearn.tree import DecisionTreeRegressor
        for _ in range(number_of_incremental_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, 
                                         random_state=self.random_state + len(self.incremental_trees))
            tree.fit(X_new, residual)
            self.incremental_trees.append(tree)
            # Update the residual using the newly added tree.
            residual -= self.alpha * tree.predict(X_new)
        print(f"Built {number_of_incremental_trees} incremental trees. Total incremental trees: {len(self.incremental_trees)}")
        return self

    def _retrain_incremental(self, X, residual, num_trees=10):
        """
        Full retraining for incremental trees only.
        (This method can be used after warmup if desired.)
        """
        from sklearn.ensemble import GradientBoostingRegressor
        print("Retraining incremental trees triggered.")
        new_gbdt = GradientBoostingRegressor(
            n_estimators=num_trees,
            learning_rate=self.alpha,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        new_gbdt.fit(X, residual)
        self.incremental_trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        print(f"Retrained incremental trees: {len(self.incremental_trees)} trees built.")
        return self
