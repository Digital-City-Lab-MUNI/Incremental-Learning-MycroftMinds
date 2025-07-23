import numpy as np
from sklearn.metrics import \
    mean_absolute_error  # Default error metric if none provided

# TODO: how to deal with very short forecasting horizon?, perhaps we need some sort of rolling forecast say always have a forecast of 10 datapoints
# but will this whole forecast be updated everytime we receive a new datapoint or will there be just a concatenation of the latest predictiction 
# while the earliest prediction is removed?
# how should this be made compatible with the update methods? 

class ElasticGBDT:
    """
    Elastic Gradient Boosting Decision Tree (eGBDT) using scikit-learn's GradientBoostingRegressor.
    
    This class supports two main pruning methods:
      - Index-Based pruning: Selects a subset of trees based on sequential error improvement.
      - Contribution-Based pruning: Removes trees that contribute little to error reduction.
    
    It also supports incremental tree addition.
    """

    def __init__(self, pretrained_model, y_train, m_threshold=40, alpha=None, max_depth=5, random_state=42, error_metric=None):
        # Set the threshold below which too few trees trigger full retraining.
        self.m_threshold = m_threshold
        self.max_depth = max_depth
        self.random_state = random_state
        
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

    def update_index_based(self, X_new, y_new):
        """
        Index-Based Update:
          - Iteratively adds trees to the prediction and tracks error improvement.
          - Finds the tree index where error is minimized.
          - If that index is less than the threshold, the model is retrained entirely.
          - Otherwise, trees after that index are pruned.
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
            self.trees = self.trees[:best_i + 1]
            print(f"[Index-Based] Pruned {pruned_count} trees. Now {len(self.trees)} remain.")
        return self

    def update_contribution_based(self, X_new, y_new, threshold_difference=-5):
        """
        Contribution-Based Update:
          - Computes the error improvement (contribution) for each tree added sequentially.
          - Prunes those trees whose improvement is below a specified threshold.
          - If the remaining trees fall below m_threshold, a full retraining is triggered.
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
            improvement = (old_error - new_error) / old_error
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
            print(f"[Contribution-Based] Only {len(self.trees)} trees remain (< {self.m_threshold}) -> Full retraining.")
            self._retrain(X_new, y_new)
        else:
            print(f"[Contribution-Based] Now {len(self.trees)} trees remain.")
        return self

    def update_ratio_based(self, X_new, y_new, fixed_multiplier=1.0):
        """
        Ratio-Based Update:
          - Computes the final error using all current trees.
          - Uses a stored last_error to compute the error ratio.
          - Determines a removal count based on the ratio, and then prunes trees with the smallest contributions.
        """
        partial_pred = np.full(y_new.shape, self.initial_pred)
        for tree in self.trees:
            partial_pred += self.alpha * tree.predict(X_new)
        final_error = self.error_metric(y_new, partial_pred)
        if not hasattr(self, 'last_error'):
            self.last_error = final_error
            ratio = 1.0
        else:
            ratio = final_error / self.last_error
            self.last_error = final_error

        base_removal = round(fixed_multiplier * ratio)
        removal_count = base_removal if ratio >= 1 else max(base_removal - 1, 0)
        removal_count = min(removal_count, len(self.trees))

        partial_pred = np.full(y_new.shape, self.initial_pred)
        contributions = []
        old_error = self.error_metric(y_new, partial_pred)
        for i, tree in enumerate(self.trees):
            new_pred = partial_pred + self.alpha * tree.predict(X_new)
            new_error = self.error_metric(y_new, new_pred)
            improvement = (old_error - new_error)   / old_error if old_error != 0 else 0
            contributions.append((improvement, i))
            partial_pred = new_pred
            old_error = new_error

        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for _, idx in contributions[:removal_count]]
        for idx in sorted(to_remove, reverse=True):
            self.trees.pop(idx)

        print(f"[Ratio-Based] final_error={final_error:.3f}, ratio={ratio:.3f}, removal_count={removal_count}")
        return self

    def update_contribution_ratio_adapt(self, X, y, fixed_multiplier):
        """
        Adaptive Ratio Strategy Update:
          - Similar to the ratio-based method but if the computed error ratio is below 1, no trees are removed.
          - (Alternatively, one could subtract 1 from the removal count when ratio < 1.)
        """
        initial_pred = np.full(y.shape, self.initial_pred)
        partial_pred = initial_pred.copy()
        contributions = []
        old_error = self.error_metric(y, initial_pred)
        for i, tree in enumerate(self.trees):
            tree_pred = tree.predict(X)
            new_pred = partial_pred + self.alpha * tree_pred
            new_error = self.error_metric(y, new_pred)
            improvement = (old_error - new_error) / old_error if old_error != 0 else 0
            contributions.append((improvement, i))
            partial_pred = new_pred
            old_error = new_error

        final_error = old_error

        if not hasattr(self, 'last_error'):
            self.last_error = final_error
            ratio = 1.0
        else:
            ratio = final_error / self.last_error
            self.last_error = final_error

        base_removal = round(fixed_multiplier * ratio)
        # If the ratio is below 1, subtract 1 from the computed removal count (but not below 0)
        if ratio < 1:
            removal_count = max(base_removal - 1, 0)
        else:
            removal_count = base_removal
        removal_count = min(removal_count, len(self.trees))
        
        print(f"[Adaptive Ratio Strategy] final_error={final_error:.3f}, adapted_ratio={ratio:.3f}, removal_count={removal_count}")
        
        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for _, idx in contributions[:removal_count]]
        for idx in sorted(to_remove, reverse=True):
            self.trees.pop(idx)
        return self

    def update_contribution_weighted_ratio(self, X, y, fixed_multiplier, gamma_multiplier, beta_multiplier, alpha_multiplier, baseline=0):
        """
        Weighted Ratio Strategy Update:
          - Uses a weighted combination of past error ratios:
                weighted_ratio = gamma_multiplier * (e(t-3)/e(t-4))
                               + beta_multiplier * (e(t-2)/e(t-3))
                               + alpha_multiplier * (e(t)/e(t-1))
          - Computes removal_count = round(fixed_multiplier * weighted_ratio + baseline)
          - Requires at least 4 past errors stored in self.error_history.
        """
        final_error = self.error_metric(y, self.predict(X))
        
        if not hasattr(self, 'error_history'):
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
        
        print(f"[Weighted Ratio] e(t-4)={e_t4:.3f}, e(t-3)={e_t3:.3f}, e(t-2)={e_t2:.3f}, e(t-1)={e_t1:.3f}, e(t)={final_error:.3f}")
        print(f"[Weighted Ratio] ratio1={ratio1:.3f}, ratio2={ratio2:.3f}, ratio3={ratio3:.3f}, weighted_ratio={weighted_ratio:.3f}, removal_count={removal_count}")
        
        partial_pred = np.full(y.shape, self.initial_pred)
        contributions = []
        error_before = self.error_metric(y, partial_pred)
        for i, tree in enumerate(self.trees):
            tree_pred = tree.predict(X)
            new_pred = partial_pred + self.alpha * tree_pred
            new_error = self.error_metric(y, new_pred)
            improvement = (error_before - new_error) / error_before if error_before != 0 else 0
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

    def build_incremental_trees(self, X_new, Y_residuals_left, number_of_incremental_trees=5):
        """
        Incrementally adds new trees to the ensemble.
          - Each new tree is trained on the current residuals.
          - After each tree is added, the residuals are updated.
        """
        from sklearn.tree import DecisionTreeRegressor
        
        for _ in range(number_of_incremental_trees):
            tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + len(self.trees))
            tree.fit(X_new, Y_residuals_left)
            self.trees.append(tree)
            # Update the residuals using the newly added tree.
            Y_residuals_left -= self.alpha * tree.predict(X_new)

    def _retrain(self, X, y, num_trees=40):
        """
        Full Retraining:
          - Resets the ensemble by retraining a new GradientBoostingRegressor on the data.
          - Updates the initial prediction to the mean of y.
          - Extracts the ensemble of trees from the new GBDT.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        print("Retraining triggered.")
        
        # Reset the initial prediction.
        self.initial_pred = np.mean(y)
        
        # Train a new GradientBoostingRegressor with the desired number of trees.
        new_gbdt = GradientBoostingRegressor(
            n_estimators=num_trees,
            learning_rate=self.alpha,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        new_gbdt.fit(X, y)
        
        # Extract trees from the new model.
        self.trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        print(f"Retrained with {len(self.trees)} trees.")
        return self

    def get_number_of_trees(self):
        """
        Returns the current number of trees in the ensemble.
        """
        return len(self.trees)
    
    def predict(self, X):
        """
        Generates predictions:
          - Starts with the initial prediction.
          - Adds the contribution of each tree (scaled by alpha).
        """
        predictions = np.full(X.shape[0], self.initial_pred)
        for tree in self.trees:
            predictions += self.alpha * tree.predict(X)
        return predictions
