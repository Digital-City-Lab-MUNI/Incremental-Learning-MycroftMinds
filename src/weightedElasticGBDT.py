import numpy as np
from sklearn.metrics import mean_absolute_error  # Default error metric if none provided

class WeightedElasticGBDT:
    """
    Elastic Gradient Boosting Decision Tree (eGBDT) using scikit-learn's GradientBoostingRegressor.

    This class supports several pruning methods and now incorporates per-tree weights.
    Each tree has a learnable weight that is updated (via a gradient update) every time a new
    datapoint is observed. The weights are normalized in a zero-sum fashion: reducing the weight
    of one tree by x will increase the combined weights of all other trees by x.
    Trees whose weight falls below a low threshold are removed.
    """

    def __init__(self, pretrained_model, y_train, m_threshold=40, alpha=None,
                 max_depth=5, random_state=42, error_metric=None,
                 buffer_size=30, delta_bar_delta_init_lr=0.05):
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
        
        # Initialize per-tree weights to 1.
        self.weights = np.ones(len(self.trees))
        
        # Set up a sliding window (buffer) for the most recent datapoints.
        self.buffer_size = buffer_size
        self.data_buffer = []  # will store tuples of (x, y)
        
        # Parameters for the incremental delta bar delta algorithm.
        # For each tree we maintain a history variable h and a log learning rate beta.
        self.delta_bar_delta_init_lr = delta_bar_delta_init_lr
        n_trees = len(self.trees)
        self.h = np.zeros(n_trees)  # one per tree
        self.beta = np.log(self.delta_bar_delta_init_lr) * np.ones(n_trees)

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

        # Note: Here we ignore weights for the purpose of index-based pruning.
        for i in range(n_trees):
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
            self.weights = self.weights[:best_i + 1]
            self.h = self.h[:best_i + 1]
            self.beta = self.beta[:best_i + 1]
            print(f"[Index-Based] Pruned {pruned_count} trees. Now {len(self.trees)} remain.")
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
            improvement = old_error - new_error
            contributions.append((improvement, i))
            partial_pred = new_pred
            old_error = new_error

        contributions.sort(key=lambda x: x[0])
        to_remove = [idx for _, idx in contributions[:removal_count]]
        for idx in sorted(to_remove, reverse=True):
            self.trees.pop(idx)
            self.weights = np.delete(self.weights, idx)
            self.h = np.delete(self.h, idx)
            self.beta = np.delete(self.beta, idx)

        print(f"[Ratio-Based] final_error={final_error:.3f}, ratio={ratio:.3f}, removal_count={removal_count}")
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
            self.weights = np.append(self.weights, 1.0)
            # Initialize delta bar delta parameters for the new tree.
            self.h = np.append(self.h, 0.0)
            self.beta = np.append(self.beta, np.log(self.delta_bar_delta_init_lr))
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
        
        self.initial_pred = np.mean(y)
        new_gbdt = GradientBoostingRegressor(
            n_estimators=num_trees,
            learning_rate=self.alpha,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        new_gbdt.fit(X, y)
        self.trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        self.weights = np.ones(len(self.trees))
        n_trees = len(self.trees)
        self.h = np.zeros(n_trees)
        self.beta = np.log(self.delta_bar_delta_init_lr) * np.ones(n_trees)
        print(f"Retrained with {len(self.trees)} trees.")
        return self

    def predict(self, X):
        """
        Generates predictions:
          - Starts with the initial prediction.
          - Adds the weighted contribution of each tree (scaled by alpha).
        """
        predictions = np.full(X.shape[0], self.initial_pred)
        for w, tree in zip(self.weights, self.trees):
            predictions += self.alpha * w * tree.predict(X)
        return predictions

    def update_gradient_weights(self, x, y, eta, weight_threshold=1e-5):
        """
        Gradient-Based Weight Update:
          - For a new datapoint (x, y), update each tree's weight using the gradient of a squared error loss.
          - The unconstrained update for tree i is:
                delta_i = 2 * eta * alpha * T_i(x) * (y - y_hat)
          - Then, for each tree, we adjust the update to enforce a zero-sum rule so that a reduction in one
            tree's weight is exactly compensated by an increase distributed equally among the other trees.
          - Finally, trees whose updated weight falls below weight_threshold are removed.
        """
        x = np.array(x).reshape(1, -1) if len(np.array(x).shape) == 1 else x
        y_hat = self.initial_pred
        for w, tree in zip(self.weights, self.trees):
            y_hat += self.alpha * w * tree.predict(x)[0]
        error = y - y_hat
        n = len(self.trees)
        if n == 0:
            return

        delta = np.zeros(n)
        for i, tree in enumerate(self.trees):
            prediction_i = tree.predict(x)[0]
            delta[i] = 2 * eta * self.alpha * prediction_i * error

        new_delta = np.zeros(n)
        for i in range(n):
            compensation = 0.0
            for j in range(n):
                if j == i:
                    continue
                if delta[j] < 0:
                    compensation += (-delta[j]) / (n - 1)
                elif delta[j] > 0:
                    compensation -= (delta[j]) / (n - 1)
            new_delta[i] = delta[i] + compensation
        
        damping = 0.5
        new_weights = damping * self.weights + new_delta
        new_weights = np.maximum(new_weights, 0.0)
        indices_to_keep = [i for i, w in enumerate(new_weights) if w >= weight_threshold]
        num_removed = n - len(indices_to_keep)
        if num_removed > 0:
            print(f"Removing {num_removed} tree(s) due to low weight.")
        self.trees = [self.trees[i] for i in indices_to_keep]
        self.weights = new_weights[indices_to_keep]
        total_weight = np.sum(self.weights)
        desired_total = len(self.trees)
        if total_weight > 0:
            self.weights = self.weights * (desired_total / total_weight)
        return
    def update_weights_by_contribution_rel(self, X_new, y_new, bottom_quantile=0.1, 
                                            weight_threshold_ratio=0.1, eps=1e-8, k=0.5):
        """
        Update tree weights based on their relative contribution to error reduction.
        
        This method uses a relative error metric (MAPE-like) to compute the contribution.
        
        For each tree i:
        - Compute the relative error before adding tree i:
                re_before = MAE(|(y - partial_pred)| / (|y|+eps))
        - Then compute re_after after adding the tree’s contribution.
        - Define contribution_i = (re_before - re_after) / (re_before + eps)
        
        For trees whose contribution is in the bottom 'bottom_quantile' and is negative,
        update the weight as:
            w_i <- w_i * (1 + k * contribution_i)
        Then normalize weights so that the average weight remains 1.
        
        Finally, if a tree's weight drops below weight_threshold_ratio * (max weight), remove it
        and replace it with a new tree trained on recent buffered data.
        
        Parameters:
        X_new, y_new: New data point(s) or batch.
        bottom_quantile: Fraction of trees (e.g., 0.1 for worst 10%) to consider for weight adjustment.
        weight_threshold_ratio: Fraction of the maximum weight below which a tree is removed.
        eps: A small constant to avoid division by zero.
        k: Damping constant to scale the relative update.
        """
        # Ensure X_new is 2D.
        X_new = np.array(X_new).reshape(1, -1) if len(np.array(X_new).shape) == 1 else X_new

        n = len(self.trees)
        contributions = np.zeros(n)
        partial_pred = np.full(y_new.shape, self.initial_pred)
        
        # Define a relative error function (MAPE-like)
        def rel_error(y_true, y_pred):
            return np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps)))
        
        # Compute contributions sequentially.
        re_before = rel_error(y_new, partial_pred)
        for i in range(n):
            pred_i = self.alpha * self.weights[i] * self.trees[i].predict(X_new)[0]
            re_after = rel_error(y_new, partial_pred + pred_i)
            contributions[i] = (re_before - re_after) / (re_before + eps)
            partial_pred += pred_i  # update ensemble prediction
            re_before = re_after  # move to the next tree
        
        # Determine the threshold for bottom quantile.
        quantile_threshold = np.quantile(contributions, bottom_quantile)
        
        # Step 1: Adjust weights for trees in the bottom quantile with negative contributions.
        for i in range(n):
            if contributions[i] < quantile_threshold and contributions[i] < 0:
                # Update weight using a damping constant k.
                self.weights[i] *= (1 + k * contributions[i])
                # Ensure weight doesn't fall below a small positive value.
                self.weights[i] = max(self.weights[i], eps)
        
        # Step 2: Normalize weights so that the average remains 1.
        if n > 0:
            avg_weight = np.mean(self.weights)
            self.weights = self.weights / (avg_weight + eps)
        
        # Step 3: Remove trees whose weight is too low relative to the best tree.
        max_weight = np.max(self.weights) if n > 0 else 1.0
        indices_to_keep = [i for i, w in enumerate(self.weights) if w >= weight_threshold_ratio * max_weight]
        removed_count = n - len(indices_to_keep)
        if removed_count > 0:
            print(f"Contribution-based Update: Removing {removed_count} tree(s) due to low relative weight.")
            self.trees = [self.trees[i] for i in indices_to_keep]
            self.weights = self.weights[indices_to_keep]
            self.h = self.h[indices_to_keep]
            self.beta = self.beta[indices_to_keep]
            # For each removed tree, add a new tree from the data buffer.
            for _ in range(removed_count):
                self._add_tree_from_buffer()
        return


    def _add_tree_from_buffer(self):
        """
        Helper method:
         - Trains a new tree on the recent datapoints stored in self.data_buffer.
         - The new tree is fit on the residuals computed from the current ensemble.
         - Its weight is initialized to 1, and its delta bar delta parameters are initialized.
        """
        if len(self.data_buffer) == 0:
            return
        # Stack buffered datapoints.
        X_buffer = np.vstack([d[0] for d in self.data_buffer])
        y_buffer = np.array([d[1] for d in self.data_buffer])
        # Compute the current ensemble predictions on the buffer.
        y_pred_current = self.predict(X_buffer)
        residuals = y_buffer - y_pred_current
        # Clip residuals to a safe range.
        residuals = np.clip(residuals, -1e5, 1e5)
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + len(self.trees))
        tree.fit(X_buffer, residuals)
        self.trees.append(tree)
        self.weights = np.append(self.weights, 1.0)
        self.h = np.append(self.h, 0.0)
        self.beta = np.append(self.beta, np.log(self.delta_bar_delta_init_lr))
    
    def get_number_of_trees(self):
        """
        Returns the current number of trees in the ensemble.
        """
        return len(self.trees)
