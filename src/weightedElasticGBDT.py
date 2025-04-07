import numpy as np
from sklearn.metrics import mean_absolute_error  # Default error metric if none provided

class AccuracyUpdatedGBDT:
    """
    Accuracy Updated Gradient Boosting Decision Tree (eGBDT) using scikit-learn's GradientBoostingRegressor.
    
    This class supports several pruning methods and now incorporates per-tree weights.
    Each tree has a learnable weight that is updated (via an update based on tree contribution)
    every time a new datapoint is observed. The weights are normalized so that the overall
    ensemble influence is maintained. Trees whose weight falls below a low threshold are removed.
    """

    def __init__(self, y_train, alpha=0.01, max_depth=5, random_state=42,
                 error_metric=None, buffer_size=30, error_damping=1, num_trees = 40):
        self.max_depth = max_depth
        self.random_state = random_state
        self.error_damping = error_damping
        self.num_trees = num_trees
        self.alpha = alpha
        
        # Use provided error_metric or default to mean absolute error.
        self.error_metric = error_metric if error_metric is not None else mean_absolute_error
        
        # The initial prediction is the mean of the training target.
        self.initial_pred = np.mean(y_train)
        
        # Start with an empty ensemble.
        self.trees = []
        self.weights = np.array([])
        
        # Set up a sliding window (buffer) for the most recent datapoints.
        self.buffer_size = buffer_size
        self.data_buffer = []  # will store tuples of (X, y)

    def train(self, X, y, num_trees=40):
        """
        Full Retraining:
          - Resets the ensemble by training a new GradientBoostingRegressor on the data.
          - Updates the initial prediction to the mean of y.
          - Extracts the ensemble of trees from the new GBDT.
        """
        from sklearn.ensemble import GradientBoostingRegressor
        print("Training triggered.")
        
        self.initial_pred = np.mean(y)
        new_gbdt = GradientBoostingRegressor(
            n_estimators=self.num_trees,
            learning_rate=self.alpha,
            max_depth=self.max_depth,
            random_state=self.random_state
        )
        new_gbdt.fit(X, y)
        self.trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        self.weights = np.ones(len(self.trees))
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
    
    def update_weights_by_contribution_rel(self, X_new, y_new,
                                            weight_threshold_ratio=0.1, eps=1e-8,
                                            error_damping=1, reg_coef=1):
        """
        Update tree weights based on their relative contribution to error reduction,
        with regularization and smoothed normalization.
        
        For each tree i:
        - Compute the relative error before adding tree i:
                re_before = RMSE(|(y - partial_pred)| / (|y|+eps))
        - Then compute re_after after adding the tree’s contribution.
        - Define contribution_i = (re_before - re_after) / (re_before + eps)
        
        Then update the weight as:
            w_i = w_i * (1 + error_damping * contribution_i) + reg_coef*(1 - w_i)
        which regularizes the weight toward 1.
        
        Next, the weights are normalized so that their average remains 1.
        
        Finally, if a tree's weight is below 
            threshold = weight_threshold_ratio * (max(weight))
        then that tree is removed and replaced with a new tree trained on recent buffered data.
        
        Parameters:
        X_new, y_new: New data point(s) or batch.
        weight_threshold_ratio: Fraction of the maximum weight below which a tree is removed.
        eps: Small constant to avoid division by zero.
        error_damping: Scaling factor for the contribution-based update.
        reg_coef: Regularization coefficient to penalize deviations from 1.
        """
        # Ensure X_new is 2D.
        X_new = np.array(X_new).reshape(1, -1) if len(np.array(X_new).shape) == 1 else X_new

        self.error_damping = error_damping 

        n = len(self.trees)
        contributions = np.zeros(n)
        partial_pred = np.full(y_new.shape, self.initial_pred)
        
        # Define a relative error function (RMSE-like)
        def rel_error(y_true, y_pred):
            return np.sum(((y_true - y_pred)**2 / (np.abs(y_true) + eps)))**0.5
        
        # Compute contributions sequentially.
        re_before = rel_error(y_new, partial_pred)
        for i in range(n):
            pred_i = self.alpha * self.weights[i] * self.trees[i].predict(X_new)[0]
            re_after = rel_error(y_new, partial_pred + pred_i)
            contributions[i] = (re_before - re_after) / (re_before + eps)
            partial_pred += pred_i  # update ensemble prediction
            re_before = re_after  # move to the next tree
        
        # Step 1: Update weights with contribution and regularization.
        for i in range(n):
            update = error_damping * contributions[i]
            reg = reg_coef * (1 - self.weights[i])  # push weight toward 1
            self.weights[i] = self.weights[i] * (1 + update) + reg
            self.weights[i] = max(self.weights[i], eps)
        
        # Step 2: Normalize weights so that the average remains 1.
        if n > 0:
            avg_weight = np.mean(self.weights)
            self.weights = self.weights / (avg_weight + eps)
        
        # Step 3: Remove trees whose weight is too low relative to the best tree.
        max_weight = np.max(self.weights) if n > 0 else 1.0
        threshold = weight_threshold_ratio * max_weight
        indices_to_keep = [i for i, w in enumerate(self.weights) if w >= threshold]
        removed_indices = [i for i in range(n) if i not in indices_to_keep]
        removed_count = len(removed_indices)
        if removed_count > 0:
            print(f"Contribution-based Update: Removing {removed_count} tree(s) at indices {removed_indices} due to low relative weight.")
            self.trees = [self.trees[i] for i in indices_to_keep]
            self.weights = self.weights[indices_to_keep]
            # For each removed tree, add a new tree from the data buffer.
            for _ in range(removed_count):
                self.add_tree_from_buffer()
        return


    def add_tree_from_buffer(self):
        """
        Helper method:
         - Trains a new tree on the recent datapoints stored in self.data_buffer.
         - The new tree is fit on the residuals computed from the current ensemble.
         - Its weight is initialized to 1.
        """
        if len(self.data_buffer) == 0:
            return
        # Stack buffered datapoints.
        X_buffer = np.vstack([d[0] for d in self.data_buffer])
        y_buffer = np.array([d[1] for d in self.data_buffer])
        y_pred_current = self.predict(X_buffer)
        residuals = y_buffer - y_pred_current
        residuals = np.clip(residuals, -1e5, 1e5)
        from sklearn.tree import DecisionTreeRegressor
        tree = DecisionTreeRegressor(max_depth=self.max_depth,
                                     random_state=self.random_state + len(self.trees))
        tree.fit(X_buffer, residuals)
        self.trees.append(tree)
        self.weights = np.append(self.weights, 1.0)

    def add_data_to_data_buffer(self, X, y, limit_of_data_buffer):
        """
        Adds new data points (X, y) to the data buffer.
        
        Parameters:
          X: numpy array of shape (n, features) for n new data points.
          y: array-like of length n corresponding to target values.
          limit_of_data_buffer: Maximum number of data points to store in the buffer.
        
        Behavior:
          - If the current buffer size is less than limit_of_data_buffer and adding the new data
            would not exceed the limit, simply append the new data.
          - Otherwise, remove the oldest data points equal in number to len(X) and then append the new data.
        """
        n_new = X.shape[0]  # number of new datapoints
        if len(self.data_buffer) < limit_of_data_buffer:
            available_slots = limit_of_data_buffer - len(self.data_buffer)
            if n_new <= available_slots:
                for i in range(n_new):
                    self.data_buffer.append((X[i:i+1], y[i]))
            else:
                for i in range(available_slots):
                    self.data_buffer.append((X[i:i+1], y[i]))
                extra = n_new - available_slots
                for _ in range(extra):
                    self.data_buffer.pop(0)
                for i in range(available_slots, n_new):
                    self.data_buffer.append((X[i:i+1], y[i]))
        else:
            for _ in range(n_new):
                if self.data_buffer:
                    self.data_buffer.pop(0)
            for i in range(n_new):
                self.data_buffer.append((X[i:i+1], y[i]))

    def get_number_of_trees(self):
        """
        Returns the current number of trees in the ensemble.
        """
        return len(self.trees)
