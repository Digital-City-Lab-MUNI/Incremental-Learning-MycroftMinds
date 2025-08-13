import numpy as np
from sklearn.metrics import root_mean_squared_error  # Default error metric


class AccuracyUpdatedGBDT:
    """
    Accuracy Updated Gradient Boosting Decision Tree (eGBDT).

    Each tree's weight is updated based on its relative contribution to error reduction.
    Weights are regularized (pushed toward 1) and normalized so that the ensemble stays balanced.
    Trees with low weight are pruned and replaced.
    """

    def __init__(
        self,
        y_train,
        alpha=0.01,
        max_depth=5,
        random_state=42,
        error_metric=None,
        eps=1e-8,
        buffer_size=30,
        error_damping=1,
        reg_coef=1,
        num_trees=40,
    ):
        self.max_depth = max_depth
        self.random_state = random_state
        self.error_damping = error_damping
        self.num_trees = num_trees
        self.alpha = alpha
        self.eps = eps
        self.error_metric = error_metric if error_metric is not None else root_mean_squared_error
        self.error_damping = error_damping
        self.reg_coef = reg_coef
        self.initial_pred = np.mean(y_train)
        self.trees = []
        self.weights = np.array([])
        self.buffer_size = buffer_size
        self.data_buffer = []

    def train(self, X, y, num_trees=40):
        from sklearn.ensemble import GradientBoostingRegressor

        print("Training triggered.")
        self.initial_pred = np.mean(y)
        new_gbdt = GradientBoostingRegressor(
            n_estimators=self.num_trees,
            learning_rate=self.alpha,
            max_depth=self.max_depth,
            random_state=self.random_state,
        )
        new_gbdt.fit(X, y)
        self.trees = [estimator for estimator in new_gbdt.estimators_[:, 0]]
        self.weights = np.ones(len(self.trees))
        print(f"Trained with {len(self.trees)} trees.")
        return self

    def predict(self, X):
        predictions = np.full(X.shape[0], self.initial_pred)
        for w, tree in zip(self.weights, self.trees):
            predictions += self.alpha * w * tree.predict(X)
        return predictions

    def update_weights_by_contribution_rel(self, X_new, y_new, weight_threshold_ratio=0.1, eps=1e-8):

        X_new = np.array(X_new).reshape(1, -1) if len(np.array(X_new).shape) == 1 else X_new

        n = len(self.trees)
        contributions = np.zeros(n)
        partial_pred = np.full(y_new.shape, self.initial_pred)

        re_before = self.error_metric(y_new, partial_pred)
        for i in range(n):
            pred_i = self.alpha * self.weights[i] * self.trees[i].predict(X_new)[0]
            re_after = self.error_metric(y_new, partial_pred + pred_i)
            contributions[i] = (re_before - re_after) / (re_before + eps)
            partial_pred += pred_i
            re_before = re_after

        for i in range(n):
            update = self.error_damping * contributions[i]
            reg = self.reg_coef * (1 - self.weights[i])
            self.weights[i] = self.weights[i] * (1 + update) + reg
            self.weights[i] = max(self.weights[i], eps)

        # avg_weight = np.mean(self.weights)
        # self.weights = self.weights / (avg_weight + eps)

        max_weight = np.max(self.weights) if n > 0 else 1.0
        threshold = weight_threshold_ratio * max_weight
        indices_to_keep = [i for i, w in enumerate(self.weights) if w >= threshold]
        removed_indices = [i for i in range(n) if i not in indices_to_keep]
        if removed_indices:
            print(f"Removing trees at indices {removed_indices} due to low relative weight.")
            self.trees = [self.trees[i] for i in indices_to_keep]
            self.weights = self.weights[indices_to_keep]
            for _ in range(len(removed_indices)):
                self.add_tree_from_buffer()
        return removed_indices

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
        from sklearn.tree import DecisionTreeRegressor

        tree = DecisionTreeRegressor(max_depth=self.max_depth, random_state=self.random_state + len(self.trees))
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
                    self.data_buffer.append((X[i : i + 1], y[i]))
            else:
                for i in range(available_slots):
                    self.data_buffer.append((X[i : i + 1], y[i]))
                extra = n_new - available_slots
                for _ in range(extra):
                    self.data_buffer.pop(0)
                for i in range(available_slots, n_new):
                    self.data_buffer.append((X[i : i + 1], y[i]))
        else:
            for _ in range(n_new):
                if self.data_buffer:
                    self.data_buffer.pop(0)
            for i in range(n_new):
                self.data_buffer.append((X[i : i + 1], y[i]))

    def get_number_of_trees(self):
        """
        Returns the current number of trees in the ensemble.
        """
        return len(self.trees)
