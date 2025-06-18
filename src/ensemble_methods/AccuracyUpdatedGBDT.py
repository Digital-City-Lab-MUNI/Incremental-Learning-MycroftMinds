import numpy as np
from sklearn.metrics import mean_squared_error, root_mean_squared_error
from functools import partial

class PowerErrorMetric:
    """
    A picklable wrapper around any base_error_fn such that
      PowerErrorMetric(base, n)(y_true, y_pred) == base(y_true, y_pred)**n
    """
    def __init__(self, base_error_fn, power):
        self.base = base_error_fn
        self.power = power

    def __call__(self, y_true, y_pred):
        return self.base(y_true, y_pred) ** self.power



class AccuracyUpdatedGBDT:
    """
    Accuracy Updated Gradient Boosting Decision Tree (eGBDT).

    Each tree's weight is updated based on its relative contribution to error reduction.
    Weights are regularized (pushed toward 1) and normalized so that the ensemble stays balanced.
    Trees with low weight are pruned and replaced.
    """

    def __init__(
        self,
        pretrained_model,
        y_train,
        alpha=0.01,
        max_depth=5,
        random_state=42,
        error_metric=None,
        error_power=1,
        eps=1e-10,
        batch_multiplier=None,
        buffer_size=None,
        error_damping=1,
        reg_coef=1,
        weight_threshold_ratio=None,
        sigmoid_k=None,
        w_scale=None,
        temp=None,
        fixed_threshold=0.1,
        ratio_power=None
    ):
        # Core model parameters
        self.pretrained_model      = pretrained_model
        self.max_depth             = pretrained_model.max_depth
        self.random_state          = (
            pretrained_model.random_state
            if hasattr(pretrained_model, 'random_state') and pretrained_model.random_state is not None
            else random_state
        )
        self.alpha                 = (
            pretrained_model.learning_rate
            if hasattr(pretrained_model, 'learning_rate')
            else alpha
        )

        # Data & initial prediction
        self.y_train               = y_train
        self.initial_pred          = np.mean(y_train)

        # Unpack the pretrained estimators
        # Assumes sklearn GradientBoosting: estimators_ is shape (n_estimators, 1)
        self.trees                 = [t for t in pretrained_model.estimators_[:, 0]]

        # Error metric & exponentiation
        base = error_metric or mean_squared_error
        self.error_metric = PowerErrorMetric(base, error_power)
        self.eps          = eps

        # Weights for each tree start at 1
        self.weights     = np.ones(len(self.trees))

        # Buffers & damping
        self.batch_multiplier      = batch_multiplier
        self.buffer_size           = buffer_size or 30
        self.data_buffer           = []

        # Update hyperparameters
        self.error_damping         = error_damping
        self.reg_coef              = reg_coef
        self.weight_threshold_ratio= (
            weight_threshold_ratio
            if weight_threshold_ratio is not None
            else 0.1
        )
        self.sigmoid_k             = 1.0 if sigmoid_k is None else sigmoid_k
        self.w_scale               = 1.0 if w_scale   is None else w_scale
        self.temp                  = 1.0 if temp      is None else temp
        self.fixed_threshold       = fixed_threshold
        self.ratio_power           = 0.5 if ratio_power is None else ratio_power

        # For downstream introspection
        self.params_ = {
            'alpha':                  self.alpha,
            'max_depth':              self.max_depth,
            'random_state':           self.random_state,
            'eps':                    self.eps,
            'buffer_size':            self.buffer_size,
            'error_damping':          self.error_damping,
            'reg_coef':               self.reg_coef,
            'weight_threshold_ratio': self.weight_threshold_ratio,
            'sigmoid_k':              self.sigmoid_k,
            'w_scale':                self.w_scale,
            'temp':                   self.temp,
            'fixed_threshold':        self.fixed_threshold,
            'ratio_power':            self.ratio_power,

        }

    def predict(self, X):
        X = np.array(X)
        preds = np.full(X.shape[0], self.initial_pred, dtype=float)
        for w, tree in zip(self.weights, self.trees):
            preds += self.alpha * w * tree.predict(X)
        return preds

    
    def __setstate__(self, state):
        self.__dict__.update(state)
        if not hasattr(self, "params_"):
            self.params_ = {
                'alpha': self.alpha,
                'max_depth': self.max_depth,
                'random_state': self.random_state,
                'eps': self.eps,
                'buffer_size': self.buffer_size,
                'error_damping': self.error_damping,
                'reg_coef': self.reg_coef,
                'weight_threshold_ratio': self.weight_threshold_ratio
            }


    def update_weights_by_contribution_rel(self, X_new, y_new, eps=1e-10):
        """
        Relative Contribution Weighted Update with Absolute‐Error Dampening:
        1) Compute final_error on full ensemble.
        2) Compute raw_ratio = final_error / last_error (default 1.0) and power-skew into [0,1): ratio_norm.
        3) Compute absolute‐error scale = avg_error / (avg_error + error_scale_const).
            - When the average error is small compared to error_scale_const, this → 0.
            - When the average error is large, this → 1.
        4) effective_norm = ratio_norm * abs_err_scale.
        5) Measure each tree’s contribution → w_new via sigmoid.
        6) Prune trees with w_new < fixed_threshold.
        7) Blend old weights toward w_new by effective_norm.
        8) Return list of removed indices.
        """
        # 1) normalize X_new
        X = np.array(X_new)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        err_fn = self.error_metric 

        # 6) precompute tree predictions
        trees      = self.trees
        n          = len(trees)
        tree_preds = [t.predict(X) for t in trees]

        # 7) compute contributions in one loop
        partial_pred = np.full_like(y_new, self.initial_pred, dtype=float)
        old_error    = err_fn(y_new, partial_pred)

        contrib   = np.empty(n, dtype=float)
        alpha     = self.alpha

        for i, t_pred in enumerate(tree_preds):
            inc       = alpha  * t_pred
            new_pred  = partial_pred + inc
            new_error = err_fn(y_new, new_pred)

            # normalized improvement
            contrib[i] = (old_error - new_error) / (old_error + eps)

            partial_pred = new_pred
            old_error    = new_error

        # 8) sigmoid-scale into proposals w_new
        scaled = contrib / self.temp
        w_new  = self.w_scale / (1.0 + np.exp(-self.sigmoid_k * scaled))

        # 9) prune any w_new below threshold
        keep    = [i for i, w in enumerate(w_new) if w >= self.fixed_threshold]
        removed = [i for i in range(n) if i not in keep]
        if removed:
            print(f"[ContributionRel] pruning indices {removed} (w < {self.fixed_threshold})")
            self.trees = [trees[i] for i in keep]
            w_new      = w_new[keep]
            old_w      = self.weights[keep]
        else:
            old_w = self.weights

        # 10) blend old_w → w_new by the dampened norm
        #    small effective_norm ⇒ mostly old_w; large ⇒ more w_new
        self.weights = (1-self.ratio_power) * old_w + self.ratio_power * w_new

        return removed



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

    def build_incremental_trees(self, X_new, Y_residuals_left, number_of_incremental_trees = 5):
        """
        Incrementally adds new trees to the ensemble.
          - Each new tree is trained on the current residuals.
          - After each tree is added, the residuals are updated.
        """
        from sklearn.tree import DecisionTreeRegressor
     
        
        for _ in range(number_of_incremental_trees):
            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                random_state=self.random_state + len(self.trees)
            )
            tree.fit(X_new, Y_residuals_left)

            # Add tree to ensemble
            self.trees.append(tree)

            # Update residuals
            Y_residuals_left -= self.alpha * tree.predict(X_new)

            # Append a weight of 1.0 for this new tree
            self.weights = np.append(self.weights, 1.0)

    def get_number_of_trees(self):
        """
        Returns the current number of trees in the ensemble.
        """
        return len(self.trees)
