import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator

class LearnNSE:
    """
    Learn++.NSE: incremental ensemble for non-stationary data, supporting classification and regression.
    Steps 1–8 of the pseudocode, with optional 'age'/'error' pruning.
    """
    def __init__(self,
                 base_classifier=None,
                 alpha=0.5,
                 beta=10.0,
                 ensemble_size=None,
                 pruning_type='age',
                 regression_task_error_metric=mean_squared_error):
        # Base learner default
        if base_classifier is None:
            self.base_classifier = DecisionTreeClassifier(random_state=0)
        else:
            self.base_classifier = base_classifier

        self.alpha = alpha
        self.beta = beta
        self.ensemble_size = ensemble_size
        self.pruning_type = pruning_type
        self.regression_task_error_metric = regression_task_error_metric

        # ensemble state
        self.models = []
        self.voting_weights = []
        self.instance_weights = None

    def _clone_with_params(self):
        """
        Create a fresh classifier with the same initialization parameters
        but without any learned state.
        """
        # If a classifier class is provided, instantiate it
        if isinstance(self.base_classifier, type):
            return self.base_classifier()

        # If a scikit-learn estimator instance is provided, clone via get_params
        if isinstance(self.base_classifier, BaseEstimator):
            cls = self.base_classifier.__class__
            params = self.base_classifier.get_params(deep=False)
            return cls(**params)

        # Fallback: deep-copy the object
        import copy
        return copy.deepcopy(self.base_classifier)

    def _init_weights(self, n_samples):
        return np.ones(n_samples) / n_samples

    def fit(self, X, y):

        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        t = len(self.models) + 1

        # Steps 1 & 2: initialize or update D_t
        if t == 1:
            D = self._init_weights(n)
        else:
            D_prev = self.instance_weights
            preds = self.predict(X)
            eps = 1e-10

            # classification branch
            if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):
                mistakes = (preds != y)
                E_t = np.sum(D_prev * mistakes)
                E_t = max(E_t, eps)
                w = np.where(mistakes, 1.0, E_t)
            else:
                # regression branch
                per_err = self.regression_task_error_metric(y, preds)
                if np.isscalar(per_err):
                    per_err = np.abs(y - preds)
                E_t = np.sum(D_prev * per_err)
                E_t = max(E_t, eps)
                w = np.where(per_err > E_t, 1.0, E_t)

            D = w / np.sum(w)

        # Step 3: bootstrap sample and train h_t
        idx = np.random.choice(n, size=n, replace=True, p=D)
        clf = self._clone_with_params()
        clf.fit(X[idx], y[idx])
        self.models.append(clf)

        # Step 4: compute errors E_k
        errors = []
        for m in self.models:
            pred_k = m.predict(X)
            if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):
                ek = np.sum(D * (pred_k != y))
            else:
                raw_err = self.regression_task_error_metric(y, pred_k)
                per_err_k = raw_err if not np.isscalar(raw_err) else np.abs(y - pred_k)
                ek = np.sum(D * per_err_k)
            errors.append(min(ek, 0.5))
        errors = np.array(errors)

        # Step 5: time-based weights (unused later)
        t_weights = np.array([1/(1+np.exp(-self.alpha*(t-k-self.beta)))
                               for k in range(1, len(self.models)+1)])
        t_weights /= t_weights.sum()

        # Step 6: pruning if ensemble_size exceeded
        if self.ensemble_size and len(self.models) > self.ensemble_size:
            if self.pruning_type == 'age':
                self.models.pop(0)
                errors = errors[1:]
                t_weights = t_weights[1:]
            else:
                worst = np.argmax(errors)
                self.models.pop(worst)
                errors = np.delete(errors, worst)
                t_weights = np.delete(t_weights, worst)

        # Step 7: voting weights
        eps = 1e-10
        ws = (1 - errors) / (errors + eps)
        self.voting_weights = np.log(ws)

        # save D_t
        self.instance_weights = D

    def predict(self, X):
        X = np.asarray(X)
        if not self.models:
            raise ValueError("No models: call fit() first.")

        # classification
        if hasattr(self.models[0], 'classes_'):
            classes = np.unique(np.concatenate([m.classes_ for m in self.models]))
            idx_map = {c: i for i, c in enumerate(classes)}
            votes = np.zeros((len(X), len(classes)))
            for W, m in zip(self.voting_weights, self.models):
                preds = m.predict(X)
                for i, p in enumerate(preds):
                    votes[i, idx_map[p]] += W
            return classes[votes.argmax(axis=1)]

        # regression
        preds = np.vstack([m.predict(X) for m in self.models])
        weighted = (self.voting_weights[:, None] * preds)
        return weighted.sum(axis=0) / self.voting_weights.sum()

    def score(self, X, y):
        preds = self.predict(X)
        if hasattr(self.models[0], 'classes_'):
            return np.mean(preds == np.asarray(y))
        else:
            u = np.sum((y - preds)**2)
            v = np.sum((y - np.mean(y))**2)
            return 1 - u/v if v != 0 else 0.0
