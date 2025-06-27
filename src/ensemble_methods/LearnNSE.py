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
                 alpha=0.5, # alpha and beta are hyperparameters for time-based weighting
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
        self.ensemble_size = ensemble_size # s in the paper
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
            D_prev = self.instance_weights # must not be historically inherited, for each batch new weights
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
        idx = np.random.choice(n, size=n, replace=True, p=D) # maybe
        clf = self._clone_with_params()
        clf.fit(X[idx], y[idx])
        self.models.append(clf)

        # Step 4: compute errors E_k 
        # for new clf the clf will be retrained if its error is above 0.5
        errors = []
        for m in self.models:
            pred_k = m.predict(X[idx]) # this needs be the bootstrapped sample hence [idx]
            if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):
                ek = np.sum(D * (pred_k != y[idx]))
                Beta = (errors) / (1- errors)
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
        # t_weights rename to voting_weights

        # Beta is missing equation 8 in the paper
        Beta = t_weights * Beta

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
        
        self.voting_weights = np.log(1/Beta)

        # save D_t
        self.instance_weights = D

        def predict(self, X):
            """
            Step 8: Final hypothesis H_t(x) = arg max_y sum_k W_k * I{h_k(x)=y}
            """
            X = np.asarray(X)
            if not self.models:
                raise ValueError("No models: call fit() first.")
            # simplify voting with weights
            # add regression equivalent

            # gather predictions from each model: shape (t, n_samples)
            preds = np.array([m.predict(X) for m in self.models])
            # find all unique classes across ensemble
            classes = np.unique(preds)
            # map class labels to column indices
            idx_map = {c: i for i, c in enumerate(classes)}
            # accumulate weighted votes for each sample and class
            votes = np.zeros((X.shape[0], classes.size))
            for k, W in enumerate(self.voting_weights):
                for i, p in enumerate(preds[k]):
                    votes[i, idx_map[p]] += W
            # select class with highest vote sum
            winner_idx = votes.argmax(axis=1)
            return classes[winner_idx]
    def score(self, X, y):
        preds = self.predict(X)
        if hasattr(self.models[0], 'classes_'):
            return np.mean(preds == np.asarray(y))
        else:
            u = np.sum((y - preds)**2)
            v = np.sum((y - np.mean(y))**2)
            return 1 - u/v if v != 0 else 0.0
