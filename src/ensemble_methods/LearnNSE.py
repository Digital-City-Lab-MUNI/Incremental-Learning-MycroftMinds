import copy

import numpy as np
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


class LearnNSE:
    """
    Learn++.NSE: an incremental ensemble for non-stationary (drifting) data,
    following the pseudocode steps 1–8.
    """
    def __init__(self, base_classifier=None, alpha=0.5, beta=10.0, ensemble_size=None, pruning_type='age'):
        # Base learner (single tree by default)
        if base_classifier is None:
            self.base_classifier = DecisionTreeRegressor(random_state=0)
        else:
            self.base_classifier = base_classifier
        self.alpha = alpha           # time-decay slope
        self.beta = beta             # crossing point
        self.ensemble_size = ensemble_size  # max ensemble size

        # ensemble state
        self.models = []
        self.voting_weights = []
        self.instance_weights = None
        self.pruning_type = pruning_type  # unused, but can be set to 'error' for error-based pruning

    def _init_weights(self, n_samples):
        # Step 0: D1(i) = 1/m1
        return np.ones(n_samples) / n_samples

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        t = len(self.models) + 1

        # -----------------------------
        # Step 1 & 2: Initialize or update instance distribution D_t
        # -----------------------------
        if t == 1:
            # First batch: uniform
            D = self._init_weights(n)
        else:
            D_prev = self.instance_weights
            preds = self.predict(X)
            mistakes = (preds != y)
            E_t = np.sum(D_prev * mistakes)
            # avoid zero (perfect ensemble) causing zero weights
            eps = 1e-10
            if E_t < eps:
                E_t = eps
            # w_t(i)=E_t if correct, else 1
            w = np.where(~mistakes, E_t, 1.0)
            D = w / np.sum(w)

        # -----------------------------
        # Step 3: Train new tree h_t on weighted sample
        # -----------------------------
        idx = np.random.choice(n, size=n, replace=True, p=D)
        clf = copy.deepcopy(self.base_classifier)
        clf.fit(X[idx], y[idx])
        self.models.append(clf)

        # -----------------------------
        # Step 4: Compute each E_k (cap at 0.5)
        # -----------------------------
        errors = []
        for m in self.models:
            E_k = np.sum(D * (m.predict(X) != y))
            errors.append(min(E_k, 0.5))
        errors = np.array(errors)

        # -----------------------------
        # Step 5: Time-based weights (unused below)
        # -----------------------------
        t_weights = np.array([1/(1+np.exp(-self.alpha*(t-k-self.beta)))
                              for k in range(1, len(self.models)+1)])
        t_weights /= t_weights.sum()

        # -----------------------------
        # Step 6: Prune if > ensemble_size
        # -----------------------------
        if self.ensemble_size and len(self.models) > self.ensemble_size:
            if self.pruning_type == 'error':
                # Prune based on error
                idx = np.argsort(errors)[:self.ensemble_size]
                self.models.pop(idx[0])  # remove worst model
                errors = np.delete(errors, idx[0])
                t_weights = np.delete(t_weights, idx[0])

            elif self.pruning_type == 'age':
                # Prune based on age
                self.models.pop(0)
                errors = errors[1:]
                t_weights = t_weights[1:]

        # -----------------------------
        # Step 7: Voting weights W_k = log((1-E_k)/E_k)
        # -----------------------------
        eps = 1e-10
        ws = (1 - errors) / (errors + eps)
        self.voting_weights = np.log(ws)

        # -----------------------------
        # Save D_t for next round
        # -----------------------------
        self.instance_weights = D

    def predict(self, X):
        if not self.models:
            raise ValueError("No classifiers yet. Call fit() first.")
        X = np.asarray(X)
        # collect seen classes
        classes = np.unique(np.concatenate([m.classes_ for m in self.models]))
        idx_map = {c:i for i,c in enumerate(classes)}
        votes = np.zeros((len(X), len(classes)))
        for W, m in zip(self.voting_weights, self.models):
            preds = m.predict(X)
            for i, p in enumerate(preds):
                votes[i, idx_map[p]] += W
        winners = votes.argmax(axis=1)
        return classes[winners]

    def score(self, X, y):
        return np.mean(self.predict(X) == np.asarray(y))
