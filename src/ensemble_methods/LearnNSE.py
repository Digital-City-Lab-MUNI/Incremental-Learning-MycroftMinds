import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import mean_squared_error
from sklearn.base import BaseEstimator

def mse_per_instance(y_true, y_pred):
    return (y_true - y_pred)**2


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
                alpha_multiplier= 0.8,
                beta_multiplier= 0.55,
                gamma_multiplier= 0.3,
                fixed_multiplier=1.2,
                baseline=1.0,
                removal_count_calculation=True,
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
        self.regression_task_error_metric = lambda y_true, y_pred: mse_per_instance(y_true, y_pred)
        # ensemble state
        self.models = []
        self.voting_weights = []
        self.instance_weights = None
        self.alpha_multiplier = alpha_multiplier
        self.beta_multiplier = beta_multiplier
        self.gamma_multiplier = gamma_multiplier
        self.fixed_multiplier = fixed_multiplier
        self.baseline = baseline
        self.removal_count_calculation = removal_count_calculation


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

    def fit(self, X, y, classification_base_error = 0.5, regression_base_error = None):

        X = np.asarray(X)
        y = np.asarray(y)
        n = len(y)
        t = len(self.models) + 1

        mean_y = np.mean(y) # up for a debate: could be np.median(y) or other robust measure
        base_err = self.regression_task_error_metric(y, np.full_like(y, mean_y))
        D = np.full(n, 1.0 / n)

        # Steps 1 & 2: initialize or update D_t
        if t == 1:
            D = np.full(n, 1.0 / n)
        else:
            # D_prev = self.instance_weights # must not be historically inherited, for each batch new weights
            preds = self.predict(X)
            eps = 1e-10

            # classification branch
            if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):
                # Step 1: Compute ensemble error uniformly (no previous weights!)
                mistakes = (preds != y).astype(float)
                ensemble_error_on_original_batch = np.sum(mistakes) / n  # uniform averaging

                # ensemble_error_on_original_batch = np.clip(ensemble_error_on_original_batch, eps, 1 - eps)        # stability

                # Step 2: Update weights based on correctness
                w = np.where(mistakes == 1, 1.0 / n, (ensemble_error_on_original_batch + eps) / n )

            else:
                # regression branch using mean(y) as base error
                per_err = self.regression_task_error_metric(y, preds)
                ensemble_error_on_original_batch = np.mean(per_err) 
                ensemble_error_on_original_batch = max(ensemble_error_on_original_batch, eps)
                w = np.where(per_err > base_err, 1.0, ensemble_error_on_original_batch)

            D = w / np.sum(w)

        # Step 3: bootstrap sample and train h_t
        idx = np.random.choice(n, size=n, replace=True, p=D) 
        base_learner = self._clone_with_params()
        base_learner.fit(X[idx], y[idx])
        idx = np.random.choice(n, size=n, replace=True, p=D)

        if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):

            if (base_learner.predict(X[idx]) != y[idx]).sum()/ n > classification_base_error:
                self.models.append(base_learner)
            else:
                counter = 0
                while ((base_learner.predict(X[idx]) != y[idx]).sum() / n > classification_base_error) and counter < 10:
                    idx = np.random.choice(n, size=n, replace=True, p=D) 
                    base_learner = self._clone_with_params() # perhap this could experiment with different params?
                    base_learner.fit(X[idx], y[idx])
                    idx = np.random.choice(n, size=n, replace=True, p=D) 
                    counter += 1
                self.models.append(base_learner)
        else:

            regression_base_error = np.mean(self.regression_task_error_metric(np.full_like(y, mean_y), y[idx]))
            if np.mean(self.regression_task_error_metric (base_learner.predict(X[idx]), y[idx])) < regression_base_error:
                self.models.append(base_learner)
            else:
                while (np.mean(self.regression_task_error_metric (base_learner.predict(X[idx]), y[idx])) < regression_base_error and counter < 10):
                    idx = np.random.choice(n, size=n, replace=True, p=D) 
                    base_learner = self._clone_with_params()
                    base_learner.fit(X[idx], y[idx])
                    idx = np.random.choice(n, size=n, replace=True, p=D) 
                    counter += 1
                self.models.append(base_learner)

        # add the new model to the ensemble
        # Step 4: compute errors E_k 
        betas = []
        for m in self.models:
            pred_k = m.predict(X[idx]) # this needs be the bootstrapped sample hence [idx]
            if hasattr(self.base_classifier, 'predict_proba') or isinstance(self.base_classifier, DecisionTreeClassifier):
                ek = np.sum(D * (pred_k != y[idx])) # maybe D multiplication is unnecessary here
                if ek < classification_base_error:
                    beta_k = (ek) / (1- ek)    
                else:
                    beta_k = (classification_base_error) / (1 - classification_base_error)
            else:
                raw_err = self.regression_task_error_metric(y, pred_k)
                ek = np.sum(D * raw_err)
                mean_regressor_error = np.mean(self.regression_task_error_metric(y, np.full_like(y, mean_y)))
                # mean_regressor_error = np.sum(D * mean_regressor_error) maybe unnecessary 

                if ek < mean_regressor_error:
                    beta_k = (ek) / (1- ek)    
                else:
                    beta_k = (mean_regressor_error) / (1 - mean_regressor_error)

            betas.append(beta_k)

        # Step 5: time-based weights (unused later)
        t_weights = np.array([1/(1+np.exp(-self.alpha*(t-k-self.beta)))
                               for k in range(1, len(self.models)+1)])
        t_weights /= t_weights.sum() 
        # t_weights rename to voting_weights
        # eqation 8 in the paper uses summation of t_weigthts in the next step, but it has little sense
        betas = t_weights * betas

        final_error = np.mean(self.regression_task_error_metric(y, self.predict(X)))
        # Step 6: pruning if ensemble_size exceeded
        if self.ensemble_size and len(self.models) > self.ensemble_size:
            # default removal count is excess models
            removal_count = len(self.models) - self.ensemble_size
            if self.removal_count_calculation:
                # record current error
                if not hasattr(self, 'error_history'):
                    self.error_history = []
                self.error_history.append(final_error)
                if len(self.error_history) >= 4:
                    e4, e3, e2, e1 = self.error_history[-4:]
                    r1 = e3 / e4 if e4 != 0 else 1.0
                    r2 = e2 / e3 if e3 != 0 else 1.0
                    r3 = e1 / e2 if e2 != 0 else 1.0
                    weighted = (self.gamma_multiplier * r1
                              + self.beta_multiplier  * r2
                              + self.alpha_multiplier * r3)
                    dynamic_count = max(0, round(self.fixed_multiplier * weighted - self.baseline))
                    print(f"[BatchErrorHistory] dynamic removal_count={dynamic_count}")
                    # ensure at least the default removal
                    removal_count = max(removal_count, dynamic_count)
                else:
                    print(f"[BatchErrorHistory] insufficient history ({len(self.error_history)}), skipping dynamic prune.")
            # remove models one at a time
            for _ in range(removal_count):
                if self.pruning_type == 'age':
                    # remove the oldest model
                    self.models.pop(0)
                    betas = betas[1:]
                    t_weights = t_weights[1:]
                else:
                    # error-based: remove model with largest beta
                    worst = np.argmax(betas)
                    self.models.pop(worst)
                    betas = np.delete(betas, worst)
                    t_weights = np.delete(t_weights, worst)

        # Step 7: voting weights
        eps = 1e-10
        
        self.voting_weights = np.log(1/betas)

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
