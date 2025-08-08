from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from src.trees.fastodt import FastODT

# from src.utils.adwin import ADWIN
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np
import tqdm
import time
import copy
import os


class IncubationBoost:

    def __init__(
        self,
        length: int = 5,
        learning_rate: float = 1.0,
        base_learner=FastODT,
        depth: int = 15,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        grace_period_depth: int = -1,
        eagerness: float = 0.0,
        bin_type: str = "adaptive",
        max_bins: int = 32,
        split_criterion: str = "vr_rel",
        verbose: bool = True,
    ):
        """
        Initializes IncubationBoost.

        Parameters:
            base_learner (object): The base learning model. Default is FastODT.
            length (int, optional): The max length of boosting chain - after this nodes are being replaced instead of addded.
            learning_rate (float, optional): The learning rate for the boosting process.
            base_learner_params (dict, optional): Options for the base learner, such as depth, grace_period, split_confidence, max_feature, grace_period_depth, eagerness, bin_type, max_bins, and split_criterion.

        """
        self.chain = []
        self.length: int = length
        self.learning_rate: float = learning_rate
        self.base_learner = base_learner
        self.depth = depth
        self.grace_period = grace_period
        self.split_confidence = split_confidence
        self.max_feature = max_feature
        self.grace_period_depth = grace_period_depth
        self.tie_threshold = tie_threshold
        self.eagerness = eagerness
        self.bin_type = bin_type
        self.max_bins = max_bins
        self.split_criterion = split_criterion
        self.incubating_node = None
        # self.drift_detector = ADWIN(delta = 0.000002)
        # self.under_drift_latch = False
        self.verbose = verbose
        try:
            # Check if running in a Jupyter notebook
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None and "IPKernelApp" in ipython.config:
                self.tqdm = tqdm.notebook.tqdm
            else:
                self.tqdm = tqdm.tqdm
        except Exception:
            # Fallback in case something goes wrong
            self.tqdm = tqdm.tqdm

    def partial_fit(self, X, y):
        if self.verbose == False:
            for i in range(len(X)):
                self._partial_fit(X[i], y[i])
                # input()
        else:
            for i in self.tqdm(range(len(X))):
                self._partial_fit(X[i], y[i])
                # input()

    def _partial_fit(self, X, y):
        residual = y
        y_pred_strong = 0.0

        pruning_index = None
        prev_pruning_error_rate = 9999.0

        # self.drift_detector.add_sample(X)
        # self.under_drift_latch = True
        # detect if drift happens - if drift already happened then keep it latched (with +) until new tree is deployed
        # self.under_drift_latch = bool(self.drift_detector.isDrift() + self.under_drift_latch)

        # if the chain is empty, create the first node
        if len(self.chain) == 0:
            newNode = self.base_learner(
                depth=self.depth,
                grace_period=self.grace_period,
                grace_period_depth=self.grace_period_depth,
                split_confidence=self.split_confidence,
                max_feature=self.max_feature,
                tie_threshold=self.tie_threshold,
                bin_type=self.bin_type,
                max_bins=self.max_bins,
                split_criterion=self.split_criterion,
            )
            self.chain.append(newNode)

        # Index-based pruning (eGBDT)
        for n, node in enumerate(self.chain):
            y_pred = node.predict(X) * (
                1.0 if n == 0 else self.learning_rate
            )  # first node is not multiplied by learning rate
            y_pred_strong += y_pred
            residual = y - y_pred_strong

            if node.is_filled():
                error_rate = node.measure_error(y, y_pred_strong)
                if prev_pruning_error_rate > error_rate:
                    pruning_index = n
                    prev_pruning_error_rate = error_rate
                else:
                    break
            else:
                node.partial_fit([X], residual)

        if pruning_index is not None:
            self.chain = self.chain[: pruning_index + 1]

        hatched_node = self.incubate(X, residual)
        if hatched_node and len(self.chain) < self.length:
            self.chain.append(hatched_node)

    def incubate(self, X, residual):
        if self.incubating_node == None:
            self.incubating_node = self.base_learner(
                depth=self.depth,
                grace_period=self.grace_period,
                grace_period_depth=self.grace_period_depth,
                split_confidence=self.split_confidence,
                max_feature=self.max_feature,
                tie_threshold=self.tie_threshold,
                bin_type=self.bin_type,
                max_bins=self.max_bins,
                split_criterion=self.split_criterion,
            )
            self.incubating_node.partial_fit([X], residual)

        else:
            self.incubating_node.partial_fit([X], residual)

            if self.incubating_node.is_filled():
                hatched_node = self.incubating_node
                self.incubating_node = None
                return hatched_node
            else:
                return None

    def predict(self, X):
        y = 0.0
        for n, node in enumerate(self.chain):
            y += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
        return y


class IncubationBoostAnyModel:

    def __init__(
        self,
        length: int = 5,
        learning_rate: float = 1.0,
        base_learner=FastODT,
        base_learner_params: dict = {
            "depth": 15,
            "grace_period": 10,
            "split_confidence": 0.01,
            "max_feature": 1.0,
            "grace_period_depth": -1,
            "eagerness": 0.0,
            "bin_type": "adaptive",
            "max_bins": 32,
            "split_criterion": "vr_rel",
        },
        verbose: bool = True,
    ):
        """
        Initializes IncubationBoost.

        Parameters:
            base_learner (object): The base learning model. Default is FastODT.
            length (int, optional): The max length of boosting chain - after this nodes are being replaced instead of addded.
            learning_rate (float, optional): The learning rate for the boosting process.
            base_learner_params (dict, optional): Options for the base learner, such as depth, grace_period, split_confidence, max_feature, grace_period_depth, eagerness, bin_type, max_bins, and split_criterion.

        """
        self.chain = []
        self.length: int = length
        self.learning_rate: float = learning_rate
        self.base_learner = base_learner
        self.base_learner_params: dict = base_learner_params
        self.incubating_node = None
        # self.drift_detector = ADWIN(delta = 0.000002)
        # self.under_drift_latch = False
        self.verbose = verbose
        try:
            # Check if running in a Jupyter notebook
            from IPython import get_ipython

            ipython = get_ipython()
            if ipython is not None and "IPKernelApp" in ipython.config:
                self.tqdm = tqdm.notebook.tqdm
            else:
                self.tqdm = tqdm.tqdm
        except Exception:
            # Fallback in case something goes wrong
            self.tqdm = tqdm.tqdm

    def partial_fit(self, X, y):
        if self.verbose == False:
            for i in range(len(X)):
                self._partial_fit(X[i], y[i])
                # input()
        else:
            for i in self.tqdm(range(len(X))):
                self._partial_fit(X[i], y[i])
                # input()

    def _partial_fit(self, X, y):
        residual = y
        y_pred_strong = 0.0

        pruning_index = None
        prev_pruning_error_rate = 9999.0

        # self.drift_detector.add_sample(X)
        # self.under_drift_latch = True
        # detect if drift happens - if drift already happened then keep it latched (with +) until new tree is deployed
        # self.under_drift_latch = bool(self.drift_detector.isDrift() + self.under_drift_latch)

        # if the chain is empty, create the first node
        if len(self.chain) == 0:
            newNode = self.base_learner(**self.base_learner_params)
            self.chain.append(newNode)

        # Index-based pruning (eGBDT)
        for n, node in enumerate(self.chain):
            y_pred = node.predict(X) * (
                1.0 if n == 0 else self.learning_rate
            )  # first node is not multiplied by learning rate
            y_pred_strong += y_pred
            residual = y - y_pred_strong

            if node.is_filled():
                error_rate = node.measure_error(y, y_pred_strong)
                if prev_pruning_error_rate > error_rate:
                    pruning_index = n
                    prev_pruning_error_rate = error_rate
                else:
                    break
            else:
                node.partial_fit([X], residual)

        if pruning_index is not None:
            self.chain = self.chain[: pruning_index + 1]

        hatched_node = self.incubate(X, residual)
        if hatched_node and len(self.chain) < self.length:
            self.chain.append(hatched_node)

    def incubate(self, X, residual):
        if self.incubating_node == None:
            self.incubating_node = self.base_learner(**self.base_learner_params)
            self.incubating_node.partial_fit([X], residual)

        else:
            self.incubating_node.partial_fit([X], residual)

            if self.incubating_node.is_filled():
                hatched_node = self.incubating_node
                self.incubating_node = None
                return hatched_node
            else:
                return None

    def predict(self, X):
        y = 0.0
        for n, node in enumerate(self.chain):
            y += node.predict(X) * (1.0 if n == 0 else self.learning_rate)
        return y
