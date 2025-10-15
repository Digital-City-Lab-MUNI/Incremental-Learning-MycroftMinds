"""Fast Online Decision Tree (FastODT).

This module defines the :class:`FastODT` incremental regression tree that uses
Hoeffding bound based split decisions with pluggable histogram summarizers and
split criteria. The tree can be trained sample-by-sample either via
``learn_one`` or ``partial_fit``, while maintaining statistics at each leaf and
deciding on splits once a "grace period" worth of observations has been
processed.

Key Features
------------
* Hoeffding bound controlled splitting with tie-breaking threshold.
* Support for multiple histogram binning strategies (adaptive, mean, etc.).
* Optional feature subsampling (``max_feature`` fraction) at split evaluation.
* Grace-period depth limitation (apply grace period only to shallow nodes).
* Rolling error tracking through ``measure_error``.

"""

from src.trees.tree import BaseTree
from src.trees.node_hf import NodeHoeffdingBound
from src.utils.utils import ensure_2d_array
from src.trees.split_criteria import get_split_criterion
from src.trees.histogram import get_histogram_type
import src.trees.constants as C
import numpy as np
import pandas as pd
import tqdm
import time
from typing import Sequence, Union


class FastODT(BaseTree):
    """Incremental regression tree using Hoeffding bound guided splits.

    Parameters
    ----------
    depth : int, optional, default=25
        Maximum depth (root at depth 0). Nodes at or beyond this depth are
        treated as leaves.
    grace_period : int, optional, default=10
        Minimum number of samples a leaf observes before evaluating a split.
    split_confidence : float, optional, default=0.01
        Hoeffding bound confidence (smaller values -> more eager splitting).
    tie_threshold : float, optional, default=0.05
        Threshold below which competing splits are forced to break ties.
    max_feature : float, optional, default=1.0
        Fraction (0,1] of features considered per split; 1.0 = all features.
    eagerness : float, optional, default=0.0
        Additional factor influencing node split aggressiveness (passed down to
        node implementation).
    grace_period_depth : int, optional, default=-1
        Depth up to which the grace period applies. ``-1`` disables the limit.
    bin_type : str or callable, optional, default="adaptive"
        Histogram strategy identifier or class implementing the required API.
    max_bins : int, optional, default=16
        Maximum bins per feature histogram.
    inherit_bins : bool, optional, default=False
        If True, child nodes receive copies of parent histograms on split.
    split_criterion : str or callable, optional, default="vr_rel"
        Split criterion identifier or function returning merit scores.
    random_state : {None, int, `numpy.random.Generator`}, optional
        Random generator or random seed for reproducibility. It has effect only
        when `max_feature` < 1.0. Pass a generator when FastODT is used in an ensemble.
        If None (default), the global NumPy RNG is used.

    Attributes
    ----------
    root : NodeHoeffdingBound
        Root node of the tree.
    splits : list
        Cached best split (feature, threshold, ...) per depth to encourage
        consistency among nodes at the same depth.
    error_rates : list of float
        Rolling window (length 20) of recent absolute errors for monitoring.
    grace_period : int
        Number of samples before checking for a split at each node.
    split_confidence : float
        Confidence level for splitting.
    depth : int
        Maximum depth of the tree.
    max_feature : float
        Fraction of features considered at each split.
    grace_period_depth : int
        Depth up to which the grace period applies.
    tie_threshold : float
        Threshold for tie-breaking between splits.
    rng : numpy.random.Generator
        Random number generator used for feature subsampling.
    """

    def __init__(
        self,
        depth: int = 25,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        eagerness: float = 0.0,
        grace_period_depth: int = -1,
        bin_type: Union[str, callable] = C.HistogramType.ADAPTIVE,
        max_bins: int = 16,
        inherit_bins: bool = False,
        split_criterion: Union[str, callable] = C.SplitCriterion.VR_REL,
        random_state: np.random.Generator | int = None,
    ) -> None:
        self.grace_period = grace_period  # Number of samples before checking for a split
        self.split_confidence = split_confidence  # Confidence level for splitting
        self.depth = depth
        self.splits = [None] * depth

        self.error_rates = np.zeros(20).tolist()

        self.max_feature = max_feature
        self.grace_period_depth = grace_period_depth
        if self.grace_period_depth == -1:
            self.grace_period_depth = 999999999999999

        self.tie_threshold = tie_threshold
        if isinstance(bin_type, str) or bin_type is None:
            bin_type = get_histogram_type(bin_type)

        if isinstance(split_criterion, str):
            split_criterion = get_split_criterion(split_criterion)

        # Initialize local RNG
        self.rng = np.random.default_rng(random_state)

        # create the root node
        self.root = NodeHoeffdingBound(
            split_confidence=self.split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            eagerness=eagerness,
            grace_period=grace_period,
            bin_type=bin_type,
            max_bins=max_bins,
            inherit_bins=inherit_bins,
            split_criterion=split_criterion,
            rng=self.rng,
        )

    def measure_error(self, y_baseline: float, y_predicted: float) -> float:
        """Update rolling error buffer and return its mean.

        Parameters
        ----------
        y_baseline : float
            Ground truth target value.
        y_predicted : float
            Model prediction.

        Returns
        -------
        float
            Mean of the rolling error window (length 20).
        """
        self.error_rates.pop(0)
        self.error_rates.append(y_baseline - y_predicted)
        return np.mean(self.error_rates)

    def partial_fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False) -> None:
        """Fit the model incrementally on a batch of samples.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix of shape (n_samples, n_features) or (n_features,) for
            a single sample.
        y : np.ndarray
            Target values of shape (n_samples,) or scalar.
        verbose : bool, default=False
            If True, displays a progress bar via ``tqdm``.
        """
        X = ensure_2d_array(X)  # 10.1 -> [10.1] -> (1,1)
        y = np.asarray(y)
        if y.ndim == 0:
            y = np.array([y])
        if X.shape[0] != len(y):
            raise ValueError(f"X and y must have the same number of samples. X: {X.shape}, y: {y.shape}")

        if verbose:
            for i in tqdm.tqdm(range(len(X))):
                self._update_tree(self.root, X[i], y[i], depth=0)
        else:
            for i in range(len(X)):
                self._update_tree(self.root, X[i], y[i], depth=0)

    def learn_one(self, x: Union[np.ndarray, Sequence[float], dict, pd.DataFrame], y: float) -> None:
        """Update the model with a single observation."""
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        elif isinstance(x, pd.DataFrame):
            x = np.asarray(x.values)
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        self._update_tree(self.root, x, y, depth=0)

    def predict_one(self, x: Union[np.ndarray, Sequence[float], dict]) -> float:
        """Predict target for a single sample."""
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        return self.predict(x)

    def _update_tree(self, node: NodeHoeffdingBound, x: np.ndarray, y: float, depth: int = 0) -> None:
        """Update the tree with a new sample."""
        # Check if we reached the maximum depth
        if depth >= self.depth:
            return

        # if node is a leaf, update its stats and check for split
        if node.left is None and node.right is None:
            node.update_stats(x, y)
            if node.reached_grace_period() or self.grace_period_depth < depth:  # check if grace period is over
                self._attempt_split(node, depth)

        # traverse the tree to reach the correct leaf
        elif node.feature is not None:
            if x[node.feature] <= node.threshold:
                self._update_tree(node.left, x, y, depth + 1)
            elif x[node.feature] > node.threshold:
                self._update_tree(node.right, x, y, depth + 1)

    def _attempt_split(self, node: NodeHoeffdingBound, depth: int) -> None:
        """Attempt to split the node when the Hoeffding Bound indicates a split is warranted."""
        best_split, _ = node.find_best_split()  # here happens hf bound calculation
        if self.splits[depth] is not None:
            node.split(self.splits[depth], self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)

        elif best_split[0] is not None:
            self.splits[depth] = best_split
            node.split(best_split, self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)


if __name__ == "__main__":
    import numpy as np
    from sklearn.metrics import mean_absolute_error
    from src.trees.histogram import MeanSplitterHistogram, AdaptiveHistogram
    from src.utils.memory import calculate_memory_usage
    import matplotlib.pyplot as plt

    print("================================NEW ATTEMPT================================")

    # Load smoke data
    X = np.load("./smoke_data/X.npy", allow_pickle=True)
    Y = np.load("./smoke_data/y.npy", allow_pickle=True)
    X = X[:, 2:]
    Y = np.clip(Y, 0, 200)
    PRINT_BUFFOR_START = 3000
    PRINT_BUFFOR_END = 3500
    split_confidences = [0.0001, 0.001, 0.01, 0.1, 0.5, 1]
    results = {
        "mean": [],
        "adapt4": [],
        "adapt8": [],
        "adapt16": [],
        "mae_mean": [],
        "mae_adapt4": [],
        "mae_adapt8": [],
        "mae_adapt16": [],
        "train_time_mean": [],
        "train_time_adapt4": [],
        "train_time_adapt8": [],
        "train_time_adapt16": [],
    }

    DEPTH = 50
    GRACE_PERIOD = 100
    GRACE_PERIOD_DEPTH = -1
    EAGERNESS = 0

    for split_conf in split_confidences:
        # MeanSplitterHistogram
        model_mean = FastODT(
            depth=DEPTH,
            grace_period=GRACE_PERIOD,
            split_confidence=split_conf,
            max_feature=1.0,
            eagerness=EAGERNESS,
            grace_period_depth=GRACE_PERIOD_DEPTH,
            bin_type=MeanSplitterHistogram,
            max_bins=1,
        )
        t_train_start = time.time()
        model_mean.partial_fit(X, Y)
        t_train_end = time.time()
        train_time_mean = t_train_end - t_train_start
        results["train_time_mean"].append(train_time_mean)
        t0 = time.time()
        size_mean = calculate_memory_usage(model_mean)
        t1 = time.time()
        print(
            f"MeanSplitterHistogram model size with deepness {DEPTH} "
            f"and split confidence {split_conf}: {size_mean} MB "
            f"(size calc time: {t1 - t0:.3f}s)"
        )
        print(f"MeanSplitterHistogram training time: {train_time_mean:.3f}s")
        y_pred_mean = np.array([model_mean.predict(x) for x in X])
        mae_mean = mean_absolute_error(Y, y_pred_mean)
        results["mean"].append(y_pred_mean)
        results["mae_mean"].append(mae_mean)

        # AdaptiveHistogram (max_bins=4)
        model_adapt4 = FastODT(
            depth=DEPTH,
            grace_period=GRACE_PERIOD,
            split_confidence=split_conf,
            max_feature=1.0,
            eagerness=EAGERNESS,
            grace_period_depth=GRACE_PERIOD_DEPTH,
            bin_type=AdaptiveHistogram,
            max_bins=4,
        )
        t_train_start = time.time()
        model_adapt4.partial_fit(X, Y)
        t_train_end = time.time()
        train_time_adapt4 = t_train_end - t_train_start
        results["train_time_adapt4"].append(train_time_adapt4)
        t0 = time.time()
        size_adapt4 = calculate_memory_usage(model_adapt4)
        t1 = time.time()
        print(
            f"AdaptiveHistogram (4 bins) model size with deepness {DEPTH} "
            f"and split confidence {split_conf}: {size_adapt4} MB "
            f"(size calc time: {t1 - t0:.3f}s)"
        )
        print(f"AdaptiveHistogram (4 bins) training time: {train_time_adapt4:.3f}s")
        y_pred_adapt4 = np.array([model_adapt4.predict(x) for x in X])
        mae_adapt4 = mean_absolute_error(Y, y_pred_adapt4)
        results["adapt4"].append(y_pred_adapt4)
        results["mae_adapt4"].append(mae_adapt4)

        # AdaptiveHistogram (max_bins=8)
        model_adapt8 = FastODT(
            depth=DEPTH,
            grace_period=GRACE_PERIOD,
            split_confidence=split_conf,
            max_feature=1.0,
            eagerness=EAGERNESS,
            grace_period_depth=GRACE_PERIOD_DEPTH,
            bin_type=AdaptiveHistogram,
            max_bins=8,
        )
        t_train_start = time.time()
        model_adapt8.partial_fit(X, Y)
        t_train_end = time.time()
        train_time_adapt8 = t_train_end - t_train_start
        results["train_time_adapt8"].append(train_time_adapt8)
        t0 = time.time()
        size_adapt8 = calculate_memory_usage(model_adapt8)
        t1 = time.time()
        print(
            f"AdaptiveHistogram (8 bins) model size with deepness {DEPTH} "
            f"and split confidence {split_conf}: {size_adapt8} MB "
            f"(size calc time: {t1 - t0:.3f}s)"
        )
        print(f"AdaptiveHistogram (8 bins) training time: {train_time_adapt8:.3f}s")
        y_pred_adapt8 = np.array([model_adapt8.predict(x) for x in X])
        mae_adapt8 = mean_absolute_error(Y, y_pred_adapt8)
        results["adapt8"].append(y_pred_adapt8)
        results["mae_adapt8"].append(mae_adapt8)

        # AdaptiveHistogram (max_bins=16)
        model_adapt16 = FastODT(
            depth=DEPTH,
            grace_period=GRACE_PERIOD,
            split_confidence=split_conf,
            max_feature=1.0,
            eagerness=EAGERNESS,
            grace_period_depth=GRACE_PERIOD_DEPTH,
            bin_type=AdaptiveHistogram,
            max_bins=16,
        )
        t_train_start = time.time()
        model_adapt16.partial_fit(X, Y)
        t_train_end = time.time()
        train_time_adapt16 = t_train_end - t_train_start
        results["train_time_adapt16"].append(train_time_adapt16)
        t0 = time.time()
        size_adapt16 = calculate_memory_usage(model_adapt16)
        t1 = time.time()
        print(
            f"AdaptiveHistogram (16 bins) model size with deepness {DEPTH} "
            f"and split confidence {split_conf}: {size_adapt16} MB "
            f"(size calc time: {t1 - t0:.3f}s)"
        )
        print(f"AdaptiveHistogram (16 bins) training time: {train_time_adapt16:.3f}s")
        y_pred_adapt16 = np.array([model_adapt16.predict(x) for x in X])
        mae_adapt16 = mean_absolute_error(Y, y_pred_adapt16)
        results["adapt16"].append(y_pred_adapt16)
        results["mae_adapt16"].append(mae_adapt16)

        print(
            f"Split confidence {split_conf} - MAE: "
            f"Mean={mae_mean:.4f}, Adapt4={mae_adapt4:.4f}, "
            f"Adapt8={mae_adapt8:.4f}, Adapt16={mae_adapt16:.4f}\n"
        )

        # After training, print number of bins in root node for each feature
        # print(f"[SplitConf={split_conf}] MeanSplitter bins per feature:")
        # for f, h in model_mean.root.histograms.items():
        #     print(f"  Feature {f}: {len(getattr(h, 'bin_edges', []))} bins")

        # print(f"[SplitConf={split_conf}] AdaptiveHistogram (4 bins) bins per feature:")
        # for f, h in model_adapt4.root.histograms.items():
        #     print(f"  Feature {f}: {len(getattr(h, 'bin_edges', []))} bins")

        # print(f"[SplitConf={split_conf}] AdaptiveHistogram (8 bins) bins per feature:")
        # for f, h in model_adapt8.root.histograms.items():
        #     print(f"  Feature {f}: {len(getattr(h, 'bin_edges', []))} bins")

        # print(f"[SplitConf={split_conf}] AdaptiveHistogram (16 bins) bins per feature:")
        # for f, h in model_adapt16.root.histograms.items():
        #     print(f"  Feature {f}: {len(getattr(h, 'bin_edges', []))} bins")

    # Plot predictions for each split confidence
    for i, split_conf in enumerate(split_confidences):
        plt.figure(figsize=(12, 5))
        plt.plot(
            Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END],
            label="Actual Y",
            color="black",
            linewidth=2,
        )
        plt.plot(
            results["mean"][i][PRINT_BUFFOR_START:PRINT_BUFFOR_END],
            label="MeanSplitter",
        )
        plt.plot(
            results["adapt4"][i][PRINT_BUFFOR_START:PRINT_BUFFOR_END],
            label="AdaptiveHistogram (4 bins)",
            linestyle="-",
            alpha=0.7,
        )
        plt.plot(
            results["adapt8"][i][PRINT_BUFFOR_START:PRINT_BUFFOR_END],
            label="AdaptiveHistogram (8 bins)",
            linestyle="-.",
            alpha=0.7,
        )
        plt.plot(
            results["adapt16"][i][PRINT_BUFFOR_START:PRINT_BUFFOR_END],
            label="AdaptiveHistogram (16 bins)",
            linestyle="--",
            alpha=0.7,
        )
        plt.xlabel("Samples")
        plt.ylabel("Target Y")
        plt.title(f"Predictions (Split Confidence={split_conf})")
        plt.ylim(0, 300)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()
