from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from src.trees.tree import BaseTree, Node
from src.trees.xnode_hf import NodeHoeffdingBound
from src.utils.utils import ensure_2d_array
from src.trees.split_criteria import BaseReduction, VarianceReduction, VarianceReductionRelative
from src.trees.histogram import get_histogram_type, HistogramContainer
from collections import defaultdict
import numpy as np
import pandas as pd
import tqdm
import time
import os

MAX_TREE_DEPTH = 999_999_999_999_999

class xFastODT(BaseTree):
    """
    Extended Fast Online Decision Tree (FastODT).

    A streaming decision tree that incrementally updates nodes using
    Hoeffding bounds, histogram-based binning, and variance reduction
    (or a user-specified split criterion). Designed for online learning
    with configurable depth, feature subsampling, and grace period
    strategies.

    Parameters
    ----------
    depth : int, default=25
        Maximum depth of the decision tree (number of splits from root to leaf).
    grace_period : int, default=10
        Number of samples a node must observe before attempting to split.
    split_confidence : float, default=0.01
        Confidence parameter for the Hoeffding bound. Smaller values allow
        more aggressive splitting; larger values make splitting more conservative.
    tie_threshold : float, default=0.05
        Threshold used to break ties when multiple splits have nearly equal gain.
    max_feature : float, default=1.0
        Fraction of features randomly sampled for each split (``(0,1]``).
    grace_period_depth : int, default=-1
        Depth control for applying the grace period:
        - ``1`` → only the root node uses grace period,
        - ``N`` → first ``N`` nodes use grace period,
        - ``-1`` → disabled (internally set to a very large value).
    splitter : callable, optional
        Function defining how splits are performed. If ``None``, defaults
        to an internal basic splitter.
    hist : object, optional
        Histogram container instance (e.g., ``HistogramContainer``).
        Provides per-feature binning for split candidate evaluation.
        If ``None``, defaults to ``HistogramContainer(max_bins=16, inherit_bins=False)``.
    split_criterion : callable, default=VarianceReduction
        Function to compute split quality. By default uses variance reduction.

    Attributes
    ----------
    depth : int
        Maximum tree depth.
    current_depth : int
        Current depth of the tree.
    grace_period : int
        Number of samples before checking for splits.
    split_confidence : float
        Confidence parameter for Hoeffding bound.
    tie_threshold : float
        Tie-breaking threshold for splitting.
    max_feature : float
        Fraction of features sampled for each split.
    grace_period_depth : int
        Depth until which grace period applies.
    depths_containers : list of dict
        Nested containers for managing nodes at each depth level.
    splits_values : list
        Stores split threshold values per depth.
    splits_features : list
        Stores feature indices per depth.
    splitter : callable
        Function responsible for splitting nodes.
    error_rates : list of float
        Stores recent error rates for adaptive control (length 20).
    mature_tree : bool
        Whether the tree has grown to its intended size/depth.
    root : NodeHoeffdingBound
        Root node of the tree.
    """

    def __init__(
        self,
        depth: int = 25,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        grace_period_depth: int = -1,
        splitter = None,

        ## things bin related
        hist = None,
        split_criterion: type[BaseReduction] = VarianceReductionRelative,

    ):
        self.grace_period: int = grace_period  # Number of samples before checking for a split
        self.split_confidence: float = split_confidence  # Confidence level for splitting
        self.depth: int = depth
        self.current_depth: int = 0
        self.splits_values: list[object] = [None] * depth
        self.splits_features: list[object] = [None] * depth
        self.depths_containers: list = []
        self.mature_tree: bool = False

        self.error_rates: list[float] = np.zeros(20).tolist()

        self.max_feature: int = max_feature
        self.grace_period_depth: int = grace_period_depth
        if self.grace_period_depth == -1:
            self.grace_period_depth = MAX_TREE_DEPTH

        self.tie_threshold = tie_threshold

        # create the root node
        for d in range(self.depth):
            self.depths_containers.append(defaultdict(lambda: None))

        if not splitter:
            splitter = lambda nodes, current_node : self.__basic_splitter(nodes, current_node)

        self.splitter = splitter

        if not hist:
            hist=HistogramContainer(
                max_bins=16,
                inherit_bins=False,
            )

        self.depths_containers[0][0] = NodeHoeffdingBound(
            split_confidence=self.split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            grace_period=grace_period,
            hist=hist,
            split_criterion=split_criterion,
        )

    def partial_fit(self, X: np.ndarray, y: np.ndarray, verbose: bool = False):
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
                self.__update_tree(None, X[i], y[i], depth=0)
        else:
            for i in range(len(X)):
                self.__update_tree(None, X[i], y[i], depth=0)

    def learn_one(self, x, y: float):
        """
        Incrementally update the tree with a single sample (x, y).
        x must be a 1D array-like structure that can be converted to a numpy array (or dict).
        """
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        elif isinstance(x, pd.DataFrame):
            x = np.asarray(x.values)
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        self.__update_tree(None, x, y, depth=0)

    def predict_one(self, x):
        """
        Predict the output for a single sample x.
        x must be a 1D array-like structure that can be converted to a numpy array (or dict).
        """
        if isinstance(x, dict):
            x = np.asarray(list(x.values()))
        else:
            x = np.asarray(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = x[0]
        return self.predict(x)

    def predict(self, X: np.ndarray):
        """Predicts y value."""

        return_vector = []
        for x in X:
            # Check if we reached the maximum depth
            active_splits_features = self.splits_features[: self.current_depth]
            active_splits_values = self.splits_values[: self.current_depth]
            decision_vector = x[active_splits_features] > active_splits_values

            node_index = sum((1 << i) for i, decision in enumerate(reversed(decision_vector)) if decision)
            node_depth = len(active_splits_features)

            node = self.depths_containers[node_depth][node_index]
            if node is None:
                node = self.__find_node(node_depth, node_index)
            return_vector.append(node.value)
        return return_vector

    def __basic_splitter(self, nodes, node):
        best_split, _ = node.find_best_split()  # here happens hf bound calculation
        return best_split

    def __update_tree(self, node, x: np.ndarray, y: float, depth=0):
        """Update the tree with a new sample."""

        # Take all currently known splits
        # At this point we should have all splits for self.current_depth
        active_splits_features = self.splits_features[: self.current_depth]
        active_splits_values = self.splits_values[: self.current_depth]

        # For each sample x, we take all features defined by `active_splits_features`
        # and we compre them against all known splits `active_splits_values`
        # that gives boolean decision vector of [true, false ,true , false, ...]
        decision_vector = x[active_splits_features] > active_splits_values

        # if decision vector is empty then there was no data passed
        if len(decision_vector) == 0:
            node_index = 0
            node_depth = 0
        else:
            # If decision vector was not empty then boolean values from
            # that vector [true, false ,true , false, ...] can be interpreted
            # as binary value 1010... which after changing into decimal will give proper result
            node_index = sum((1 << i) for i, decision in enumerate(reversed(decision_vector)) if decision)
            node_depth = len(active_splits_features)

        node = self.depths_containers[node_depth][node_index]
        if node is None:
            node = self.__find_node(node_depth, node_index)
        else:
            if self.mature_tree:
                return

        node.update_stats(x, y)

        # this can be used to find best splits with histograms on each level
        best_split = self.splitter(self.depths_containers[node_depth], node)

        if best_split[0] is not None:
            depth = len(decision_vector)

            self.splits_features[depth] = best_split[0]
            self.splits_values[depth] = best_split[1]

            if depth + 1 < self.depth:
                left, right = node.produce_split(best_split, self.split_confidence)
                left.update_from_parent_node(node)
                right.update_from_parent_node(node)
                self.depths_containers[depth + 1][left.index] = left
                self.depths_containers[depth + 1][right.index] = right
                self.current_depth += 1
            else:
                self.mature_tree = True

    def __find_node(self, depth, index):
        """Finds last populated node by index."""

        node = self.depths_containers[depth][index]
        # print(f"looking for index: {index}")
        if node is None:
            parent_node_index = (
                index // 2
            )  # find parent index by rule that it is child index / 2 and floored to nearest integer
            node = self.__find_node(depth - 1, parent_node_index)
            node.index = index
            self.depths_containers[depth][index] = node
            # this is cheap trick and needs to check if we initialize nodes correclty in this case
        return node

    def _attempt_split(self, node, depth: int):
        """Attempts splitting and producing two new nodes."""

        best_split, _ = node.find_best_split()  # here happens hf bound calculation
        if self.splits_features[depth] is not None:
            node.split(self.splits_features[depth], self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)

        elif best_split[0] is not None:
            self.splits_features[depth] = best_split
            node.split(best_split, self.split_confidence)
            node.left.update_from_parent_node(node)
            node.right.update_from_parent_node(node)


if __name__ == "__main__":
    X = np.load("./smoke_data/X.npy", allow_pickle=True)
    Y = np.load("./smoke_data/y.npy", allow_pickle=True)
    print(X.shape, Y.shape)
    X = X[:, 2:]
    # print(type(X), type(Y), X[0])

    # Test if model will be partially filled after receiving only one sample
    start = time.time()
    model_hf = xFastODT(depth=50, grace_period=10, split_confidence=0.0001)
    start = time.time()
    model_hf.partial_fit(X[:, :], Y, verbose=True)
    print(f"It took: {time.time() - start}s")

    Y_pred_hf = model_hf.predict(X)
    mae_hf = mean_absolute_error(Y, Y_pred_hf)
    print(f"MAE: {mae_hf:.4f}")

    PRINT_BUFFOR_START = 3000
    PRINT_BUFFOR_END = 3500

    plt.figure(figsize=(12, 6))
    plt.plot(Y[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Actual Y", color="black", linewidth=2)
    plt.plot(Y_pred_hf[PRINT_BUFFOR_START:PRINT_BUFFOR_END], label="Predicted Y", color="red", linewidth=2)

    plt.xlabel(f"Samples")
    plt.ylabel("Target Y")
    plt.title(f"FastODT Regression on smoke data.")
    plt.ylim(-10, 300)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
