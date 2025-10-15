import math
import numpy as np
from collections import defaultdict
from src.utils.utils import get_unique_random_numbers
from src.trees.histogram import AdaptiveHistogram, EquiWidthHistogram
from src.trees.split_criteria import variance_reduction, variance_reduction_relative


def hoeffding_bound(R: float, n: int, delta: float) -> float:
    """
    Computes Hoeffding bound given number of samples and confidence level.

    Parameters
    ----------
    R: float
        Range of the split criterion (e.g., variance reduction).
    n: int
        number of samples seen so far
    delta: float
        confidence level (e.g., 0.01 for 99% confidence)
    """
    return np.sqrt((R**2) * np.log(1 / delta) / (2 * n))


class NodeHoeffdingBound:  # one leaf of the FastODT
    """
    Incremental tree node with histogram-based binning for split selection.

    Parameters
    ----------
    split_confidence : float
        Confidence level for splitting criterion.
    tie_threshold : float
        Threshold for tie-breaking in split selection. Default is 0.05.
    max_feature : float
        Fraction of features to consider for splitting at each node.
    eagerness : float
        Eagerness parameter (currently unused). Default is 0.0.
    grace_period : int
        Number of samples a node should observe before attempting to split.
    index : int
        Node index in the tree structure. Default is 0.
    bin_type : class
        Histogram class to use for binning (e.g., AdaptiveHistogram, EquiWidthHistogram).
    max_bins : int
        Maximum number of bins per feature per node.
    inherit_bins : bool
        Whether to inherit bins from parent node on split.
    split_criterion : callable
        Function to compute split quality (default: variance reduction).
    rng : np.random.Generator
        Random number generator for reproducibility. It has effect only when max_feature < 1.0

    Attributes
    ----------
    index : int
        Node index in the tree structure.
    tie_threshold : float
        Threshold for tie-breaking in split selection.
    delta : float
        Confidence level for the Hoeffding bound (same as split_confidence).
    samples_seen : int
        Total number of samples seen by this node.
    y_max : float
        Maximum target value seen by this node.
    y_min : float
        Minimum target value seen by this node.
    sum_y : float
        Sum of target values for samples seen by this node.
    sum_y_squared : float
        Sum of squared target values for samples seen by this node.
    value : float
        Average target value for samples seen by this node.
    left : NodeHoeffdingBound
        Left child node (None if leaf).
    right : NodeHoeffdingBound
        Right child node (None if leaf).
    feature : int
        Index of the feature used for splitting.
    threshold : float
        Threshold value for the split on the feature.
    grace_period : int
        Number of samples a node should observe before attempting to split.
    grace_period_offset : int
        Offset for the grace period, used to determine when to split.
    max_feature : float
        Fraction of features to consider for splitting at each node.
    feature_indices : np.ndarray
        Randomly selected feature indices based on `max_feature`.
    bin_type : callable
        Histogram class to use for binning.
    max_bins : int
        Maximum number of bins per feature per node.
    histograms : dict
        Per-feature histograms for binning and split candidate generation.
    split_criterion : callable
        Function to compute split quality.
    inherit_bins : bool
        Whether to inherit bins from parent node on split.
    rng : np.random.Generator
        Random number generator for reproducibility.
    """

    # TODO:
    # reduction of memory footprint

    def __init__(
        self,
        split_confidence,
        tie_threshold=0.05,
        max_feature=1.0,
        eagerness=0.0,
        grace_period=0,
        index=0,
        bin_type=AdaptiveHistogram,
        max_bins=32,
        inherit_bins: bool = False,
        split_criterion: callable = None,
        rng: np.random.Generator = None,
    ):
        self.index: int = index

        self.tie_threshold: float = tie_threshold
        self.delta: float = split_confidence
        self.samples_seen: int = 0
        self.y_max: float = -np.inf
        self.y_min: float = np.inf
        self.sum_y: float = 0
        self.sum_y_squared: float = 0
        self.value: float = (
            0  # average target value for samples seen by this node, this value is used for leaf prediction
        )
        self.left: NodeHoeffdingBound = None  # None means it's a leaf node
        self.right: NodeHoeffdingBound = None  # None means it's a leaf node
        self.feature: int = None  # splitting feature index in the dataset (e.g. feature 'weight' with index 0)
        self.threshold: float = None  # splitting feature threshold (e.g. split weight at 33.3)

        # 200 is magic number but it is hyperparam which makes hf work for us
        self.grace_period: int = grace_period
        self.grace_period_offset: int = 0  # this is the offset from the parent's `samples_seen` that's passed to child for proper grace period calculation

        self.max_feature: float = max_feature
        self.feature_indices: np.ndarray = None

        self.bin_type: callable = bin_type  # e.g. 'adaptive', 'equiwidth', 'mean'
        self.max_bins: int = max_bins
        self.histograms: dict = None  # Will be initialized after seeing first sample

        self.split_criterion: callable = split_criterion if split_criterion is not None else variance_reduction
        self.inherit_bins: bool = inherit_bins
        self.rng: np.random.Generator = rng

    def check_grace_period(self):
        """Check if grace period is met."""
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def update_stats(self, x: np.ndarray, y: float) -> None:
        """Update the statistics of the node's histograms with a new sample."""

        # Generate random feature indices based on max_feature if not already set
        if self.feature_indices is None:
            max_feature_length = int(x.shape[0] * self.max_feature)
            self.feature_indices = np.sort(
                get_unique_random_numbers(max_feature_length, x.shape[0], self.rng.integers(1e6) if self.rng else None)
                )

        # Select only the features that are in the feature_indices
        x_shortlisted = x[self.feature_indices]

        # Update node statistics
        self.samples_seen += 1
        self.y_max = max(self.y_max, y)
        self.y_min = min(self.y_min, y)
        self.sum_y += y
        self.sum_y_squared += y**2
        self.value = self.sum_y / self.samples_seen  # Average target value

        # Initialize histograms if needed
        if self.histograms is None:
            self.histograms = {}
            for feature in self.feature_indices:
                self.histograms[feature] = self.bin_type(max_bins=self.max_bins)

        # Update histograms for each feature
        for feature, val in zip(self.feature_indices, x_shortlisted):
            self.histograms[feature].insert(val, y)

    def update_from_parent_node(self, parent) -> None:
        """Update this node's statistics from a parent node."""
        self.value = parent.value
        self.rng = parent.rng

    def _hoeffding_R(self) -> float:
        """
        Calculate the Hoeffding bound R value based on the variance reduction used.
        For relative reduction, R is fixed to 1.0 (as the range is normalized).
        For absolute reduction, R is calculated as 0.25 * (y_max - y_min) ** 2, which is the Popoviciu upper bound.
        """
        # fixed range
        if self.split_criterion == variance_reduction_relative:
            return 1.0

        # absolute range
        elif self.split_criterion == variance_reduction:
            return 0.25 * (self.y_max - self.y_min) ** 2  # Popoviciu upper bound

    def find_best_split(self) -> tuple:
        """Find the best split based on current histograms and splitting criteria.
        We use Hoeffding bound to determine if the split is statistically significant."""

        best_split_1st = (None, None, 0)
        best_split_2nd = (None, None, 0)

        for feature, hist in (self.histograms or {}).items():
            candidates = hist.get_candidates()
            for thr in candidates:
                left_stats = hist.aggregate_stats(thr, left=True)
                right_stats = hist.aggregate_stats(thr, left=False)
                reduction = self.split_criterion(
                    left_stats, right_stats, self.samples_seen, self.sum_y, self.sum_y_squared
                )
                if reduction > best_split_1st[2]:
                    best_split_2nd = best_split_1st
                    best_split_1st = (feature, thr, reduction)
                elif reduction > best_split_2nd[2] and reduction < best_split_1st[2]:
                    best_split_2nd = (feature, thr, reduction)

        # Apply Hoeffding bound after considering all features
        R = self._hoeffding_R()  # range based on absolute or relative reduction
        e = hoeffding_bound(R, self.samples_seen, self.delta)

        if best_split_1st[2] - best_split_2nd[2] > e or self.tie_threshold > e:
            return (best_split_1st[0], best_split_1st[1]), best_split_1st[2]

        return (None, None), 0

    def split(self, split_info, split_confidence):
        """Split the node based on the best split found."""
        left, right = self.produce_split(split_info, split_confidence)
        self.left = left
        self.right = right
        # If not inheriting, children will initialize new empty histograms on first update_stats

    def produce_split(self, split_info, split_confidence):
        """Split the node based on the best split found."""
        self.feature, self.threshold = split_info
        left = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            grace_period=self.grace_period,
            index=2 * self.index,  # left should have index same as parent
            bin_type=self.bin_type,
            max_bins=self.max_bins,
            inherit_bins=self.inherit_bins,
            split_criterion=self.split_criterion,
            rng=self.rng,
        )
        right = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            grace_period=self.grace_period,
            index=2 * self.index + 1,  # right should have index same as parent + 1
            bin_type=self.bin_type,
            max_bins=self.max_bins,
            inherit_bins=self.inherit_bins,
            split_criterion=self.split_criterion,
            rng=self.rng,
        )

        # TODO: this should be done inside histogram splitter class
        if self.inherit_bins and self.histograms is not None:
            left.histograms = {}
            right.histograms = {}
            for feature, hist in self.histograms.items():
                if feature == self.feature:
                    left.histograms[feature] = hist.partition_bins(self.threshold, left=True)
                    right.histograms[feature] = hist.partition_bins(self.threshold, left=False)
                else:
                    left.histograms[feature] = hist.copy_bins()
                    right.histograms[feature] = hist.copy_bins()
        return left, right

    def reached_grace_period(self) -> bool:
        """Check if the node has seen enough samples to consider splitting."""
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def show(self, prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")


if __name__ == "__main__":
    print("Testing NodeHoeffdingBound with AdaptiveHistogram...")
    np.random.seed(42)
    X = np.random.uniform(0, 10, (100, 2))
    y = X[:, 0] * 2 + np.random.normal(0, 1, 100)

    node = NodeHoeffdingBound(
        split_confidence=0.01, max_feature=1.0, grace_period=10, bin_type=AdaptiveHistogram, max_bins=4
    )
    for i in range(20):
        node.update_stats(X[i], y[i])
    print("Samples seen:", node.samples_seen)
    print("Histograms:")
    for f, h in node.histograms.items():
        print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")

    best_split, gain = node.find_best_split()
    print("Best split:", best_split, "Gain:", gain)

    print("Splitting node...")
    node.split(best_split, split_confidence=0.01)
    print("Left child histograms:")
    # for f, h in node.left.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")
    # print("Right child histograms:")
    # for f, h in node.right.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")

    print("\n\nTesting NodeHoeffdingBound with MeanSplitterHistogram (average splitter)...")
    from src.trees.histogram import MeanSplitterHistogram

    node_mean = NodeHoeffdingBound(
        split_confidence=0.01, max_feature=1.0, grace_period=10, bin_type=MeanSplitterHistogram, max_bins=1
    )
    for i in range(20):
        node_mean.update_stats(X[i], y[i])
    print("Samples seen (mean splitter):", node_mean.samples_seen)
    print("Histograms (mean splitter):")
    for f, h in node_mean.histograms.items():
        print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")
    best_split_mean, gain_mean = node_mean.find_best_split()
    print("Best split (mean splitter):", best_split_mean, "Gain:", gain_mean)
    print("Splitting node (mean splitter)...")
    node_mean.split(best_split_mean, split_confidence=0.01)
    # print("Left child histograms (mean splitter):")
    # for f, h in node_mean.left.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")
    # print("Right child histograms (mean splitter):")
    # for f, h in node_mean.right.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")

    # print("Testing with EquiWidthHistogram...")
    # node2 = NodeHoeffdingBound(
    #     split_confidence=0.01,
    #     max_feature=1.0,
    #     grace_period=10,
    #     bin_class=EquiWidthHistogram,
    #     bin_params={'max_bins': 8, 'min_val': 0, 'max_val': 10}
    # )
    # for i in range(20):
    #     node2.update_stats(X[i], y[i])
    # print("EquiWidth Histograms:")
    # for f, h in node2.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")

    # best_split2, gain2 = node2.find_best_split()
    # print("Best split (EquiWidth):", best_split2, "Gain:", gain2)

    # print("Splitting node2...")
    # node2.split(best_split2, split_confidence=0.01)
    # print("Left child histograms (EquiWidth):")
    # for f, h in node2.left.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")
    # print("Right child histograms (EquiWidth):")
    # for f, h in node2.right.histograms.items():
    #     print(f"Feature {f}: bin_edges={h.bin_edges}, bin_stats={[s['count'] for s in h.bin_stats]}")
