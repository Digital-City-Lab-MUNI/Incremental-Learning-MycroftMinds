import math
import numpy as np
from collections import defaultdict
from src.utils.utils import get_unique_random_numbers
from src.trees.histogram import AdaptiveHistogram, EquiWidthHistogram, HistogramContainer
from src.trees.split_criteria import BaseReduction, VarianceReduction, VarianceReductionRelative


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
    # harmonic_mean_n = 0
    # if n_left == 0:
    #     harmonic_mean_n = n_right
    # if n_right == 0:
    #     harmonic_mean_n = n_left
    # if n_left > 0 and n_right > 0:
    #     harmonic_mean_n = 1/((1/n_left)+(1/n_right))
    # return math.sqrt(1/(2*harmonic_mean_n) / delta)

class NodeHoeffdingBound:  # one leaf of the FastODT
    """
    Incremental tree node with histogram-based split selection.

    Maintains per-feature histograms for candidate thresholds, applies a
    variance-reduction criterion (by default), and uses a Hoeffding bound to
    decide when a statistically reliable split can be performed.

    Parameters
    ----------
    split_confidence : float
        Confidence parameter ``delta`` for the Hoeffding test.
    tie_threshold : float, default=0.05
        Tie-breaking threshold when best and second-best gains are close.
    max_feature : float, default=1.0
        Fraction of features randomly sampled for this node (``(0,1]``).
    grace_period : int, default=0
        Minimum samples observed before considering a split.
    index : int, default=0
        Unique index or identifier for the node.
    hist : type, default=HistogramContainer
        Histogram container class used to maintain per-feature distributions.
    split_criterion : callable, default=VarianceReduction
        Function to compute split quality.

    Attributes
    ----------
    index : int
        Identifier of this node within the tree.
    delta : float
        Confidence parameter for Hoeffding bound.
    tie_threshold : float
        Tie-breaking threshold when best and second-best gains are close.
    samples_seen : int
        Total number of samples observed at this node.
    y_max : float
        Maximum target value seen.
    y_min : float
        Minimum target value seen.
    sum_y : float
        Sum of target values observed.
    sum_y_squared : float
        Sum of squared target values observed.
    value : float
        Average target value for samples seen (used for leaf prediction).
    left : NodeHoeffdingBound or None
        Left child node; ``None`` means this is a leaf.
    right : NodeHoeffdingBound or None
        Right child node; ``None`` means this is a leaf.
    feature : int or None
        Splitting feature index.
    threshold : float or None
        Splitting feature threshold.
    grace_period : int
        Minimum samples observed before considering a split.
    grace_period_offset : int
        Offset from parent's ``samples_seen`` for proper grace period calculation.
    max_feature : float
        Fraction of features randomly sampled for this node.
    feature_indices : np.ndarray or None
        Indices of features considered for splitting.
    split_criterion : callable
        Function to evaluate split quality.
    hist : type
        Histogram container class used by this node.
    """
    def __init__(
        self,
        split_confidence,
        tie_threshold=0.05,
        max_feature=1.0,
        grace_period=0,
        index=0,
        hist=HistogramContainer,
        split_criterion: type[BaseReduction] = VarianceReductionRelative,
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

        self.split_criterion: callable = split_criterion
        self.hist = hist

    def check_grace_period(self):
        """Check if grace period is met."""
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def update_stats(self, x: np.ndarray, y: float) -> None:
        """Update the statistics of the node's histograms with a new sample."""

        # Generate random feature indices based on max_feature if not already set
        if self.feature_indices is None:
            max_feature_length = int(x.shape[0] * self.max_feature)
            self.feature_indices = np.sort(get_unique_random_numbers(max_feature_length, x.shape[0]))

        # Select only the features that are in the feature_indices
        x_shortlisted = x[self.feature_indices]

        # Update node statistics
        self.samples_seen += 1
        self.y_max = max(self.y_max, y)
        self.y_min = min(self.y_min, y)
        self.sum_y += y
        self.sum_y_squared += y**2
        self.value = self.sum_y / self.samples_seen  # Average target value

        self.hist.update_stats(self.feature_indices,x_shortlisted,y)

    def update_from_parent_node(self, parent) -> None:
        """Update this node's statistics from a parent node."""
        self.value = parent.value

    def _hoeffding_R(self) -> float:
        """
        Calculate the Hoeffding bound R value based on the variance reduction used.
        For relative reduction, R is fixed to 1.0 (as the range is normalized).
        For absolute reduction, R is calculated as 0.25 * (y_max - y_min) ** 2, which is the Popoviciu upper bound.
        """
        # fixed range
        return self.split_criterion.R(self.y_min, self.y_max)

    def find_best_split(self) -> tuple:
        """Find the best split based on current histograms and splitting criteria.
        We use Hoeffding bound to determine if the split is statistically significant."""

        # Apply Hoeffding bound after considering all features
        R = self._hoeffding_R()  # range based on absolute or relative reduction
        e = hoeffding_bound(R, self.samples_seen, self.delta)

        best_split_1st, best_split_2nd = self.hist.find_best_split(
            self.split_criterion,
            self.samples_seen,
            self.sum_y,
            self.sum_y_squared
        )

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
        hist_left, hist_right = self.hist.split(
            self.feature,
            self.threshold
        )

        left = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            grace_period=self.grace_period,
            index=2 * self.index,  # left should have index same as parent
            split_criterion=self.split_criterion,
            hist = hist_left
        )
        right = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            grace_period=self.grace_period,
            index=2 * self.index + 1,  # right should have index same as parent + 1
            split_criterion=self.split_criterion,
            hist = hist_right
        )

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

    histContainer = HistogramContainer(
        bin_type=AdaptiveHistogram, 
        max_bins=4
    )

    node = NodeHoeffdingBound(
        split_confidence=0.01, 
        max_feature=1.0, 
        grace_period=10, 
        hist=histContainer
    )
    for i in range(20):
        node.update_stats(X[i], y[i])
    print("Samples seen:", node.samples_seen)
    print("Histograms:")
    for f, h in node.hist.histograms.items():
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

    histContainer = HistogramContainer(
        bin_type=MeanSplitterHistogram, 
        max_bins=1
    )

    node_mean = NodeHoeffdingBound(
        split_confidence=0.01, 
        max_feature=1.0, 
        grace_period=10,
        hist=histContainer,
    )
    for i in range(20):
        node_mean.update_stats(X[i], y[i])
    print("Samples seen (mean splitter):", node_mean.samples_seen)
    print("Histograms (mean splitter):")
    for f, h in node_mean.hist.histograms.items():
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
