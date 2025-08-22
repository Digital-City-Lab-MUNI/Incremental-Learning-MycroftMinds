"""Hoeffding-bound node used by FastODT for streaming regression trees.

Maintains per-feature histograms to propose split thresholds, evaluates them
with a variance-reduction criterion, and applies a Hoeffding test to decide
when a split is statistically justified under streaming constraints.
"""

import numpy as np
from src.utils.utils import get_unique_random_numbers
from src.trees.histogram import AdaptiveHistogram
from src.trees.split_criteria import variance_reduction, variance_reduction_relative


def hoeffding_bound(R: float, n: int, delta: float) -> float:
    """Hoeffding bound for a bounded range statistic.

    Parameters
    ----------
    R : float
        Range of the split criterion (e.g. variance reduction).
    n : int
        Number of samples observed.
    delta : float
        Confidence parameter (e.g. 0.01 for 99% confidence, smaller delta => tighter bound requirement).

    Returns
    -------
    float
        Upper bound epsilon such that true mean lies within ``empirical +/- eps``
        with probability ``1 - delta``.
    """
    return np.sqrt((R**2) * np.log(1 / delta) / (2 * n))


class NodeHoeffdingBound:  # one leaf of the FastODT
    """Incremental node with histogram-based split selection.

    Maintains per-feature histograms for candidate thresholds, applies a
    variance-reduction criterion, and uses a Hoeffding bound to decide when a
    statistically reliable split can be performed.

    Parameters
    ----------
    split_confidence : float
        Confidence parameter ``delta`` for Hoeffding test.
    tie_threshold : float, default=0.05
        Tie-breaking threshold when best and second-best gains are close.
    max_feature : float, default=1.0
        Fraction of features randomly sampled for this node (``(0,1]``).
    eagerness : float, default=0.0
        Factor controlling speed of splitting (increases effective attempts).
    grace_period : int, default=0
        Minimum samples observed before considering a split.
    bin_type : type, default=AdaptiveHistogram
        Histogram class (AdaptiveHistogram, EquiWidthHistogram, MeanSplitter...).
    max_bins : int, default=32
        Maximum bins per feature histogram.
    inherit_bins : bool, default=False
        If True children receive partitioned / copied parent histograms.
    split_criterion : callable, optional
        Function computing split quality; defaults to absolute variance
        reduction.

    Attributes
    ----------
    tie_threshold : float
        Tie-breaking threshold when best and second-best gains are close.
    delta : float
        Confidence parameter for Hoeffding bound.
    samples_seen : int
        Number of samples seen by this node.
    y_max : float
        Maximum target value seen by this node.
    y_min : float
        Minimum target value seen by this node.
    sum_y : float
        Sum of target values seen by this node.
    sum_y_squared : float
        Sum of squared target values seen by this node.
    value : float
        Average target value for samples seen by this node.
    left : NodeHoeffdingBound
        Left child node.
    right : NodeHoeffdingBound
        Right child node.
    feature : int
        Splitting feature index.
    threshold : float
        Splitting feature threshold.
    eagerness : float
        Eagerness factor for splitting.
    eager_splitting_counter : float
        Counter for eager splitting.
    split_period_counter : int
        Counter for split periods.
    split_ideal_period : int
        Ideal split period.
    grace_period : int
        Grace period for splitting.
    grace_period_offset : int
        Offset for grace period.
    prev_splitting_period : int
        Previous splitting period.
    max_feature : float
        Maximum feature fraction.
    feature_indices : np.ndarray
        Indices of features to consider for splitting.
    bin_type : callable
        Histogram binning strategy.
    max_bins : int
        Maximum number of bins for histograms.
    histograms : dict
        Histograms for each feature.
    split_criterion : callable
        Function to evaluate split quality.
    inherit_bins : bool
        Whether to inherit bins from parent.

    """

    # TODO:
    # reduction of memory footprint

    def __init__(
        self,
        split_confidence: float,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        eagerness: float = 0.0,
        grace_period: int = 0,
        bin_type: type = AdaptiveHistogram,
        max_bins: int = 32,
        inherit_bins: bool = False,
        split_criterion: callable = None,
    ) -> None:
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

        self.eagerness: float = eagerness
        # 200 is magic number but it is hyperparam which makes hf work for us
        self.eager_splitting_counter: float = 200.0
        self.split_period_counter: int = 0
        self.split_ideal_period: int = 10
        self.grace_period: int = grace_period
        # Offset from the parent's `samples_seen` that's passed to child
        # for proper grace period calculation
        self.grace_period_offset: int = 0
        self.prev_splitting_period: int = self.split_ideal_period

        self.max_feature: float = max_feature
        self.feature_indices: np.ndarray = None

        self.bin_type: callable = bin_type  # e.g. 'adaptive', 'equiwidth', 'mean'
        self.max_bins: int = max_bins
        self.histograms: dict = None  # Will be initialized after seeing first sample

        self.split_criterion: callable = split_criterion if split_criterion is not None else variance_reduction
        self.inherit_bins: bool = inherit_bins

    def check_grace_period(self) -> bool:
        """Check if the grace period has expired."""
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def update_stats(self, x: np.ndarray, y: float) -> None:
        """
        Update histograms and aggregate stats with a new sample.

        Parameters
        ----------
        x : np.ndarray
            Feature vector for the new sample.
        y : float
            Target value for the new sample.
        """
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

        # Initialize histograms if needed
        if self.histograms is None:
            self.histograms = {}
            for feature in self.feature_indices:
                self.histograms[feature] = self.bin_type(max_bins=self.max_bins)

        # Update histograms for each feature
        for feature, val in zip(self.feature_indices, x_shortlisted, strict=True):
            self.histograms[feature].insert(val, y)

    def update_from_parent_node(self, parent: "NodeHoeffdingBound") -> None:
        """Initialize from a parent node (only value retained)."""
        # self.samples_seen = parent.samples_seen
        # self.sum_y = parent.sum_y
        # self.sum_y_squared = parent.sum_y_squared
        self.value = parent.value  # but not this

        # if self.grace_period > 0:
        #     self.grace_period_offset = parent.samples_seen  # remove this too

    def _hoeffding_R(self) -> float:
        """
        Calculate the Hoeffding bound ``R`` value based on the variance reduction used.

        Notes
        -----
        * For relative reduction, ``R`` is fixed to 1.0 (as the range is normalized).
        * For absolute reduction, ``R`` is calculated as ``0.25 * (y_max - y_min) ** 2``,
          which is the Popoviciu upper bound.
        """
        # fixed range
        if self.split_criterion == variance_reduction_relative:
            return 1.0

        # absolute range
        elif self.split_criterion == variance_reduction:
            return 0.25 * (self.y_max - self.y_min) ** 2  # Popoviciu upper bound

    def find_best_split(self) -> tuple:
        """Return best split if Hoeffding Bound criterion is met, else ``(None, None)``."""
        self.eager_splitting_counter += self.prev_splitting_period / self.split_ideal_period * self.eagerness
        self.split_period_counter += 1

        best_split_1st = (None, None, 0)
        best_split_2nd = (None, None, 0)

        for feature, hist in (self.histograms or {}).items():
            candidates = hist.get_candidates()
            for thr in candidates:
                left_stats = hist.aggregate_stats(thr, left=True)
                right_stats = hist.aggregate_stats(thr, left=False)
                reduction = self.split_criterion(
                    left_stats,
                    right_stats,
                    self.samples_seen,
                    self.sum_y,
                    self.sum_y_squared,
                )
                if reduction > best_split_1st[2]:
                    best_split_2nd = best_split_1st
                    best_split_1st = (feature, thr, reduction)
                elif reduction > best_split_2nd[2] and reduction < best_split_1st[2]:
                    best_split_2nd = (feature, thr, reduction)

        # Apply Hoeffding bound after considering all features
        R = self._hoeffding_R()  # range based on absolute or relative reduction
        e = hoeffding_bound(R, self.samples_seen, self.delta)  # / self.eager_splitting_counter

        if best_split_1st[2] - best_split_2nd[2] > e or self.tie_threshold > e:
            self.eager_splitting_counter = 200.0
            self.prev_splitting_period = self.split_period_counter
            self.split_period_counter = 0.0
            return (best_split_1st[0], best_split_1st[1]), best_split_1st[2]

        return (None, None), 0

    def split(self, split_info: tuple, split_confidence: float) -> None:
        """
        Instantiate left/right children according to split_info.

        Parameters
        ----------
        split_info : tuple
            Contains the feature index and threshold for the split.
        split_confidence : float
            Confidence level for the split.
        """
        self.feature, self.threshold = split_info
        self.left = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            eagerness=self.eagerness,
            grace_period=self.grace_period,
            bin_type=self.bin_type,
            max_bins=self.max_bins,
            inherit_bins=self.inherit_bins,
            split_criterion=self.split_criterion,
        )
        self.right = NodeHoeffdingBound(
            split_confidence=split_confidence,
            tie_threshold=self.tie_threshold,
            max_feature=self.max_feature,
            eagerness=self.eagerness,
            grace_period=self.grace_period,
            bin_type=self.bin_type,
            max_bins=self.max_bins,
            inherit_bins=self.inherit_bins,
            split_criterion=self.split_criterion,
        )
        if self.inherit_bins and self.histograms is not None:
            self.left.histograms = {}
            self.right.histograms = {}
            for feature, hist in self.histograms.items():
                if feature == self.feature:
                    self.left.histograms[feature] = hist.partition_bins(self.threshold, left=True)
                    self.right.histograms[feature] = hist.partition_bins(self.threshold, left=False)
                else:
                    self.left.histograms[feature] = hist.copy_bins()
                    self.right.histograms[feature] = hist.copy_bins()
        # If not inheriting, children will initialize new empty histograms on first update_stats

    def reached_grace_period(self) -> bool:
        """Return True if grace period satisfied for attempting a split."""
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def show(self, prefix: str = "") -> None:
        """Show the details of the node."""
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")


if __name__ == "__main__":
    import numpy as np
    from src.trees.histogram import AdaptiveHistogram  # , EquiWidthHistogram

    print("Testing NodeHoeffdingBound with AdaptiveHistogram...")
    np.random.seed(42)
    X = np.random.uniform(0, 10, (100, 2))
    y = X[:, 0] * 2 + np.random.normal(0, 1, 100)

    node = NodeHoeffdingBound(
        split_confidence=0.01,
        max_feature=1.0,
        grace_period=10,
        bin_type=AdaptiveHistogram,
        max_bins=4,
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
        split_confidence=0.01,
        max_feature=1.0,
        grace_period=10,
        bin_type=MeanSplitterHistogram,
        max_bins=1,
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
