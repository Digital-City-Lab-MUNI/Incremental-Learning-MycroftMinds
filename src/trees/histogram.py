"""Histogram data structures for streaming feature split discovery.

This module provides lightweight, streaming-friendly histogram abstractions used
by incremental tree learners to propose candidate split thresholds without
storing the full set of observed feature values.

Implemented variants
--------------------
AdaptiveHistogram
    Maintains a variable set of representative bin edges (cluster centers)
    subject to a maximum number of bins. When the limit is exceeded, the least
    populated adjacent pair of bins is merged. Produces split candidates at the
    midpoints between consecutive bin edges.
EquiWidthHistogram
    Uses a fixed number of equal-width bins over a preset value range defined
    after an initial warm‑up phase. Produces midpoints between bin boundaries.
MeanSplitterHistogram
    Not a true histogram: tracks only the running mean of feature values and
    exposes it as the single split candidate. Retains synthetic left/right
    aggregates for compatibility with downstream variance / impurity routines.

All histogram objects expose a shared minimal API relied upon by the tree code:

    insert(x, y) -> None
        Consume a (feature_value, target_value) sample.
    get_candidates() -> list[float]
        Return an ordered list of prospective split thresholds.
    partition_bins(split_thr, left=True)
        Return a new histogram instance containing only bins on one side of a
        threshold (used for optimistic / incremental split evaluation).
    aggregate_stats(threshold, left=True) -> dict
        Aggregate statistics (count, sum_y, sum_y2) for one side of a threshold.

Statistics stored per (synthetic) bin enable downstream calculation of simple
impurity measures such as variance or mean squared error without revisiting raw
samples.
"""

from __future__ import annotations

from typing import Type, Union, Dict, List
import bisect
import src.trees.constants as C


class AdaptiveHistogram:
    """
    Adaptive, memory-bounded histogram for streaming numerical data.

    Parameters
    ----------
    max_bins : int, default=32
        Maximum number of bins before triggering merges.

    Step-by-step description
    ----------
    1. Initialization:
       - max_bins: maximum number of bins to retain.
       - bin_edges: empty list; will hold one representative value per bin.
       - bin_stats: empty list; each entry is a dict with keys:
         {'count', 'sum_y', 'sum_y2'}.

    2. Inserting a data point (x, y):
       a. If no bins exist yet:
          • Append x to bin_edges.
          • Append {'count':1, 'sum_y':y, 'sum_y2':y*y} to bin_stats.
       b. Otherwise, find insertion index idx = bisect_left(bin_edges, x):
          • If x < bin_edges[0]: insert new leftmost bin with representative x.
          • Elif x > bin_edges[-1]: append new rightmost bin with representative x.
          • Else: update existing bin_stats[idx]:
               count += 1, sum_y += y, sum_y2 += y*y.
       c. If number of bins > max_bins:
          • Merge the two adjacent bins with the smallest combined count.

    3. Merging bins:
       - Identify adjacent edges with the smallest counts
            i = argmin over j of (bin_stats[j]['count'] + bin_stats[j+1]['count']).
       - Merge bin j+1 into bin j by summing counts and sums.
       - Remove bin_edges[i] and bin_stats[i+1].

    4. Generating split candidates:
       - Compute midpoints between consecutive entries in bin_edges:
         mids[k] = (bin_edges[k] + bin_edges[k+1]) / 2.0
       - Return list of these mids as potential thresholds.
    """

    def __init__(self, max_bins: int = 32) -> None:
        self.max_bins: int = max_bins
        self.bin_edges: List[float] = []  # sorted representative values
        self.bin_stats: List[Dict[str, float | int]] = []  # per-bin aggregates

    def copy_bins(self) -> "AdaptiveHistogram":
        """Return a deep copy of the histogram state."""
        new_hist = AdaptiveHistogram(self.max_bins)
        new_hist.bin_edges = list(self.bin_edges)
        new_hist.bin_stats = [dict(s) for s in self.bin_stats]
        return new_hist

    def partition_bins(self, split_thr: float, left: bool = True) -> "AdaptiveHistogram":
        """Return a new histogram keeping only bins on one side of a threshold.

        Parameters
        ----------
        split_thr : float
            Threshold value used to select bins.
        left : bool, default=True
            If True keep bins with edge ``<= split_thr``; otherwise keep bins
            with edge ``> split_thr``.

        Returns
        -------
        AdaptiveHistogram
            New histogram with a subset of bins.
        """
        new_hist = AdaptiveHistogram(self.max_bins)
        for edge, stat in zip(self.bin_edges, self.bin_stats, strict=True):
            if (left and edge <= split_thr) or (not left and edge > split_thr):
                new_hist.bin_edges.append(edge)
                new_hist.bin_stats.append(dict(stat))
        return new_hist

    def insert(self, x: float, y: float) -> None:
        """Insert a new sample into one of the bins.

        * If the value falls into an existing bin, update that bin's statistics.
        * If the value exceeds the current bin limits, create a new bin.
        * If the number of bins exceeds the maximum, merge the two least populated bins.

        Parameters
        ----------
        x : float
            Feature value.
        y : float
            Target (label) value whose aggregate statistics are stored.
        """
        # if empty: create first bin
        if not self.bin_edges:
            self.bin_edges.append(x)
            self.bin_stats.append({C.StatKey.COUNT: 1, C.StatKey.SUM_Y: y, C.StatKey.SUM_Y2: y * y})
            return

        # find insertion index
        idx = bisect.bisect_left(self.bin_edges, x)

        # new extreme on the left → create new left‐most bin
        if idx == 0 and x < self.bin_edges[0]:
            self.bin_edges.insert(0, x)
            self.bin_stats.insert(0, {C.StatKey.COUNT: 1, C.StatKey.SUM_Y: y, C.StatKey.SUM_Y2: y * y})

        # new extreme on the right → create new right‐most bin
        elif idx == len(self.bin_edges) and x > self.bin_edges[-1]:
            self.bin_edges.append(x)
            self.bin_stats.append({C.StatKey.COUNT: 1, C.StatKey.SUM_Y: y, C.StatKey.SUM_Y2: y * y})

        else:
            # falls into existing bin at idx (or between idx-1 and idx)
            # we choose to aggregate into bin idx
            st = self.bin_stats[idx]
            st[C.StatKey.COUNT] += 1
            st[C.StatKey.SUM_Y] += y
            st[C.StatKey.SUM_Y2] += y * y

        # merge if we exceeded max_bins
        if len(self.bin_stats) > self.max_bins:
            self._merge_bins()

    def _merge_bins(self) -> None:
        """Merge the least-populated adjacent pair of bins."""
        # find the smallest-count adjacent pair
        i = min(
            range(len(self.bin_stats) - 1),
            key=lambda j: self.bin_stats[j][C.StatKey.COUNT] + self.bin_stats[j + 1][C.StatKey.COUNT],
        )
        # merge bin i+1 into bin i
        a, b = self.bin_stats[i], self.bin_stats[i + 1]
        a[C.StatKey.COUNT] += b[C.StatKey.COUNT]
        a[C.StatKey.SUM_Y] += b[C.StatKey.SUM_Y]
        a[C.StatKey.SUM_Y2] += b[C.StatKey.SUM_Y2]
        del self.bin_stats[i + 1]
        del self.bin_edges[i]

    def get_candidates(self) -> List[float]:
        """Return midpoint split candidates between consecutive bin edges.

        Returns
        -------
        list[float]
            Midpoints between successive bin edges. Empty list if fewer than
            two bins exist.

        Notes
        -----
        Using raw bin edges as split thresholds means you'll split exactly at
        one of the observed values (or cluster representatives), which can lead to:
        - Imbalanced splits
            If a bin edge corresponds to an extreme outlier, you might send almost all data to one side.
        - Boundary ambiguity
            Points equal to the edge need a tie-breaking rule (<= vs. >).

        By contrast, using the midpoint between two adjacent edges:
        - Sits in the 'gap' between clusters
            No data value lies exactly at that split, so left/right assignments are unambiguous.
        - Better approximates the true best split
            If the ideal split falls anywhere between two observed values, the midpoint is a natural proxy.
        - Encourages more balanced partitions
            You're dividing bins, not cutting right through one, so you're distributing whole clusters more evenly.

        In short, midpoints give you a cleaner, more robust split point when you only have coarse bin boundaries.
        """
        # Calculate midpoints between consecutive bin edges
        mids = []
        if len(self.bin_edges) < 2:
            return mids
        prev = self.bin_edges[0]
        for edge in self.bin_edges[1:]:
            mids.append((prev + edge) / 2.0)
            prev = edge
        return mids

    def aggregate_stats(self, threshold: float, left: bool = True) -> Dict[str, float | int]:
        """Aggregate statistics on one side of a threshold."""
        count = 0
        sum_y = 0.0
        sum_y2 = 0.0
        for edge, stat in zip(self.bin_edges, self.bin_stats, strict=True):
            if (left and edge <= threshold) or (not left and edge > threshold):
                count += int(stat[C.StatKey.COUNT])
                sum_y += float(stat[C.StatKey.SUM_Y])
                sum_y2 += float(stat[C.StatKey.SUM_Y2])
        return {C.StatKey.COUNT: count, C.StatKey.SUM_Y: sum_y, C.StatKey.SUM_Y2: sum_y2}


class EquiWidthHistogram:
    """Fixed-width histogram over a predefined numeric range.

    Parameters
    ----------
    max_bins : int, default=32
        Number of equal-width bins.
    min_val : float, optional
        Lower bound of the feature range. If provided with ``max_val`` the
        histogram is initialized immediately.
    max_val : float, optional
        Upper bound of the feature range. If provided with ``min_val`` the
        histogram is initialized immediately.

    Step-by-Step
    ------------
    1. Initialization:
       - max_bins: number of equal-width bins.
       - min_val, max_val: optional; if provided, bins are immediately initialized.
       - bin_edges: list of upper boundaries for each bin.
       - bin_stats: list of dicts {'count', 'sum_y', 'sum_y2'} initialized to zero.

    2. Defining the value range:
       • Call set_range(min_val, max_val) once you have a warm-up sample.
       • Compute width = (max_val - min_val) / max_bins.
       • Set bin_edges[i] = min_val + width*(i+1) for i in 0..max_bins-1.
       • Zero out bin_stats for all bins.

    3. Inserting a data point (x, y):
       a. Ensure min_val and max_val are set, else raise error.
       b. Compute normalized position: ratio = (x - min_val) / (max_val - min_val).
       c. Determine bin index idx = floor(ratio * max_bins), clamped to [0, max_bins-1].
       d. Update bin_stats[idx]: count += 1, sum_y += y, sum_y2 += y*y.

    4. Generating split candidates:
       - Let edges = [min_val] + bin_edges.
       - Compute midpoints between edges[k] and edges[k+1] for k in 0..max_bins-1.
       - Return list of these midpoints as thresholds.

    You must call set_range() (e.g. after a warm-up) before insert().
    """

    def __init__(self, max_bins: int = 32, min_val: float | None = None, max_val: float | None = None) -> None:
        self.max_bins: int = max_bins
        self.min_val: float | None = min_val
        self.max_val: float | None = max_val
        self.bin_edges: List[float] = []  # upper boundary of each bin
        self.bin_stats: List[Dict[str, float | int]] = []
        if min_val is not None and max_val is not None:
            self._init_bins()

    def _init_bins(self) -> None:
        """(Re)initialize bin edges and zero statistics based on current range."""
        width = (self.max_val - self.min_val) / self.max_bins
        # build bin boundaries and zeroed stats
        self.bin_edges = [self.min_val + width * (i + 1) for i in range(self.max_bins)]
        self.bin_stats = [
            {C.StatKey.COUNT: 0, C.StatKey.SUM_Y: 0.0, C.StatKey.SUM_Y2: 0.0} for _ in range(self.max_bins)
        ]

    def copy_bins(self) -> "EquiWidthHistogram":
        """Return a deep copy of the histogram state."""
        new_hist = EquiWidthHistogram(self.max_bins, self.min_val, self.max_val)
        new_hist.bin_edges = list(self.bin_edges)
        new_hist.bin_stats = [dict(s) for s in self.bin_stats]
        return new_hist

    def partition_bins(self, split_thr: float, left: bool = True) -> "EquiWidthHistogram":
        """Partition bins relative to a threshold.

        Parameters
        ----------
        split_thr : float
            Threshold value (compared against each bin's upper edge).
        left : bool, default=True
            If True keep bins with edge ``<= split_thr`` else keep strictly
            larger edges.

        Returns
        -------
        EquiWidthHistogram
            Histogram with selected bins only.
        """
        new_hist = EquiWidthHistogram(self.max_bins, self.min_val, self.max_val)
        for edge, stat in zip(self.bin_edges, self.bin_stats, strict=True):
            # For equi-width, bin_edges are upper boundaries
            if (left and edge <= split_thr) or (not left and edge > split_thr):
                new_hist.bin_edges.append(edge)
                new_hist.bin_stats.append(dict(stat))
        return new_hist

    def set_range(self, min_val: float, max_val: float) -> None:
        """Define or redefine the full feature range.

        Parameters
        ----------
        min_val : float
            New lower bound.
        max_val : float
            New upper bound.

        Notes
        -----
        Existing statistics are discarded. Call once after an initial warm-up
        sample if the range is not known a priori.
        """
        self.min_val = min_val
        self.max_val = max_val
        self._init_bins()

    def insert(self, x: float, y: float) -> None:
        """Insert a sample accumulating target statistics in the proper bin.

        Parameters
        ----------
        x : float
            Feature value. Values outside the configured range are clamped to
            the nearest boundary bin.
        y : float
            Target / label value.

        Raises
        ------
        ValueError
            If the range has not yet been set via :meth:`set_range`.
        """
        if self.min_val is None or self.max_val is None:
            raise ValueError("EquiWidthHistogram: call set_range(min, max) before insert()")
        # map x into a bin index
        span = self.max_val - self.min_val
        if span == 0.0:
            idx = 0
        else:
            ratio = (x - self.min_val) / span
            idx = int(ratio * self.max_bins)
            idx = max(0, min(self.max_bins - 1, idx))
        st = self.bin_stats[idx]
        st[C.StatKey.COUNT] += 1
        st[C.StatKey.SUM_Y] += y
        st[C.StatKey.SUM_Y2] += y * y

    def get_candidates(self) -> List[float]:
        """Return midpoint split candidates between successive boundaries.

        Returns
        -------
        list[float]
            Midpoints between successive boundaries including the implicit
            leftmost boundary ``min_val``. Empty list if uninitialized.
        """
        # midpoint between successive bin boundaries
        mids = []
        if not self.bin_edges:
            return mids
        prev = self.min_val
        for edge in self.bin_edges:
            mids.append((prev + edge) / 2.0)
            prev = edge
        return mids


class MeanSplitterHistogram:
    """Facade that proposes only the running mean as a split threshold.

    Unlike true histograms this structure maintains only global aggregates and
    synthesizes two pseudo-bins (left / right of the current mean) for API
    compatibility. Primarily useful for very small memory footprints.

    Parameters
    ----------
    max_bins : int, default=1
        Ignored (retained for interface uniformity).
    """

    def __init__(self, max_bins: int = 1) -> None:  # noqa: D401 (parameter kept for interface)
        self.count: int = 0
        self.sum_x: float = 0.0
        self.sum_y: float = 0.0
        self.sum_y2: float = 0.0
        self.left_count: int = 0
        self.left_sum_y: float = 0.0
        self.left_sum_y2: float = 0.0
        self.right_count: int = 0
        self.right_sum_y: float = 0.0
        self.right_sum_y2: float = 0.0

    def insert(self, x: float, y: float) -> None:
        """Insert a new sample updating left/right aggregates."""
        self.count += 1
        self.sum_x += x
        self.sum_y += y
        self.sum_y2 += y * y
        # Use running mean BEFORE this sample for partitioning
        if self.count == 1:
            # First sample, assign to left by convention
            self.left_count += 1
            self.left_sum_y += y
            self.left_sum_y2 += y * y
        else:
            mean = (self.sum_x - x) / (self.count - 1)
            if x <= mean:
                self.left_count += 1
                self.left_sum_y += y
                self.left_sum_y2 += y * y
            else:
                self.right_count += 1
                self.right_sum_y += y
                self.right_sum_y2 += y * y

    def get_candidates(self) -> List[float]:
        """Return a single-element list with the current running mean as the sole split candidate."""
        if self.count == 0:
            return []
        return [self.sum_x / self.count]

    @property
    def bin_edges(self) -> List[float]:
        """Representative 'edge' list containing only the running mean."""
        if self.count == 0:
            return []
        return [self.sum_x / self.count]

    @property
    def bin_stats(self) -> List[Dict[str, float | int]]:
        """Synthetic two-bin statistics (left ``<=`` mean, right ``>`` mean)."""
        return [
            {C.StatKey.COUNT: self.left_count, C.StatKey.SUM_Y: self.left_sum_y, C.StatKey.SUM_Y2: self.left_sum_y2},
            {C.StatKey.COUNT: self.right_count, C.StatKey.SUM_Y: self.right_sum_y, C.StatKey.SUM_Y2: self.right_sum_y2},
        ]

    def copy_bins(self) -> "MeanSplitterHistogram":
        """Return a deep copy of the current state."""
        new_hist = MeanSplitterHistogram()
        new_hist.count = self.count
        new_hist.sum_x = self.sum_x
        new_hist.sum_y = self.sum_y
        new_hist.sum_y2 = self.sum_y2
        new_hist.left_count = self.left_count
        new_hist.left_sum_y = self.left_sum_y
        new_hist.left_sum_y2 = self.left_sum_y2
        new_hist.right_count = self.right_count
        new_hist.right_sum_y = self.right_sum_y
        new_hist.right_sum_y2 = self.right_sum_y2
        return new_hist

    def partition_bins(self, split_thr: float, left: bool = True) -> "MeanSplitterHistogram":
        """Partition synthetic bins given an external threshold.

        Parameters
        ----------
        split_thr : float
            Threshold to compare against the current mean.
        left : bool, default=True
            If True retain the side (relative to ``split_thr``) interpreted as
            *left*; otherwise retain the *right* side.

        Returns
        -------
        MeanSplitterHistogram
            Histogram containing aggregates for the retained side(s).
        """
        new_hist = MeanSplitterHistogram()
        mean = self.sum_x / self.count if self.count > 0 else 0.0
        if left:
            if mean <= split_thr:
                new_hist.left_count = self.left_count
                new_hist.left_sum_y = self.left_sum_y
                new_hist.left_sum_y2 = self.left_sum_y2
            if mean > split_thr:
                new_hist.right_count = self.right_count
                new_hist.right_sum_y = self.right_sum_y
                new_hist.right_sum_y2 = self.right_sum_y2
        else:
            if mean > split_thr:
                new_hist.right_count = self.right_count
                new_hist.right_sum_y = self.right_sum_y
                new_hist.right_sum_y2 = self.right_sum_y2
            if mean <= split_thr:
                new_hist.left_count = self.left_count
                new_hist.left_sum_y = self.left_sum_y
                new_hist.left_sum_y2 = self.left_sum_y2
        return new_hist

    def aggregate_stats(self, threshold: float, left: bool = True) -> Dict[str, float | int]:  # noqa: D401
        """Return aggregate stats for one side of the synthetic split.

        Parameters
        ----------
        threshold : float
            External threshold (ignored for partitioning, included for API
            symmetry with real histograms).
        left : bool, default=True
            If True return left-side stats else right-side stats.

        Returns
        -------
        dict
            Keys: ``count`` (int), ``sum_y`` (float), ``sum_y2`` (float).
        """
        if left:
            return {
                C.StatKey.COUNT: self.left_count,
                C.StatKey.SUM_Y: self.left_sum_y,
                C.StatKey.SUM_Y2: self.left_sum_y2,
            }
        return {
            C.StatKey.COUNT: self.right_count,
            C.StatKey.SUM_Y: self.right_sum_y,
            C.StatKey.SUM_Y2: self.right_sum_y2,
        }


def get_histogram_type(name: str | None) -> Type[Union[AdaptiveHistogram, EquiWidthHistogram, MeanSplitterHistogram]]:
    """Return a histogram class given its name.

    Parameters
    ----------
    name : str or None
        One of ``{'adaptive', 'equiwidth', 'mean', 'average'}``. ``None`` maps
        to ``'adaptive'``. ``'mean'`` and ``'average'`` are aliases.

    Returns
    -------
    type
        Corresponding histogram class.

    Raises
    ------
    ValueError
        If ``name`` is unrecognized.
    """
    if name is None or name == C.HistogramType.ADAPTIVE:
        return AdaptiveHistogram
    if name == C.HistogramType.EQUIWIDTH:
        return EquiWidthHistogram
    if name in (C.HistogramType.MEAN, C.HistogramType.AVERAGE):
        return MeanSplitterHistogram
    raise ValueError(f"Unknown histogram type: {name}")
