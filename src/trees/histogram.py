import bisect

class AdaptiveHistogram:
    
    """
    Maintains an adaptive, memory-bounded histogram for streaming data.

    Step-by-Step:
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
       - Identify adjacent edges with the smallest counts i = argmin over j of (bin_stats[j]['count'] + bin_stats[j+1]['count']).
       - Merge bin j+1 into bin j by summing counts and sums.
       - Remove bin_edges[i] and bin_stats[i+1].

    4. Generating split candidates:
       - Compute midpoints between consecutive entries in bin_edges:
         mids[k] = (bin_edges[k] + bin_edges[k+1]) / 2.0
       - Return list of these mids as potential thresholds.
    """

    def __init__(self, max_bins=32):
        self.max_bins   = max_bins
        self.bin_edges  = []     # sorted list of 'representative' values
        self.bin_stats  = []     # list of dicts: {'count','sum_y','sum_y2'}

    def copy_bins(self):
        """
        Returns a deep copy of the histogram (all bins and stats).
        """
        new_hist = AdaptiveHistogram(self.max_bins)
        new_hist.bin_edges = list(self.bin_edges)
        new_hist.bin_stats = [dict(s) for s in self.bin_stats]
        return new_hist

    def partition_bins(self, split_thr, left=True):
        """
        Returns a new AdaptiveHistogram containing only bins to the left (<=) or right (>) of split_thr.
        """
        new_hist = AdaptiveHistogram(self.max_bins)
        for edge, stat in zip(self.bin_edges, self.bin_stats):
            if (left and edge <= split_thr) or (not left and edge > split_thr):
                new_hist.bin_edges.append(edge)
                new_hist.bin_stats.append(dict(stat))
        return new_hist
    


    def insert(self, x: float, y: float) -> None:
        # if empty: create first bin
        if not self.bin_edges:
            self.bin_edges.append(x)
            self.bin_stats.append({'count':1,'sum_y':y,'sum_y2':y*y})
            return
        
        # find insertion index
        idx = bisect.bisect_left(self.bin_edges, x)

        # new extreme on the left → create new left‐most bin
        if idx == 0 and x < self.bin_edges[0]:
            self.bin_edges.insert(0, x)
            self.bin_stats.insert(0, {'count':1,'sum_y':y,'sum_y2':y*y})

        # new extreme on the right → create new right‐most bin
        elif idx == len(self.bin_edges) and x > self.bin_edges[-1]:
            self.bin_edges.append(x)
            self.bin_stats.append({'count':1,'sum_y':y,'sum_y2':y*y})
        
        else:
            # falls into existing bin at idx (or between idx-1 and idx)
            # we choose to aggregate into bin idx
            st = self.bin_stats[idx]
            st['count']   += 1
            st['sum_y']   += y
            st['sum_y2'] += y*y

        # merge if we exceeded max_bins
        if len(self.bin_stats) > self.max_bins:
            self._merge_bins()

    def _merge_bins(self):
        # find the smallest‐count adjacent pair
        i = min(
            range(len(self.bin_stats)-1),
            key=lambda j: self.bin_stats[j]['count'] + self.bin_stats[j+1]['count']
        )
        # merge bin i+1 into bin i
        a, b = self.bin_stats[i], self.bin_stats[i+1]
        a['count'] += b['count']
        a['sum_y'] += b['sum_y']
        a['sum_y2'] += b['sum_y2']
        del self.bin_stats[i+1]
        del self.bin_edges[i]

    def get_candidates(self):
        """
        Using raw bin edges as split thresholds means you'll split exactly at one of the observed values (or cluster representatives), which can lead to:

        • Imbalanced splits
        - If a bin edge corresponds to an extreme outlier, you might send almost all data to one side.
        • Boundary ambiguity
        - Points equal to the edge need a tie-breaking rule (<= vs. >).

        By contrast, using the midpoint between two adjacent edges:

        • Sits in the 'gap' between clusters
        - No data value lies exactly at that split, so left/right assignments are unambiguous.
        • Better approximates the true best split
        - If the ideal split falls anywhere between two observed values, the midpoint is a natural proxy.
        • Encourages more balanced partitions
        - You're dividing bins, not cutting right through one, so you're distributing whole clusters more evenly.

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
    
    def aggregate_stats(self, threshold, left=True):
        count = sum_y = sum_y2 = 0
        for edge, stat in zip(self.bin_edges, self.bin_stats):
            if (left and edge <= threshold) or (not left and edge > threshold):
                count += stat['count']
                sum_y += stat['sum_y']
                sum_y2 += stat['sum_y2']
        return {'count': count, 'sum_y': sum_y, 'sum_y2': sum_y2}

class EquiWidthHistogram:
    def copy_bins(self):
        """
        Returns a deep copy of the histogram (all bins and stats).
        """
        new_hist = EquiWidthHistogram(self.max_bins, self.min_val, self.max_val)
        new_hist.bin_edges = list(self.bin_edges)
        new_hist.bin_stats = [dict(s) for s in self.bin_stats]
        return new_hist

    def partition_bins(self, split_thr, left=True):
        """
        Returns a new EquiWidthHistogram containing only bins to the left (<=) or right (>) of split_thr.
        """
        new_hist = EquiWidthHistogram(self.max_bins, self.min_val, self.max_val)
        for edge, stat in zip(self.bin_edges, self.bin_stats):
            # For equi-width, bin_edges are upper boundaries
            if (left and edge <= split_thr) or (not left and edge > split_thr):
                new_hist.bin_edges.append(edge)
                new_hist.bin_stats.append(dict(stat))
        return new_hist
    """
    Maintains a fixed-width (equi-distance) histogram over a predefined range.

    Step-by-Step:
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
    
    def __init__(self, max_bins=32, min_val=None, max_val=None):
        self.max_bins = max_bins
        self.min_val  = min_val
        self.max_val  = max_val
        self.bin_edges = []   # upper boundary of each bin
        self.bin_stats = []   # same format: list of {'count','sum_y','sum_y2'}
        if min_val is not None and max_val is not None:
            self._init_bins()

    def _init_bins(self):
        width = (self.max_val - self.min_val) / self.max_bins
        # build bin boundaries and zeroed stats
        self.bin_edges = [self.min_val + width * (i+1) for i in range(self.max_bins)]
        self.bin_stats = [
            {'count':0, 'sum_y':0.0, 'sum_y2':0.0}
            for _ in range(self.max_bins)
        ]

    def set_range(self, min_val, max_val):
        """
        Define or redefine the full feature range.
        Drops all previous stats.
        """
        self.min_val = min_val
        self.max_val = max_val
        self._init_bins()

    def insert(self, x, y):
        if self.min_val is None or self.max_val is None:
            raise ValueError("EquiWidthHistogram: call set_range(min, max) before insert()")
        # map x into a bin index
        span = self.max_val - self.min_val
        if span == 0.0:
            idx = 0
        else:
            ratio = (x - self.min_val) / span
            idx = int(ratio * self.max_bins)
            idx = max(0, min(self.max_bins-1, idx))
        st = self.bin_stats[idx]
        st['count']   += 1
        st['sum_y']   += y
        st['sum_y2'] += y*y

    def get_candidates(self):
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
    """
    Histogram-like interface that always proposes the running mean as the only split candidate.
    Used for average-based splitting (legacy behavior).
    """
    def __init__(self, max_bins=1):
        self.count = 0
        self.sum_x = 0.0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        # Legacy-style left/right stats
        self.left_count = 0
        self.left_sum_y = 0.0
        self.left_sum_y2 = 0.0
        self.right_count = 0
        self.right_sum_y = 0.0
        self.right_sum_y2 = 0.0

    def insert(self, x, y):
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

    def get_candidates(self):
        if self.count == 0:
            return []
        return [self.sum_x / self.count]

    @property
    def bin_edges(self):
        # Return the mean as the split point
        if self.count == 0:
            return []
        return [self.sum_x / self.count]

    @property
    def bin_stats(self):
        # Return two bins: left (<= mean), right (> mean)
        return [
            {
                'count': self.left_count,
                'sum_y': self.left_sum_y,
                'sum_y2': self.left_sum_y2
            },
            {
                'count': self.right_count,
                'sum_y': self.right_sum_y,
                'sum_y2': self.right_sum_y2
            }
        ]

    def copy_bins(self):
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

    def partition_bins(self, split_thr, left=True):
        # Partition left/right stats based on mean vs split_thr
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

    def aggregate_stats(self, threshold, left=True):
        if left:
            return {'count': self.left_count, 'sum_y': self.left_sum_y, 'sum_y2': self.left_sum_y2}
        return {'count': self.right_count, 'sum_y': self.right_sum_y, 'sum_y2': self.right_sum_y2}

def get_histogram_type(name):
    """
    Return a histogram class by name.
    Supported names: 'adaptive' (default), 'equiwidth', 'mean', 'average'.
    """
    if name is None or name == 'adaptive':
        return AdaptiveHistogram
    elif name == 'equiwidth':
        return EquiWidthHistogram
    elif name in ('mean', 'average'):
        return MeanSplitterHistogram
    else:
        raise ValueError(f"Unknown histogram type: {name}")