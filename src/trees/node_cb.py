import numpy as np

def hoeffding_bound(R,n, delta):
    """Computes Hoeffding bound given number of samples and confidence level."""
    """n - samples """
    """delta - confidence factor """
    return np.sqrt((R**2)*np.log(1 / delta) / (2 * n))

import numpy as np
from collections import defaultdict

class NodeConfidenceBound:
    def __init__(self):
        self.samples_seen = 0
        self.sum_y = 0
        self.sum_y_squared = 0
        self.value = 0
        self.left = None  # None means it's a leaf node
        self.right = None  # None means it's a leaf node
        self.feature = None
        self.threshold = None
        self.attribute_stats = defaultdict(lambda: [0, 0, 0])  # Attribute sum and count
        self.data_from_before_split = []
        self.left_y = []
        self.right_y = []

    def update_stats(self, x, y):
        self.samples_seen += 1
        self.sum_y += y
        self.sum_y_squared += y**2
        self.value = self.sum_y/self.samples_seen
        for feature, val in enumerate(x):
            self.attribute_stats[feature][0] += val
            self.attribute_stats[feature][1] += 1
            self.attribute_stats[feature][2] = val

    def update_from_parent_node(self, parent):
        self.samples_seen = parent.samples_seen
        self.sum_y = parent.sum_y
        self.sum_y_squared = parent.sum_y_squared
        self.value = parent.sum_y/self.samples_seen

    def find_best_split(self):
        best_feature, best_value, best_reduction = None, None, 0
        for feature, (sum_x, count, val) in self.attribute_stats.items():
            if count < 2:
                continue
            avg_x = sum_x / count
            reduction = self._variance_reduction(avg_x, val)
            if reduction > best_reduction:
                best_feature, best_value, best_reduction = feature, avg_x, reduction
        return (best_feature, best_value), best_reduction
    
    def _variance_reduction(self, split_value, threshold):
        for _ in range(self.samples_seen):
            if threshold <= split_value:
                self.left_y.append(self.sum_y / self.samples_seen)
            else:
                self.right_y.append(self.sum_y / self.samples_seen)

        if not self.left_y and not self.right_y:
            return 0
        var_before = (self.sum_y_squared / self.samples_seen) - (self.sum_y / self.samples_seen) ** 2
        var_left = np.var(self.left_y) if self.left_y else 0
        var_right = np.var(self.right_y) if self.right_y else 0
        return var_before - ((len(self.left_y) / self.samples_seen) * var_left + (len(self.right_y) / self.samples_seen) * var_right)
    
    def split(self, split_info):
        self.feature, self.threshold = split_info
        self.left = NodeConfidenceBound()
        self.right = NodeConfidenceBound()

    def show(self,prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")

