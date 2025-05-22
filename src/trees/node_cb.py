import numpy as np
from src.utils.utils import get_unique_random_numbers

def hoeffding_bound(R,n, delta):
    """Computes Hoeffding bound given number of samples and confidence level."""
    """n - samples """
    """delta - confidence factor """
    return np.sqrt((R**2)*np.log(1 / delta) / (2 * n))

import numpy as np
from collections import defaultdict

class NodeConfidenceBound:

    __SUM_X = 0 
    __COUNT = 1 
    __VAL = 2 
    __LEFT_Y = 3 
    __LEFT_Y_SQUARED = 4 
    __LEFT_N = 5 
    __RIGHT_Y = 6 
    __RIGHT_Y_SQUARED = 7 
    __RIGHT_N = 8

    def __init__(self, max_feature=1.0, grace_period=0.0):
        self.samples_seen = 0
        self.sum_y = 0
        self.sum_y_squared = 0
        self.value = 0
        self.left = None  # None means it's a leaf node
        self.right = None  # None means it's a leaf node
        self.feature = None
        self.threshold = None

        self.max_feature = max_feature
        self.grace_period = grace_period
        self.feature_indecies = None

        self.attribute_stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0])  # Attribute sum and count
        self.data_from_before_split = []

    def update_stats(self, x, y):

        if self.feature_indecies is None:
            max_feature_length = int(x.shape[0] * self.max_feature)
            self.feature_indecies = np.sort(get_unique_random_numbers(max_feature_length,x.shape[0]))

        x_shortlisted = x[self.feature_indecies]

        self.samples_seen += 1
        self.sum_y += y
        self.sum_y_squared += y**2
        self.value = self.sum_y/self.samples_seen
        for feature, val in zip(self.feature_indecies,x_shortlisted):
            self.attribute_stats[feature][self.__SUM_X] += val
            self.attribute_stats[feature][self.__COUNT] += 1
            self.attribute_stats[feature][self.__VAL] = val
            sum_x = self.attribute_stats[feature][self.__SUM_X]
            count = self.attribute_stats[feature][self.__COUNT]
            avg_x = sum_x / count
            if val <= avg_x:
                self.attribute_stats[feature][self.__LEFT_Y] += y
                self.attribute_stats[feature][self.__LEFT_Y_SQUARED] += y**2
                self.attribute_stats[feature][self.__LEFT_N] += 1
            else:
                self.attribute_stats[feature][self.__RIGHT_Y] += y
                self.attribute_stats[feature][self.__RIGHT_Y_SQUARED] += y**2
                self.attribute_stats[feature][self.__RIGHT_N] += 1

    def update_from_parent_node(self, parent):
        self.samples_seen = parent.samples_seen
        self.sum_y = parent.sum_y
        self.sum_y_squared = parent.sum_y_squared
        self.value = parent.sum_y/self.samples_seen
        if self.grace_period > 0: 
            self.grace_period += parent.samples_seen

    def find_best_split(self):
        best_feature, best_value, best_reduction = (None, None, 0)
        for feature, stats in self.attribute_stats.items():
            (
                sum_x,
                count, 
                val, 
                left_y, 
                left_y_squared, 
                left_n, 
                right_y, 
                right_y_squared, 
                right_n
            ) = stats
            
            if count < 2:
                continue
            avg_x = sum_x / count
            reduction = self._variance_reduction(
                left_y, left_y_squared, left_n, 
                right_y, right_y_squared, right_n
            )

            if reduction > best_reduction:
                best_feature, best_value, best_reduction = feature, avg_x, reduction
        return (best_feature, best_value), best_reduction
    
    def _variance_reduction(self, left_y, left_y_squared, left_n, right_y, right_y_squared, right_n):
        if left_n == 0 or right_n == 0:
            return 0
        # Parent variance
        var_total = (self.sum_y_squared / self.samples_seen) - (self.sum_y / self.samples_seen) ** 2
        # Left and right variances
        var_left = (left_y_squared / left_n) - (left_y / left_n) ** 2
        var_right = (right_y_squared / right_n) - (right_y / right_n) ** 2
        # Weighted variance after split
        var_children = (left_n / self.samples_seen) * var_left + (right_n / self.samples_seen) * var_right

        return var_total - var_children  # variance reduction

    def split(self, split_info):
        self.feature, self.threshold = split_info
        self.left = NodeConfidenceBound(
            max_feature=self.max_feature,
            grace_period=self.grace_period,
        )
        self.right = NodeConfidenceBound(
            max_feature=self.max_feature,
            grace_period=self.grace_period,
        )

    def reached_grace_period(self):
        return self.grace_period < self.samples_seen

    def show(self,prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")

