import math
import numpy as np
from collections import defaultdict
from src.utils.utils import get_unique_random_numbers

def hoeffding_bound(R,n, delta):
    """Computes Hoeffding bound given number of samples and confidence level."""
    """n - samples """
    """delta - confidence factor """
    return np.sqrt((R**2)*np.log(1 / delta) / (2 * n))
    # harmonic_mean_n = 0
    # if n_left == 0:
    #     harmonic_mean_n = n_right
    # if n_right == 0:
    #     harmonic_mean_n = n_left
    # if n_left > 0 and n_right > 0:
    #     harmonic_mean_n = 1/((1/n_left)+(1/n_right))
    # return math.sqrt(1/(2*harmonic_mean_n) / delta)

class NodeHoeffdingBound:

    __SUM_X = 0 
    __COUNT = 1 
    __VAL = 2 
    __LEFT_Y = 3 
    __LEFT_Y_SQUARED = 4 
    __LEFT_N = 5 
    __RIGHT_Y = 6 
    __RIGHT_Y_SQUARED = 7 
    __RIGHT_N = 8
    
    def __init__(self,split_confidence, tie_threshold = 0.05, max_feature=1.0, eagerness=0.0, grace_period = 0):
        self.tie_threshold = tie_threshold
        self.delta = split_confidence
        self.samples_seen = 0
        self.sum_y = 0
        self.sum_y_squared = 0
        self.value = 0
        self.left = None  # None means it's a leaf node
        self.right = None  # None means it's a leaf node
        self.feature = None
        self.threshold = None

        self.eagerness = eagerness
        # 200 is magic number but it is hyperparam which makes hf work for us
        self.eager_splitting_counter = 200.0
        self.split_period_counter = 0
        self.split_ideal_period = 10
        self.grace_period = grace_period
        self.grace_period_offset = 0
        self.prev_splitting_period = self.split_ideal_period

        self.max_feature = max_feature
        self.feature_indecies = None

        self.attribute_stats = defaultdict(lambda: [0, 0, 0, 0, 0, 0, 0, 0, 0])  # Attribute sum and count
        self.data_from_before_split = []

    def check_grace_period(self):
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def update_stats(self, x, y):
        # store raw target for MAE computation
        self.data_from_before_split.append(y) # this was suggested by GPT,
        # there is no line so far updating the self.data_from_before_split list
        # perhaps instead val from self.attribute_stats should be used?

        if self.feature_indecies is None:
            max_feature_length = int(x.shape[0] * self.max_feature)
            self.feature_indecies = np.sort(get_unique_random_numbers(max_feature_length,x.shape[0]))

        # x_shortlisted = x
        x_shortlisted = x[self.feature_indecies]

        self.samples_seen += 1
        self.sum_y += y
        self.sum_y_squared += y**2
        self.value = self.sum_y/self.samples_seen

        for feature, val in zip(self.feature_indecies,x_shortlisted):
            self.attribute_stats[feature][self.__SUM_X] += val
            self.attribute_stats[feature][self.__COUNT] += 1
            self.attribute_stats[feature][self.__VAL] = val
            sum_x = self.attribute_stats[feature][self.__SUM_X] # sum_x is the sum of all x values for this feature
            count = self.attribute_stats[feature][self.__COUNT]
            avg_x = sum_x / count # median, is this justified? probably it is, but possibly we could split 
            # in the first 30% which would create more homogeneous groups after splits than mean
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
            self.grace_period_offset = parent.samples_seen


    def find_best_split(self):
        self.eager_splitting_counter += self.prev_splitting_period/self.split_ideal_period * self.eagerness
        self.split_period_counter += 1

        __best_split_1st = (None, None, 0)
        __best_split_2nd = (None, None, 0)
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

            if reduction > __best_split_1st[2]:
                __best_split_2nd = __best_split_1st
                __best_split_1st = (feature, avg_x, reduction)
            elif reduction > __best_split_2nd[2] and reduction < __best_split_1st[2]:
                __best_split_2nd = (feature, avg_x, reduction)

            R = len(self.attribute_stats)
            e = hoeffding_bound(R,count,self.delta)/self.eager_splitting_counter
            if R > 1:
                if __best_split_1st[2] - __best_split_2nd[2] > e:
                    self.eager_splitting_counter = 200.0  
                    self.prev_splitting_period = self.split_period_counter
                    self.split_period_counter = 0.0
                    return (__best_split_1st[0],__best_split_1st[1]), __best_split_1st[2]
            elif R == 1:
                if __best_split_1st[2] > e:
                    self.eager_splitting_counter = 200.0 
                    self.prev_splitting_period = self.split_period_counter
                    self.split_period_counter = 0.0
                    return (__best_split_1st[0],__best_split_1st[1]), __best_split_1st[2]

        return (None, None), 0

    def _variance_reduction(self, left_y, left_y_squared, left_n, right_y, right_y_squared, right_n):
        # Also known as MSE gain
        if left_n == 0 or right_n == 0:
            return 0
        # Parent variance
        var_parent = (self.sum_y_squared / self.samples_seen) - (self.sum_y / self.samples_seen) ** 2
        # Left and right variances
        var_left = (left_y_squared / left_n) - (left_y / left_n) ** 2
        var_right = (right_y_squared / right_n) - (right_y / right_n) ** 2
        # Weighted variance after split
        var_children = (left_n / self.samples_seen) * var_left + (right_n / self.samples_seen) * var_right

        return var_parent - var_children  # variance reduction
    
    def _friedman_mse(self, left_y, left_n,
                      right_y, right_n):
        
        if left_n == 0 or right_n == 0:
            return 0
        
        mean_left = left_y / left_n
        mean_right = right_y / right_n
        diff_squared = (mean_left - mean_right) **2

        return ((left_n * right_n) / (left_n + right_n)) * diff_squared
    
    def _poisson_deviance(self,
                          left_y, left_y_log, left_n,
                          right_y, right_y_log, right_n):
        # Poisson deviance reduction
        # Requires: all y > 0, and self.sum_y_log = sum(y * log(y)) over the parent
        if left_n == 0 or right_n == 0:
            return 0.0

        # parent mean
        mu = self.sum_y / self.samples_seen
        # parent deviance
        D_parent = 2 * (self.sum_y_log - self.sum_y * np.log(mu))

        # left deviance
        mu_L = left_y / left_n
        D_left = 2 * (left_y_log - left_y * np.log(mu_L))

        # right deviance
        mu_R = right_y / right_n
        D_right = 2 * (right_y_log - right_y * np.log(mu_R))

        # weighted children deviance
        D_children = (left_n  / self.samples_seen) * D_left \
                   + (right_n / self.samples_seen) * D_right

        return D_parent - D_children


    def _mean_absolute_error(self, left_values, right_values):
        # MAE gain (reduction in absolute deviation around median)
        if len(left_values) == 0 or len(right_values) == 0:
            return 0.0

        # parent median & total absolute deviation
        parent = np.array(self.data_from_before_split) # is this correct variable to use? perhaps val would be better
        med_p = np.median(parent)
        mae_p = np.sum(np.abs(parent - med_p))

        # left median & deviation
        L = np.array(left_values)
        med_L = np.median(L)
        mae_L = np.sum(np.abs(L - med_L))

        # right median & deviation
        R = np.array(right_values)
        med_R = np.median(R)
        mae_R = np.sum(np.abs(R - med_R))

        # reduction in total absolute deviation
        return mae_p - (mae_L + mae_R)



    def split(self, split_info, split_confidence):
        self.feature, self.threshold = split_info
        self.left = NodeHoeffdingBound(
            split_confidence,
            max_feature=self.max_feature,
            eagerness=self.eagerness,
            grace_period=self.grace_period
        )
        self.right = NodeHoeffdingBound(
            split_confidence,
            max_feature=self.max_feature,
            eagerness=self.eagerness,
            grace_period=self.grace_period
        )

    def reached_grace_period(self):
        return self.grace_period_offset + self.grace_period < self.samples_seen

    def show(self,prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")
