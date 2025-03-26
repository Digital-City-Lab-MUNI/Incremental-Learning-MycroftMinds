from sklearn.model_selection import train_test_split
from tree import BaseTree, Node
import numpy as np
import heapq

def hoeffding_bound(n, delta):
    """Computes Hoeffding bound given number of samples and confidence level."""
    """n - samples """
    """delta - confidence factor """
    return np.sqrt(np.log(1 / delta) / (2 * n))

import numpy as np
from collections import defaultdict

class Node:
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

        if not self.left_y or not self.right_y:
            return 0
        var_before = (self.sum_y_squared / self.samples_seen) - (self.sum_y / self.samples_seen) ** 2
        var_left = np.var(self.left_y) if self.left_y else 0
        var_right = np.var(self.right_y) if self.right_y else 0
        return var_before - ((len(self.left_y) / self.samples_seen) * var_left + (len(self.right_y) / self.samples_seen) * var_right)
    
    def split(self, split_info):
        self.feature, self.threshold = split_info
        self.left = Node()
        self.right = Node()

    def show(self,prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")


class FastODT(BaseTree):
    def __init__(self, depth = 5, grace_period=2, split_confidence=0.0001):
        self.grace_period = grace_period  # Number of samples before checking for a split
        self.split_confidence = split_confidence  # Confidence level for splitting
        self.root = Node()
        self.depth = depth
        self.splits = [None] * depth

    def partial_fit(self, X, y):
        for i in range(len(X)):
            self._update_tree(self.root, X[i], y[i], depth = 0)

    def _update_tree(self, node , x, y, depth = 0):

        # this should be updated later
        if node.left is None and node.right is None:
            node.update_stats(x, y)
            if node.samples_seen >= self.grace_period:
                self._attempt_split(node, depth)
        
        elif x[node.feature] <= node.threshold:
            self._update_tree(node.left,x,y,depth + 1)

        elif x[node.feature] > node.threshold:
            self._update_tree(node.right,x,y,depth + 1)

    def _attempt_split(self, node, depth):
        best_split, best_reduction = node.find_best_split()
        if self.splits[depth] is not None:
            node.split(self.splits[depth])

        elif best_split and best_reduction > self.split_confidence:
            self.splits[depth] = best_split 
            node.split(best_split)

if __name__ == "__main__":

    Fodt = FastODT()

    X = np.array([
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 4], [3, 4], [5, 4], [4, 6], [2, 2],
        [5, 3], [1, 6], [2, 5], [1, 3], [5, 2],
        # [3, 5], [4, 3], [1, 2], [2, 6], [5, 6],
        # [3, 2], [2, 1], [4, 5], [1, 5], [2, 4],
        # [5, 1], [4, 4], [3, 1], [1, 1], [2, 2]
    ])

    Y = np.array([
        10, 12, 15, 13, 16, 10, 12, 15, 13, 16,
        11, 14, 17, 18, 9, 13, 14, 11, 12, 8,
        # 9, 11, 5, 8, 15,
        # 7, 4, 10, 6, 9,
        # 3, 12, 5, 2, 7
    ])
    print(len(X),len(Y))
    Fodt.partial_fit(X, Y)
    # print("Check values: ")
    for i in range(len(X)):
        print(f"predictions[{X[i]}]: {Fodt.predict(X[i])} == {Y[i]}")

    # htr.show()