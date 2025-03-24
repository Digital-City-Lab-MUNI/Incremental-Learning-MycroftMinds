from sklearn.model_selection import train_test_split
from tree import BaseTree, Node
import numpy as np
import heapq

def hoeffding_bound(n, delta):
    """Computes Hoeffding bound given number of samples and confidence level."""
    """n - samples """
    """delta - confidence factor """
    return np.sqrt(np.log(1 / delta) / (2 * n))

class FastNode(Node):

    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        super(FastNode, self).__init__(feature,threshold,left,right,value)
        self.n_data_points_hits = 0
        self.y_sum = 0
        self.y_mean = 0
        self.y_sum_pwr_2 = 0

    def add_point(self,y):
        self.y_sum += y
        self.n_data_points_hits += 1
        self.y_sum_2 = self.y_sum**2
        self.y_mean = self.y_sum/self.n_data_points_hits
        self.value = self.y_mean 

    def variance(self):
        if self.n_data_points_hits:
            return self.y_sum_2/self.n_data_points_hits - self.y_mean
        return 0

class Split():

    def __init__(self,feature = None, threshold = None, variance = None):
        self.feature = feature
        self.threshold = threshold
        self.variance = variance

    # Define comparison so that heapq sorts by variance
    def __lt__(self, other):
        return self.variance > other.variance  # Max-heap behavior


class FastODTree(BaseTree):

    def __init__(self,depth):
        super(FastODTree, self).__init__(depth)
        self.splits = [None] * depth
        self.already_used_features = dict()
        self.delta = 0.001
        self.samples_seen = 0
        self.seen_y_sum = 0
        self.seen_y_len = 0
        self.root = FastNode()

    def fit(self,x,y):
        self.__fit_traverse(self.root,x,y,depth = 0)

    def __fit_traverse(self,node,X,y,depth):
        ''' check depth '''
        if depth >= self.depth:
            return

        split = self.splits[depth]
        print(node,split,self.splits,depth)
        if split == None:
            best_split, second_best_split = self.__split(node, X, y)
            if (best_split.variance - second_best_split.variance) < hoeffding_bound(self.samples_seen,self.delta):
                self.splits[depth] = best_split
                node.left = FastNode()
                node.right = FastNode()
                print("split added")
        else:
            if X[split.feature] < split.threshold:
                if node.left == None:
                    node.left = FastNode()
                self.__fit_traverse(node.left,X,y,depth+1)
            else:
                if node.right == None:
                    node.right = FastNode()
                self.__fit_traverse(node.right,X,y,depth+1)

    def __split(self,parent_node,X,y) -> {int,int}:
        self.features_max = []

        print(parent_node)
        parent_node.add_point(y)
        for feature,_ in enumerate(X):
            threshold = X[feature]

            found = False
            for split in self.splits:
                if split == None:
                    continue

                if split.feature == feature and split.threshold == threshold:
                    found = True

            if found:
                continue

            left_mask = X[feature] < threshold
            right_mask = ~left_mask
            if type(left_mask) != list:
                left_mask = [left_mask]
            if type(right_mask) != list:
                right_mask = [right_mask]

            ''' check if mask is doing any split and not just moving all data into right or left set'''
            if len(left_mask) == 0 or len(right_mask) == 0:
                continue

            variance = parent_node.variance()
            split = Split(
                feature,
                threshold,
                variance
            )

            heapq.heappush(self.features_max,split)

        best_split = heapq.heappop(self.features_max)
        second_best_split = heapq.heappop(self.features_max)
        return best_split, second_best_split


if __name__ == "__main__":

    X = np.array([
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 4], [3, 4], [5, 4], [4, 6], [2, 2],
        [5, 3], [1, 6], [2, 5], [1, 3], [5, 2],
        [3, 5], [4, 3], [1, 2], [2, 6], [5, 6],
        [3, 2], [2, 1], [4, 5], [1, 5], [2, 4],
        [5, 1], [4, 4], [3, 1], [1, 1], [2, 2]
    ])

    Y = np.array([
        10, 12, 15, 13, 16, 10, 12, 15, 13, 16,
        11, 14, 17, 18, 9, 13, 14, 11, 12, 8,

        9, 11, 5, 8, 15,
        7, 4, 10, 6, 9,
        3, 12, 5, 2, 7
    ])

    tree = FastODTree(depth=3)
    for x,y in zip(X,Y):
        tree.fit(x, y)

    print("Check values: ")
    for i in range(len(X)):
        print(f"predictions[{X[i]}]: {tree.predict(X[i])} == {Y[i]}")

    # tree.show()