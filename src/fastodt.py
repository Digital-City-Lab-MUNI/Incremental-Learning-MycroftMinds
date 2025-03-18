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
        self.y_sum_2 = 0

    def add_point(self,y):
        self.y_sum += y
        self.n_data_points_hits += 1
        self.y_sum_2 = self.y_sum**2
        self.y_mean = self.y_sum/self.n_data_points_hits

    def variance(self):
        if self.n_data_points_hits:
            return self.y_sum_2/self.n_data_points_hits - self.y_mean
        return 0

class FastODTree(BaseTree):

    def __init__(self,depth):
        super(FastODTree, self).__init__(depth)
        self.splits = [None] * depth
        self.already_used_features = dict()
        self.delta = 0.001
        self.samples_seen = 0
        self.seen_y_sum = 0
        self.seen_y_len = 0

    def fit(self,x,y):
        self.seen_y_sum += x
        self.seen_y_len += 1
        if self.root:
            self.__fit_traverse(self.root,x,y,depth = 0)
        # if root does not exist, init it with grow function
        else:
            # TODO: variance tracking has to be changed
            self.root = self.__grow(,x,y,depth = 0)

    def __fit_traverse(self,node,X,y,depth):
        ''' check depth '''
        if depth >= self.depth:
            return None

        if(X[node.feature] < node.threshold):
            if node.left:
                self.__fit_traverse(
                    node.left,
                    X,
                    y,
                    depth+1
                )
            else:
                node.left = self.__grow(
                    X,
                    y,
                    depth+1
                )
        else:
            if node.right:
                self.__fit_traverse(
                    node.left,
                    X,
                    y,
                    depth+1
                )
            else:
                node.right = self.__grow(
                    X,
                    y,
                    depth+1
                )

    def __grow(self,parent_node,X,y,depth) -> Node:

        ''' get mean value for regression'''
        if  self.seen_y_len == 0:
            value = 0
        else:
            value = self.seen_y_sum / self.seen_y_len

        # check if split for this level exists, if not then find new split
        # TODO: here we need to add hf_bound
        if not self.splits[depth]:
            best_feature, second_best_feature = self.__split(parent_node,X,y)

            b_feature = best_feature[2]
            b_threshold = best_feature[2]
            sb_feature = second_best_feature[2]

            self.splits[depth][0] = b_feature
            self.splits[depth][1] = b_threshold

            self.samples_seen += 1
            if (b_feature - sb_feature) < hoeffding_bound(self.samples_seen,self.delta):
                # reset seen data
                self.samples_seen = 0
                self.seen_y_sum = 0
                self.seen_y_len = 0

                return FastNode(
                    value    = value,
                    feature  = b_feature,
                    threshold= b_threshold,
                    left     = None,
                    right    = None
                )
            else:
                # if not ready to make split, return None
                return None
        # split already exists so use it
        else:
            b_feature, b_threshold = self.splits[depth][0], self.splits[depth][1]
            new_node = FastNode(
                value    = value,
                feature  = b_feature,
                threshold= b_threshold,
                left     = None,
                right    = None
            )
            self.__fit_traverse(new_node,X,y,depth+1)
            return new_node

    def __split(self,parent_node,x) -> {int,int}:
        self.features_max = []

        for feature,_ in enumerate(x):
            print(f"x: {x} {feature}")
            threshold = x[feature]

            left_mask = x[feature] < threshold
            right_mask = ~left_mask

            print(left_mask,right_mask)
            ''' check if mask is doing any split and not just moving all data into right or left set'''
            if sum(left_mask) == 0 or sum(right_mask) == 0:
                continue

            parent_node.add(y)
            variance = parent_node.variance()
            heapq.heappush(self.features_max,(variance,threshold,feature))

        best_feature = heapq.heappop(self.features_max)
        second_best_feature = heapq.heappop(self.features_max)
        return best_feature, second_best_feature


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

    y = np.array([
        10, 12, 15, 13, 16, 10, 12, 15, 13, 16,
        11, 14, 17, 18, 9, 13, 14, 11, 12, 8,

        9, 11, 5, 8, 15,
        7, 4, 10, 6, 9,
        3, 12, 5, 2, 7
    ])

    tree = FastODTree(depth=3)
    for x,y in zip(X,y):
        tree.fit(x, y)

    print("Check values: ")
    for i in range(len(X)):
        print(f"predictions[{X[i]}]: {tree.predict(X[i])} == {y[i]}")

    # tree.show()