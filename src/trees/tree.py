from sklearn.model_selection import train_test_split
from src.trees.base import BaseLeaner
import numpy as np

class Node:

    def __init__(self, feature = None, threshold = None, left = None, right = None, value = None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def show(self,prefix=""):
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")

class BaseTree(BaseLeaner):

    def __init__(self,depth):
        super()
        self.root = None
        self.depth = depth

    def fit(self,X,y):
        self.root = self.__grow(X,y,depth = 0)

    def predict(self,X) -> float:
        return self.__traverse(X,self.root)

    def __traverse(self,X,node) -> float:
        ret = node.value
        if node.threshold == None:
            return ret

        if(X[node.feature] < node.threshold):
            return self.__traverse(X, node.left)
        else:
            return self.__traverse(X, node.right)

    def __grow(self,X,y,depth) -> Node:

        ''' get mean value for regression'''
        value = sum(y) / len(y)

        ''' check depth '''
        if depth >= self.depth:
            return Node(value = value)

        threshold, feature = self.__split(X,y)
        if threshold == None or feature == None:
            return Node(value = value)

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_child = self.__grow(X[left_indices],y[left_indices],depth+1)
        right_child = self.__grow(X[right_indices],y[right_indices],depth+1)

        return Node(
            value    = value,
            feature  = feature,
            threshold= threshold,
            left     = left_child,
            right    = right_child
        )


    def __split(self,X,y) -> {int,int}:
        __best_variance = 0
        __best_threshold = None
        __best_feature = None

        for feature,_ in enumerate(X[0, :]):
            thresholds = X[:, feature]
            for threshold in thresholds:
                left_mask = X[:,feature] < threshold
                right_mask = ~left_mask

                ''' check if mask is doing any split and not just moving all data into right or left set'''
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                variance = self._variance_reduction(y,y[left_mask],y[right_mask])
                if variance > __best_variance:
                    __best_variance = variance
                    __best_threshold = threshold
                    __best_feature = feature

        return __best_threshold, __best_feature

    def _variance_reduction(self,y,y_left,y_right):
        def variance(labels):
            mean = sum(labels) / len(labels)
            return sum((x - mean) ** 2 for x in labels) / len(labels)

        y_variance = variance(y)
        y_left_variance = variance(y_left)
        y_right_variance = variance(y_right)

        left_weight = len(y_left)/len(y)
        right_weight = len(y_right)/len(y)

        return y_variance - (left_weight * y_left_variance + right_weight * y_right_variance)

    def show(self):
        self.__traverse_show(self.root,"")

    def __traverse_show(self,node,prefix):
        node.show(prefix)
        if(node.left):
            print(f"{prefix}Left:")
            self.__traverse_show(node.left,prefix+" ")
        if(node.right):
            print(f"{prefix}Right:")
            self.__traverse_show(node.right,prefix+" ")

    def show_final(self):
        self.__traverse_show_final(self.root,"")

    def __traverse_show_final(self,node,prefix):
        
        if node.left is None and node.right is None:
            node.show(prefix)
    
        if(node.left):
            self.__traverse_show_final(node.left,prefix+" ")
        if(node.right):
            self.__traverse_show_final(node.right,prefix+" ")

if __name__ == "__main__":

    X = np.array([
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 3], [1, 4], [3, 6], [4, 2], [5, 5],
        [2, 4], [3, 4], [5, 4], [4, 6], [2, 2],
        [5, 3], [1, 6], [2, 5], [1, 3], [5, 2]
    ])

    y = np.array([10, 12, 15, 13, 16, 10, 12, 15, 13, 16, 11, 14, 17, 18, 9, 13, 14, 11, 12, 8])

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = BaseTree(depth=3)
    tree.fit(X_train, y_train)

    print("Seen values: ")
    for i in range(len(X_train)):
        print(f"predictions[{X_train[i]}]: {tree.predict(X_train[i])} == {y_train[i]}")

    print("\nUnseen values: ")
    for i in range(len(X_eval)):
        print(f"predictions[{X_eval[i]}]: {tree.predict(X_eval[i])} == {y_eval[i]}")

    tree.show()