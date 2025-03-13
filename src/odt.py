from sklearn.model_selection import train_test_split
from tree import BaseTree, Node
import numpy as np
class ODTree(BaseTree):

    def __init__(self,depth):
        super(ODTree, self).__init__(depth)
        self.splits = [None] * depth
        self.already_used_features = dict()

    def fit(self,X,y):
        self.root = self.__grow(X,y,X,y,depth = 0)

    def __grow(self,X,y,X_masked,y_masked,depth) -> Node:

        ''' get mean value for regression'''
        if  len(y_masked) == 0:
            value = 0
        else:
            value = sum(y_masked) / len(y_masked)

        ''' check depth '''
        if depth >= self.depth:
            return Node(value = value)

        # check if split for this level exists, if not then find new split
        if not self.splits[depth]:
            self.splits[depth] = self.__split(X,y)
        threshold, feature = self.splits[depth][0], self.splits[depth][1]

        if threshold == None or feature == None:
            return Node(value = value)

        left_indices = X_masked[:, feature] < threshold
        right_indices = ~left_indices

        left_child = self.__grow(X,y,X_masked[left_indices],y_masked[left_indices],depth+1)
        right_child = self.__grow(X,y,X_masked[right_indices],y_masked[right_indices],depth+1)

        return Node(
            value    = value,
            feature  = feature,
            threshold= threshold,
            left     = left_child,
            right    = right_child
        )

    def __areAllFeaturesUsed(self,X):
        ret = True
        for feature,_ in enumerate(X[0, :]):
            if feature not in self.already_used_features:
                ret = False
        return ret

    def __split(self,X,y) -> {int,int}:
        __best_variance = 0
        __best_threshold = None
        __best_feature = None

        for feature,_ in enumerate(X[0, :]):

            # if all features were used, then allow for reusing feature
            if feature in self.already_used_features and not self.__areAllFeaturesUsed(X):
                continue

            thresholds = X[:, feature]
            for threshold in thresholds:

                if feature in self.already_used_features:
                    if threshold in self.already_used_features[feature]:
                        continue

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

        if __best_feature != None:
            self.already_used_features[__best_feature] = [__best_threshold]
        return __best_threshold, __best_feature


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

    X_train, X_eval, y_train, y_eval = train_test_split(X, y, test_size=0.2, random_state=42)

    tree = ODTree(depth=3)
    tree.fit(X_train, y_train)

    print("Seen values: ")
    for i in range(len(X_train)):
        print(f"predictions[{X_train[i]}]: {tree.predict(X_train[i])} == {y_train[i]}")

    print("\nUnseen values: ")
    for i in range(len(X_eval)):
        print(f"predictions[{X_eval[i]}]: {tree.predict(X_eval[i])} == {y_eval[i]}")

    # tree.show()