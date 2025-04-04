from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
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

    tree = ODTree(depth=3)
    # Generate synthetic data following a sine wave pattern
    X = np.linspace(0, 10, 100).reshape(-1, 1)  # Features: 100 points from 0 to 10
    Y = np.sin(X).ravel() + np.random.normal(0, 0.1, X.shape[0])  # Sine wave with noise

    tree.fit(X, Y)
    tree.show()
    # Make predictions
    Y_pred = np.array([tree.predict(x) for x in X])

    # Compute prediction accuracy using Mean Absolute Error (MAE)
    mae = mean_absolute_error(Y, Y_pred)
    print(f"Mean Absolute Error: {mae:.4f}")

    # Plot the results
    plt.scatter(X, Y, label="Actual Data", color="blue", s=10)
    plt.plot(X, Y_pred, label="Predicted Data", color="red")
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("FastODT Predictions vs Actual Data")
    plt.show()
