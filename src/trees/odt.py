"""Oblivious Decision Tree (ODT) implementation for regression.

Applies a single (feature, threshold) split uniformly across all nodes at a
given depth level, enabling fast traversal and compact models.
"""

from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt
from tree import BaseTree, Node
import numpy as np


class ODTree(BaseTree):
    """Oblivious depth-limited regression tree (same split per level).

    Uses one (feature, threshold) pair per depth level applied uniformly to
    all nodes at that level (oblivious structure). Splits are chosen greedily
    by maximizing variance reduction. A feature is not reused until all have
    been used (unless ``depth`` exceeds the number of features).

    Parameters
    ----------
    depth : int
        Maximum tree depth (number of split levels).
    """

    def __init__(self, depth: int) -> None:
        super().__init__(depth)
        self.splits: list[tuple] = [None] * depth
        self.already_used_features: dict[int, list] = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit tree, computing uniform splits per depth level."""
        self.root = self.__grow(X, y, X, y, depth=0)

    def __grow(self, X: np.ndarray, y: np.ndarray, X_masked: np.ndarray, y_masked: np.ndarray, depth: int) -> Node:
        """Recursive builder applying level-wise (oblivious) splits.

        Returns a leaf when maximum depth reached or no valid split exists.
        """
        if len(y_masked) == 0:
            value = 0
        else:
            value = sum(y_masked) / len(y_masked)
        # depth check
        if depth >= self.depth:
            return Node(value=value)

        # check if split for this level exists, if not then find new split
        if not self.splits[depth]:
            self.splits[depth] = self.__split(X, y)
        threshold, feature = self.splits[depth]

        if threshold is None or feature is None:
            return Node(value=value)

        left_indices = X_masked[:, feature] < threshold
        right_indices = ~left_indices

        left_child = self.__grow(X, y, X_masked[left_indices], y_masked[left_indices], depth + 1)
        right_child = self.__grow(X, y, X_masked[right_indices], y_masked[right_indices], depth + 1)

        return Node(
            value=value,
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def __areAllFeaturesUsed(self, X: np.ndarray) -> bool:
        """Return True if each feature index appears in already-used map."""
        return all(feature in self.already_used_features for feature, _ in enumerate(X[0, :]))

    def __split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Find best (threshold, feature) maximizing variance reduction."""
        __best_variance: float = 0
        __best_threshold = None
        __best_feature = None

        for feature, _ in enumerate(X[0, :]):
            # if all features were used, then allow for reusing feature
            if feature in self.already_used_features and not self.__areAllFeaturesUsed(X):
                continue

            thresholds = X[:, feature]
            for threshold in thresholds:
                if feature in self.already_used_features:
                    if threshold in self.already_used_features[feature]:
                        continue

                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                # ensure split is non-trivial
                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue

                variance = self._variance_reduction(y, y[left_mask], y[right_mask])
                if variance > __best_variance:
                    __best_variance = variance
                    __best_threshold = threshold
                    __best_feature = feature

        if __best_feature is not None:
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
