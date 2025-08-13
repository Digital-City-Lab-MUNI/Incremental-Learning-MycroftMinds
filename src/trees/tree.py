from sklearn.model_selection import train_test_split
from src.trees.base import BaseLeaner
import numpy as np


class Node:
    def __init__(
        self,
        feature: int = None,
        threshold: float = None,
        left: "Node" = None,
        right: "Node" = None,
        value: float = None,
    ):
        """Regression tree node.

        Parameters
        ----------
        feature : int, optional
            Index of the feature used for splitting (``None`` for leaf).
        threshold : float, optional
            Threshold value for split (``None`` for leaf).
        left : Node, optional
            Left child.
        right : Node, optional
            Right child.
        value : float, optional
            Prediction value (mean target for the leaf / node).
        """
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def show(self, prefix: str = "") -> None:
        """Print node information with optional prefix."""
        print(f"{prefix}feature: {self.feature}")
        print(f"{prefix}threshold: {self.threshold}")
        print(f"{prefix}value: {self.value}")


class BaseTree(BaseLeaner):
    """
    Base regression tree.

    Parameters
    ----------
    depth : int
        Maximum depth allowed for the tree.
    """

    def __init__(self, depth: int):
        super().__init__()
        self.root: Node = None
        self.depth: int = depth

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """Fit the regression tree to the data.

        Parameters
        ----------
        X : np.ndarray
            Feature matrix ``(n_samples, n_features)``.
        y : np.ndarray
            Target vector ``(n_samples,)``.
        """
        self.root = self._grow(X, y, depth=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict target values for one or many samples.

        Parameters
        ----------
        X : np.ndarray
            1D feature vector or 2D feature matrix.

        Returns
        -------
        np.ndarray or float
            Prediction(s) matching the input dimensionality.
        """
        X = np.asarray(X)
        if X.ndim == 1:
            return self._traverse(X, self.root)
        elif X.ndim == 2:
            return np.array([self._traverse(x, self.root) for x in X])
        else:
            raise ValueError(f"Received data with shape {X.shape}, but expected 1D or 2D array.")

    def _traverse(self, X: np.ndarray, node: Node) -> float:
        """Traverse the tree recursively for a single sample."""
        ret = node.value
        if node.threshold is None:
            return ret

        if X[node.feature] < node.threshold:
            return self._traverse(X, node.left)
        else:
            return self._traverse(X, node.right)

    def _grow(self, X: np.ndarray, y: np.ndarray, depth: int) -> Node:
        """Recursively grow the regression tree returning subtree root."""
        value = float(sum(y)) / len(y)

        if depth >= self.depth:
            return Node(value=value)

        threshold, feature = self._split(X, y)
        if threshold is None or feature is None:
            return Node(value=value)

        left_indices = X[:, feature] < threshold
        right_indices = ~left_indices

        left_child = self._grow(X[left_indices], y[left_indices], depth + 1)
        right_child = self._grow(X[right_indices], y[right_indices], depth + 1)

        return Node(
            value=value,
            feature=feature,
            threshold=threshold,
            left=left_child,
            right=right_child,
        )

    def _split(self, X: np.ndarray, y: np.ndarray) -> tuple:
        """Return (threshold, feature) giving maximum variance reduction."""
        best_variance = 0
        best_threshold = None
        best_feature = None

        for feature, _ in enumerate(X[0, :]):
            thresholds = X[:, feature]
            for threshold in thresholds:
                left_mask = X[:, feature] < threshold
                right_mask = ~left_mask

                if sum(left_mask) == 0 or sum(right_mask) == 0:
                    continue
                variance = self._variance_reduction(y, y[left_mask], y[right_mask])
                if variance > best_variance:
                    best_variance = variance
                    best_threshold = threshold
                    best_feature = feature

        return best_threshold, best_feature

    def _variance_reduction(self, y: np.ndarray, y_left: np.ndarray, y_right: np.ndarray) -> float:
        """Compute absolute variance reduction for given left/right splits."""

        def variance(labels):
            mean = sum(labels) / len(labels)
            return sum((x - mean) ** 2 for x in labels) / len(labels)

        y_variance = variance(y)
        y_left_variance = variance(y_left)
        y_right_variance = variance(y_right)

        left_weight = len(y_left) / len(y)
        right_weight = len(y_right) / len(y)

        return y_variance - (left_weight * y_left_variance + right_weight * y_right_variance)

    def show(self) -> None:
        """Print the structure of the tree."""
        self._traverse_show(self.root, "")

    def _traverse_show(self, node: Node, prefix: str) -> None:
        node.show(prefix)
        if node.left:
            print(f"{prefix}Left:")
            self._traverse_show(node.left, prefix + " ")
        if node.right:
            print(f"{prefix}Right:")
            self._traverse_show(node.right, prefix + " ")

    def show_final(self) -> None:
        """Print only the leaf nodes of the tree."""
        self._traverse_show_final(self.root, "")

    def is_filled(self) -> bool:
        """Return True if the tree is fully grown to the specified depth."""
        return self._check_if_filled(self.root, 0)

    def _check_if_filled(self, node: Node, depth: int = 0) -> bool:
        if node is None:
            return False

        if depth == self.depth:
            return True

        if node.left:
            return self._check_if_filled(node.left, depth + 1)
        if node.right:
            return self._check_if_filled(node.right, depth + 1)
        return False

    def how_deep(self) -> bool:
        """Return depth reached (max depth of any leaf)."""
        return self._how_deep(self.root, 0)

    def _how_deep(self, node: Node, depth: int = 0) -> bool:
        if node is None:
            return depth

        if depth == self.depth:
            return depth

        if node.left:
            return self._how_deep(node.left, depth + 1)
        if node.right:
            return self._how_deep(node.right, depth + 1)
        return depth

    def _traverse_show_final(self, node: Node, prefix: str) -> None:
        if node.left is None and node.right is None:
            node.show(prefix)

        if node.left:
            self._traverse_show_final(node.left, prefix + " ")
        if node.right:
            self._traverse_show_final(node.right, prefix + " ")


if __name__ == "__main__":

    X = np.array(
        [
            [2, 3],
            [1, 4],
            [3, 6],
            [4, 2],
            [5, 5],
            [2, 3],
            [1, 4],
            [3, 6],
            [4, 2],
            [5, 5],
            [2, 4],
            [3, 4],
            [5, 4],
            [4, 6],
            [2, 2],
            [5, 3],
            [1, 6],
            [2, 5],
            [1, 3],
            [5, 2],
        ]
    )

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
