"""FastODT learning semantics with Boolean-vector routing and indexed nodes.

Node statistics, histogram maintenance, split selection, child initialization and
grace-period rules are inherited from FastODT. Only lookup/routing is replaced.
Prediction falls back to an existing ancestor without allocating model state.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from src.trees.fastodt import FastODT
from src.trees.node_hf import NodeHoeffdingBound


class xFastODT(FastODT):
    """Use FastODT's learner with independent per-depth Boolean comparisons.

    ``depth=D`` permits D shared splits and leaves at depth D, with the root at
    depth zero, exactly as in FastODT. Indexed containers reference the same nodes
    as the pointer tree; they do not hold separate observation statistics.
    """

    def __init__(
        self,
        depth: int = 25,
        grace_period: int = 10,
        split_confidence: float = 0.01,
        tie_threshold: float = 0.05,
        max_feature: float = 1.0,
        grace_period_depth: int = -1,
        bin_type: str = "adaptive",
        max_bins: int = 16,
        inherit_bins: bool = False,
        split_criterion: str = "vr_rel",
        random_state: Any = None,
        eagerness: float = 0.0,
    ) -> None:
        if isinstance(depth, bool) or not isinstance(depth, (int, np.integer)) or depth < 0:
            raise ValueError("depth must be a nonnegative integer")
        super().__init__(
            depth=depth,
            grace_period=grace_period,
            split_confidence=split_confidence,
            tie_threshold=tie_threshold,
            max_feature=max_feature,
            eagerness=eagerness,
            grace_period_depth=grace_period_depth,
            bin_type=bin_type,
            max_bins=max_bins,
            inherit_bins=inherit_bins,
            split_criterion=split_criterion,
            random_state=random_state,
        )
        self.current_depth = 0
        self.mature_tree = depth == 0
        self.root.index = 0
        self.depths_containers: list[dict[int, NodeHoeffdingBound]] = [{} for _ in range(depth + 1)]
        self.depths_containers[0][0] = self.root
        self._routing_features = np.zeros(depth, dtype=np.intp)
        self._routing_thresholds = np.zeros(depth, dtype=np.float64)

    @property
    def splits_features(self) -> list[int | None]:
        """Registered shared split features, including unused depth slots."""
        return [None if split is None else split[0] for split in self.splits]

    @property
    def splits_values(self) -> list[float | None]:
        """Registered shared thresholds, including unused depth slots."""
        return [None if split is None else split[1] for split in self.splits]

    def _decision_index(self, sample: np.ndarray) -> int:
        features = self._routing_features[: self.current_depth]
        thresholds = self._routing_thresholds[: self.current_depth]
        # Equality goes left. Negated <= also matches FastODT for NaN prediction
        # inputs; for finite values this is identical to sample[features] > thresholds.
        decisions = np.logical_not(sample[features] <= thresholds)
        return sum(1 << bit for bit, decision in enumerate(reversed(decisions)) if decision)

    def _existing_node(self, depth: int, index: int) -> NodeHoeffdingBound:
        while True:
            node = self.depths_containers[depth].get(index)
            if node is not None:
                return node
            depth -= 1
            index //= 2

    def _training_node(self, depth: int, index: int) -> NodeHoeffdingBound:
        missing = []
        node = self.depths_containers[depth].get(index)
        while node is None:
            missing.append((depth, index))
            depth -= 1
            index //= 2
            node = self.depths_containers[depth].get(index)
        for child_depth, child_index in reversed(missing):
            # Apply the registered split exactly as pointer-based FastODT does
            # when it next visits this leaf: create both children and inherit state.
            self._attempt_split(node, child_depth - 1)
            node = self.depths_containers[child_depth][child_index]
        return node

    def _attempt_split(self, node: NodeHoeffdingBound, depth: int) -> None:
        super()._attempt_split(node, depth)
        if node.left is None or node.right is None:
            return
        node.left.index = 2 * node.index
        node.right.index = 2 * node.index + 1
        self.depths_containers[depth + 1][node.left.index] = node.left
        self.depths_containers[depth + 1][node.right.index] = node.right
        feature, threshold = self.splits[depth]
        self._routing_features[depth] = feature
        self._routing_thresholds[depth] = threshold
        self.current_depth = max(self.current_depth, depth + 1)
        self.mature_tree = self.current_depth >= self.depth

    def _update_tree(
        self,
        node: NodeHoeffdingBound,
        x: np.ndarray,
        y: float,
        depth: int = 0,
        payload_print: str = "",
        debug_index: int = 0,
    ) -> None:
        depth = self.current_depth
        index = self._decision_index(x)
        node = self._training_node(depth, index)
        node.update_stats(x, y)
        if depth >= self.depth:
            return
        if node.reached_grace_period() or self.grace_period_depth < depth:
            self._attempt_split(node, depth)

    def predict(self, X: Any) -> np.ndarray | float:
        """Predict by Boolean-vector index, without materializing missing nodes."""
        features = np.asarray(X)
        if features.ndim == 1:
            return self._existing_node(self.current_depth, self._decision_index(features)).value
        if features.ndim != 2:
            raise ValueError(f"Received data with shape {features.shape}, but expected 1D or 2D array.")
        return np.array(
            [self._existing_node(self.current_depth, self._decision_index(sample)).value for sample in features]
        )


if __name__ == "__main__":
    import time

    from matplotlib import pyplot as plt
    from sklearn.metrics import mean_absolute_error

    X = np.load("./smoke_data/X.npy", allow_pickle=True)[:, 2:]
    Y = np.load("./smoke_data/y.npy", allow_pickle=True)
    model_hf = xFastODT(depth=50, grace_period=10, split_confidence=0.0001)
    started = time.time()
    model_hf.partial_fit(X, Y, verbose=True)
    print(f"It took: {time.time() - started}s")
    Y_pred_hf = model_hf.predict(X)
    print(f"MAE: {mean_absolute_error(Y, Y_pred_hf):.4f}")
    plt.plot(Y[3000:3500], label="Actual Y", color="black", linewidth=2)
    plt.plot(Y_pred_hf[3000:3500], label="Predicted Y", color="red", linewidth=2)
    plt.legend()
    plt.show()
