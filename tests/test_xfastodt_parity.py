"""Routing optimizations must not change FastODT's learning algorithm."""

import unittest

import numpy as np

from src.trees.fastodt import FastODT
from src.trees.xfastodt import xFastODT


def pointer_state(model: FastODT) -> dict:
    """Describe every node's counts, predictions, feature subsets and histograms."""
    state = {}
    pending = [(model.root, 0, 0)]
    while pending:
        node, depth, index = pending.pop()
        histograms = []
        for feature, histogram in (node.histograms or {}).items():
            bins = tuple((item["count"], item["sum_y"], item["sum_y2"]) for item in histogram.bin_stats)
            histograms.append((feature, tuple(histogram.bin_edges), bins))
        state[depth, index] = (
            node.samples_seen,
            node.sum_y,
            node.sum_y_squared,
            node.y_min,
            node.y_max,
            node.value,
            node.grace_period_offset,
            node.feature,
            node.threshold,
            None if node.feature_indices is None else tuple(node.feature_indices),
            tuple(histograms),
        )
        if node.left is not None:
            pending.extend([(node.left, depth + 1, 2 * index), (node.right, depth + 1, 2 * index + 1)])
    return state


class XFastODTParityTests(unittest.TestCase):
    """Compare actual model state, not just predictive error averages."""

    def test_streamed_state_and_predictions_match(self) -> None:
        """All supported shared learning rules match for seeded individual updates."""
        rng = np.random.default_rng(7)
        features = np.round(rng.normal(size=(100, 5)), 1)
        targets = features[:, 0] * 3 + features[:, 2] ** 2
        targets[50:] += 2
        for histogram in ("adaptive", "mean"):
            for depth in (0, 1, 4):
                for grace_depth in (-1, 0):
                    with self.subTest(histogram=histogram, depth=depth, grace_depth=grace_depth):
                        params = dict(
                            depth=depth,
                            grace_period=3,
                            grace_period_depth=grace_depth,
                            split_confidence=0.5,
                            tie_threshold=0.2,
                            max_feature=0.6,
                            bin_type=histogram,
                            max_bins=4,
                            random_state=42,
                        )
                        reference = FastODT(**params)
                        vector = xFastODT(**params)
                        for sample, target in zip(features, targets, strict=True):
                            self.assertEqual(reference.predict(sample), vector.predict(sample))
                            reference.learn_one(sample, target)
                            vector.learn_one(sample, target)
                            self.assertEqual(pointer_state(reference), pointer_state(vector))
                        np.testing.assert_array_equal(reference.predict(features), vector.predict(features))
                        self.assertEqual(reference.rng.bit_generator.state, vector.rng.bit_generator.state)

    def test_prediction_does_not_materialize_missing_nodes(self) -> None:
        """Unvisited indexed paths return the same ancestor value without allocations."""
        params = dict(depth=3, grace_period=0, split_confidence=1, random_state=42)
        reference = FastODT(**params)
        vector = xFastODT(**params)
        for value, target in ((0.0, 10), (1.0, 20), (0.0, 10), (1.0, 20), (0.7, 40)):
            reference.learn_one([value], target)
            vector.learn_one([value], target)
        self.assertEqual(vector.current_depth, 2)
        before = pointer_state(vector)
        sizes = [len(level) for level in vector.depths_containers]
        queries = np.array([[0.2], [0.85], [0.9], [np.nan]])
        np.testing.assert_array_equal(reference.predict(queries), vector.predict(queries))
        self.assertEqual(before, pointer_state(vector))
        self.assertEqual(sizes, [len(level) for level in vector.depths_containers])
        reference.learn_one([0.2], 50)
        vector.learn_one([0.2], 50)
        self.assertEqual(pointer_state(reference), pointer_state(vector))
        self.assertIn(0, vector.depths_containers[2])
        self.assertIn(1, vector.depths_containers[2])

    def test_histogram_inheritance_matches(self) -> None:
        """Registered splits materialize both children with FastODT's inherited histograms."""
        rng = np.random.default_rng(17)
        features = rng.normal(size=(150, 3))
        targets = features[:, 0] + features[:, 2] * 3
        for histogram in ("adaptive", "mean"):
            params = dict(
                depth=4,
                grace_period=0,
                split_confidence=1,
                bin_type=histogram,
                inherit_bins=True,
                random_state=42,
                max_bins=4,
            )
            reference = FastODT(**params)
            vector = xFastODT(**params)
            reference.partial_fit(features, targets)
            vector.partial_fit(features, targets)
            self.assertEqual(pointer_state(reference), pointer_state(vector))
            np.testing.assert_array_equal(reference.predict(features), vector.predict(features))
            self.assertEqual(reference.is_filled(), vector.is_filled())
            self.assertEqual(reference.how_deep(), vector.how_deep())


if __name__ == "__main__":
    unittest.main()
