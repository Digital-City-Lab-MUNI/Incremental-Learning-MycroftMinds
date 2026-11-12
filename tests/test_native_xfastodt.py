"""Full online FastODT/Python-vector/C-vector parity tests and native memory."""

import unittest
from pathlib import Path

import numpy as np
from sklearn.base import clone

from src.trees.fastodt import FastODT
from src.trees.native_xfastodt import NativeXFastODT
from src.trees.xfastodt import xFastODT
from src.utils.memory import calculate_memory_usage, model_memory_breakdown

NATIVE_LIBRARY = Path(__file__).resolve().parents[1] / "src/trees/_native/libxfastodt.so"


def stream_fixture(rows: int = 320) -> tuple[np.ndarray, np.ndarray]:
    """Generate finite features, exact ties and changing target means."""
    rng = np.random.default_rng(19)
    features = np.round(rng.normal(size=(rows, 5)), 1)
    targets = features[:, 0] * 3 + features[:, 1] ** 2 - features[:, 3]
    targets[rows // 2 :] += 2
    return features, targets


def compare_stream(params: dict, features: np.ndarray, targets: np.ndarray) -> tuple[FastODT, NativeXFastODT]:
    """Compare both indexed implementations against pointer-based FastODT."""
    reference = FastODT(**params)
    vector = xFastODT(**params)
    native = NativeXFastODT(**params)
    try:
        for index, (row, target) in enumerate(zip(features, targets, strict=True)):
            expected = reference.predict(row.reshape(1, -1))
            np.testing.assert_allclose(expected, native.predict(row), rtol=1e-10, atol=1e-10)
            np.testing.assert_array_equal(expected, vector.predict(row.reshape(1, -1)))
            reference.partial_fit(row, [target])
            vector.partial_fit(row, [target])
            native.partial_fit(row, [target])
            if index % 29 == 0:
                # Query unvisited regions to exercise read-only ancestor fallback.
                queries = features[index : index + 9]
                np.testing.assert_allclose(reference.predict(queries), native.predict(queries), rtol=1e-10, atol=1e-10)
                np.testing.assert_array_equal(reference.predict(queries), vector.predict(queries))
        np.testing.assert_allclose(reference.predict(features), native.predict(features), rtol=1e-10, atol=1e-10)
        np.testing.assert_array_equal(reference.predict(features), vector.predict(features))
        expected_features = [None if split is None else split[0] for split in reference.splits]
        expected_values = [None if split is None else split[1] for split in reference.splits]
        assert vector.current_depth == native.current_depth == sum(split is not None for split in reference.splits)
        assert vector.mature_tree == native.mature_tree == all(split is not None for split in reference.splits)
        assert reference.is_filled() == vector.is_filled() == native.is_filled()
        assert reference.how_deep() == vector.how_deep() == native.how_deep()
        assert expected_features == vector.splits_features == native.splits_features
        for expected, actual in zip(expected_values, native.splits_values, strict=True):
            if expected is None:
                assert actual is None
            else:
                np.testing.assert_allclose(expected, actual, rtol=1e-10, atol=1e-10)
        return reference, native
    except BaseException:
        native.close()
        raise


@unittest.skipUnless(NATIVE_LIBRARY.exists(), "build the C learner with `make native` first")
class NativeXFastODTTests(unittest.TestCase):
    """Keep training behavior unchanged while replacing Python state with C storage."""

    def test_stream_parity_across_histograms_criteria_and_subsampling(self) -> None:
        """Match both histogram modes, both variance criteria, and seeded subsets."""
        features, targets = stream_fixture()
        for histogram in ("adaptive", "mean"):
            for criterion in ("vr_abs", "vr_rel"):
                for fraction in (1.0, 0.6):
                    with self.subTest(histogram=histogram, criterion=criterion, fraction=fraction):
                        _, native = compare_stream(
                            {
                                "depth": 5,
                                "grace_period": 3,
                                "split_confidence": 0.5,
                                "tie_threshold": 0.1,
                                "max_feature": fraction,
                                "bin_type": histogram,
                                "max_bins": 4,
                                "split_criterion": criterion,
                                "random_state": 42,
                            },
                            features,
                            targets,
                        )
                        native.close()

    def test_batch_training_mature_updates_and_inheritance(self) -> None:
        """Batch updates and inherited histograms preserve final predictions."""
        features, targets = stream_fixture()
        for histogram in ("adaptive", "mean"):
            for inherit in (False, True):
                with self.subTest(histogram=histogram, inherit=inherit):
                    params = dict(
                        depth=3,
                        grace_period=0,
                        split_confidence=1,
                        tie_threshold=0.5,
                        bin_type=histogram,
                        max_bins=4,
                        inherit_bins=inherit,
                        random_state=7,
                    )
                    reference = FastODT(**params)
                    native = NativeXFastODT(**params)
                    self.addCleanup(native.close)
                    for start in range(0, len(features), 17):
                        reference.partial_fit(features[start : start + 17], targets[start : start + 17])
                        native.partial_fit(features[start : start + 17], targets[start : start + 17])
                        np.testing.assert_allclose(reference.predict(features), native.predict(features))
                    self.assertEqual(native.current_depth, sum(split is not None for split in reference.splits))
                    self.assertEqual(native.mature_tree, all(split is not None for split in reference.splits))

    def test_sparse_indices_beyond_64_bits(self) -> None:
        """Deep routing must not truncate the MSB-first path to a machine word."""
        # Increasing values always visit the newest rightmost leaf, producing
        # one shared split every two observations instead of scattering updates.
        features = np.arange(240.0).reshape(-1, 1)
        targets = features[:, 0].copy()
        _, native = compare_stream(
            dict(depth=100, grace_period=0, split_confidence=1, tie_threshold=1, bin_type="mean", random_state=42),
            features,
            targets,
        )
        self.addCleanup(native.close)
        self.assertGreater(native.current_depth, 64)

    def test_constant_targets_depth_one_and_zero_selected_features(self) -> None:
        """Degenerate targets and feature subsets must not create spurious splits."""
        features, _ = stream_fixture(60)
        for params in ({"depth": 0}, {"depth": 1}, {"depth": 4, "max_feature": 0.1}, {"depth": 4, "max_bins": 1}):
            _, native = compare_stream(dict(params, random_state=42), features, np.full(len(features), 3.0))
            native.close()

    def test_memory_accounting_and_explicit_release(self) -> None:
        """Reported object memory includes live C allocations, with separate peak bytes."""
        features, targets = stream_fixture(600)
        reference, native = compare_stream(
            dict(depth=6, grace_period=0, split_confidence=0.5, tie_threshold=0.1, max_bins=8, random_state=42),
            features,
            targets,
        )
        self.addCleanup(native.close)
        memory = model_memory_breakdown(native)
        self.assertGreater(memory["native_bytes"], 0)
        self.assertGreaterEqual(memory["native_peak_bytes"], memory["native_bytes"])
        self.assertEqual(memory["total_bytes"], memory["python_bytes"] + memory["native_bytes"])
        self.assertAlmostEqual(calculate_memory_usage(native), memory["total_bytes"] / 1024**2)
        pending = [reference.root]
        reference_nodes = 0
        while pending:
            node = pending.pop()
            reference_nodes += 1
            if node.left is not None:
                pending.extend([node.left, node.right])
        self.assertEqual(native.n_nodes_, reference_nodes)
        print(
            f"Synthetic 600-row model: Python={calculate_memory_usage(reference):.6f} MiB, "
            f"C+wrapper={calculate_memory_usage(native):.6f} MiB, native={memory['native_bytes']} bytes, "
            f"nodes={native.n_nodes_}"
        )
        native.close()
        self.assertEqual(native.native_memory_bytes, 0)
        native.close()
        with self.assertRaises(ValueError):
            native.predict(features)

    def test_validation_refit_and_sklearn_clone(self) -> None:
        """Reject invalid inputs, support standard estimator cloning, and reset on fit."""
        features, targets = stream_fixture(50)
        native = NativeXFastODT(depth=4, random_state=42)
        self.addCleanup(native.close)
        native.fit(features, targets)
        expected = native.predict(features)
        native.fit(features, targets)
        np.testing.assert_array_equal(expected, native.predict(features))
        copied = clone(native)
        self.addCleanup(copied.close)
        copied.partial_fit(features, targets)
        np.testing.assert_array_equal(expected, copied.predict(features))
        for invalid in (np.full((2, 5), np.nan), np.full((2, 5), np.inf), np.ones((2, 4))):
            with self.assertRaises(ValueError):
                native.predict(invalid)
        with self.assertRaises(ValueError):
            native.partial_fit(features, targets[:-1])
        with self.assertRaises(ValueError):
            native.partial_fit(features[:1], [np.nan])
        for params in (
            {"depth": -1},
            {"depth": 129},
            {"bin_type": "equiwidth"},
            {"inherit_bins": True, "max_feature": 0.5},
        ):
            with self.assertRaises(ValueError):
                NativeXFastODT(**params)


if __name__ == "__main__":
    unittest.main()
