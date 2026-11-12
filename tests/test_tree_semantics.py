"""Focused, data-independent behavioral tests for FastODT / xFastODT update semantics.

Covers the TODO Priority 2 findings:
* threshold-equality routing must be consistent between training and prediction
  (``<=`` goes left, matching histograms, xFastODT mask routing, and Algorithm 1);
* maximum-depth / mature leaves keep updating their prediction statistics
  (the depth guard stops structural growth, not updates);
* grace period is an interval between split attempts, reset after every attempt
  (successful or not), with the depth-limited bypass;
* lazily created xFastODT nodes inherit only the ancestor prediction, with
  zero-initialized observation statistics and split-attempt counters.

Run from the repo root:  python3 -m pytest tests/test_tree_semantics.py -v
"""

from unittest import mock

import numpy as np
import pytest

from src.trees.fastodt import FastODT
from src.trees.xfastodt import xFastODT


def make_fastodt(**kwargs):
    params = dict(
        depth=5,
        grace_period=0,
        split_confidence=1.0,  # Hoeffding bound e = 0 -> any candidate splits
        tie_threshold=0.05,
        max_feature=1.0,
        bin_type="adaptive",
        max_bins=8,
        split_criterion="vr_rel",
        random_state=0,
    )
    params.update(kwargs)
    return FastODT(**params)


def make_xfastodt(**kwargs):
    params = dict(
        depth=5,
        grace_period=0,
        split_confidence=1.0,
        tie_threshold=0.05,
        max_feature=1.0,
        bin_type="adaptive",
        max_bins=8,
        split_criterion="vr_rel",
        random_state=0,
    )
    params.update(kwargs)
    return xFastODT(**params)


class TestThresholdEqualityRouting:
    def test_equal_sample_trains_and_predicts_same_leaf(self):
        """A sample exactly at the threshold must update and be predicted from
        the left (<=) leaf in FastODT."""
        tree = make_fastodt(depth=3)
        tree.learn_one(np.array([0.0]), 0.0)
        tree.learn_one(np.array([1.0]), 10.0)  # root splits at threshold 0.5
        assert tree.root.feature == 0
        assert tree.root.threshold == pytest.approx(0.5)

        tree.learn_one(np.array([0.5]), 100.0)  # equality routes left
        assert tree.root.left.samples_seen == 1
        assert tree.root.left.value == pytest.approx(100.0)
        # prediction for the same threshold-equal sample comes from the left leaf
        assert tree.predict(np.array([0.5])) == pytest.approx(100.0)
        # the right leaf is untouched
        assert tree.root.right.samples_seen == 0

    def test_xfastodt_routes_equality_left(self):
        """xFastODT mask routing: bit is x > threshold, so equality goes left."""
        model = make_xfastodt(depth=3)
        model.learn_one(np.array([0.0]), 0.0)
        model.learn_one(np.array([1.0]), 10.0)  # split at depth 0, threshold 0.5
        assert model.current_depth == 1
        model.learn_one(np.array([0.5]), 100.0)  # 0.5 > 0.5 is False -> left (index 0)
        left = model.depths_containers[1][0]
        assert left.samples_seen == 1
        assert left.value == pytest.approx(100.0)
        assert model.predict(np.array([[0.5]])) == [pytest.approx(100.0)]


class TestMaxDepthPredictionUpdates:
    def test_fastodt_max_depth_leaf_keeps_updating(self):
        """Depth-1 leaf inheriting prediction 10 must move to 100 (count 1)
        after training on target 100, instead of staying at 10 with 0 observations."""
        tree = make_fastodt(depth=1)
        tree.learn_one(np.array([0.0]), 5.0)
        tree.learn_one(np.array([1.0]), 15.0)  # root splits; children inherit value 10
        assert tree.root.left is not None
        left = tree.root.left
        with mock.patch.object(left, "find_best_split", wraps=left.find_best_split) as spy:
            tree.learn_one(np.array([0.0]), 100.0)  # routes to the depth-1 leaf
        assert left.samples_seen == 1
        assert left.value == pytest.approx(100.0)
        assert spy.call_count == 0  # no structural growth at max depth
        assert tree.predict(np.array([0.0])) == pytest.approx(100.0)

    def test_xfastodt_mature_tree_keeps_updating(self):
        """Once mature (max depth reached), xFastODT must still update leaf
        prediction statistics."""
        model = make_xfastodt(depth=1)
        model.learn_one(np.array([0.0]), 0.0)
        model.learn_one(np.array([1.0]), 10.0)  # split at deepest level -> mature
        assert model.mature_tree is True
        model.learn_one(np.array([0.0]), 100.0)
        root = model.depths_containers[0][0]
        assert root.samples_seen == 2
        assert root.left.samples_seen == 1
        assert root.left.value == pytest.approx(100.0)
        assert model.current_depth == 1
        assert model.predict(np.array([[0.0]])) == [pytest.approx(root.left.value)]


class TestGracePeriod:
    def test_fastodt_attempts_spaced_by_grace_period(self):
        """Grace period 2 with six constant samples: split attempts at samples
        3 and 6 only (offset resets after each attempt), not on every sample."""
        tree = make_fastodt(depth=5, grace_period=2, split_confidence=0.01)
        with mock.patch.object(tree.root, "find_best_split", wraps=tree.root.find_best_split) as spy:
            for _ in range(6):
                tree.learn_one(np.array([0.0]), 7.0)
        assert spy.call_count == 2
        assert tree.root.samples_seen == 6

    def test_fastodt_grace_period_blocked_below_threshold(self):
        """With grace period 1000 and no depth-limited bypass, no attempt occurs."""
        tree = make_fastodt(depth=5, grace_period=1000)
        with mock.patch.object(tree.root, "find_best_split", wraps=tree.root.find_best_split) as spy:
            for xv, yv in [(0.0, 0.0), (1.0, 10.0), (0.5, 5.0)]:
                tree.learn_one(np.array([xv]), yv)
        assert spy.call_count == 0
        assert tree.root.left is None  # no split

    def test_fastodt_depth_limited_grace_bypass(self):
        """Beyond grace_period_depth, every sample triggers an attempt."""
        tree = make_fastodt(depth=3, grace_period=2, grace_period_depth=0, split_confidence=1.0)
        tree.learn_one(np.array([0.0]), 0.0)
        tree.learn_one(np.array([1.0]), 10.0)
        tree.learn_one(np.array([0.0]), 0.0)  # sample 3 -> root splits at 0.5
        assert tree.root.feature == 0
        left = tree.root.left
        with mock.patch.object(left, "find_best_split", wraps=left.find_best_split) as spy:
            for yv in (100.0, 200.0, 300.0):
                tree.learn_one(np.array([0.0]), yv)  # depth 1 > grace_period_depth -> bypass
        assert spy.call_count == 3

    def test_xfastodt_grace_period_gates_split_attempts(self):
        """xFastODT must not attempt a split on the first sample even with
        grace period 1000, even when the data warrants a split."""
        model = make_xfastodt(depth=5, grace_period=1000, split_confidence=1.0)
        root = model.depths_containers[0][0]
        with mock.patch.object(root, "find_best_split", wraps=root.find_best_split) as spy:
            for xv, yv in [(0.0, 0.0), (1.0, 10.0), (0.5, 5.0)]:
                model.learn_one(np.array([xv]), yv)
        assert spy.call_count == 0
        assert model.current_depth == 0
        assert all(f is None for f in model.splits_features)
        assert root.samples_seen == 3  # statistics still update

    def test_xfastodt_attempts_spaced_by_grace_period(self):
        """Grace period 2 with six constant samples: attempts at samples 3 and 6."""
        model = make_xfastodt(depth=5, grace_period=2, split_confidence=0.01)
        root = model.depths_containers[0][0]
        with mock.patch.object(root, "find_best_split", wraps=root.find_best_split) as spy:
            for _ in range(6):
                model.learn_one(np.array([0.0]), 7.0)
        assert spy.call_count == 2


class TestXLazyNodeInitialization:
    def test_lazy_node_inherits_prediction_with_fresh_stats(self):
        """Ancestor count 1 / sum 10 plus a lazy child's first target 50 must
        yield child count 1 / sum 50 / value 50 (not count 2 / sum 60)."""
        model = make_xfastodt(depth=3)
        model.learn_one(np.array([0.0]), 10.0)  # root: count 1, sum 10
        model.learn_one(np.array([1.0]), 20.0)  # root splits at 0.5 -> depth 1 children
        assert model.current_depth == 1
        model.learn_one(np.array([0.0]), 10.0)  # node (1,0): count 1, sum 10, value 10
        model.learn_one(np.array([1.0]), 20.0)  # node (1,1): count 1
        model.learn_one(np.array([0.7]), 40.0)  # node (1,1) splits at 0.85 -> depth 2 nodes 2,3
        assert model.current_depth == 2

        # x = 0.2 routes to depth-2 index 0, which was never created -> lazy node
        model.learn_one(np.array([0.2]), 50.0)
        lazy = model.depths_containers[2][0]
        assert lazy is not None
        assert lazy.samples_seen == 1
        assert lazy.sum_y == pytest.approx(50.0)
        assert lazy.value == pytest.approx(50.0)
        assert model.predict(np.array([[0.2]])) == [pytest.approx(50.0)]

    def test_split_children_have_fresh_stats(self):
        """Children produced by a split inherit the prediction but start at
        zero observations."""
        model = make_xfastodt(depth=3)
        model.learn_one(np.array([0.0]), 0.0)
        model.learn_one(np.array([1.0]), 10.0)  # root splits at 0.5
        for idx in (0, 1):
            child = model.depths_containers[1][idx]
            assert child.samples_seen == 0
            assert child.sum_y == 0.0
            assert child.value == pytest.approx(5.0)  # inherited prediction


class TestSmallDeterministicStreams:
    def test_fastodt_repeated_failed_splits_stay_gated(self):
        """Constant stream: every grace-period attempt fails to find a split;
        attempts must still be spaced by the grace period."""
        tree = make_fastodt(depth=4, grace_period=3, split_confidence=0.01)
        with mock.patch.object(tree.root, "find_best_split", wraps=tree.root.find_best_split) as spy:
            for _ in range(10):
                tree.learn_one(np.array([1.0]), 2.0)
        # attempts at samples 4, 8 (offset resets to samples_seen each attempt)
        assert spy.call_count == 2
        assert tree.root.left is None

    def test_xfastodt_depth_saturation(self):
        """depth=1: after the root split the tree is mature; further samples
         update the depth-1 leaves but never attempt another split."""
        model = make_xfastodt(depth=1)
        model.learn_one(np.array([0.0]), 0.0)
        model.learn_one(np.array([1.0]), 10.0)
        assert model.mature_tree is True
        root = model.depths_containers[0][0]
        with mock.patch.object(root, "find_best_split", wraps=root.find_best_split) as spy:
            for xv, yv in [(0.0, 1.0), (1.0, 2.0), (0.5, 3.0)]:
                model.learn_one(np.array([xv]), yv)
        assert spy.call_count == 0
        assert root.samples_seen == 2
        assert root.left.samples_seen == 2
        assert root.right.samples_seen == 1

    def test_both_trees_learn_simple_function(self):
        """End-to-end smoke: both tree variants fit y = 2x + 5 on a fixed grid."""
        rng = np.random.default_rng(1)
        X = rng.uniform(0.0, 10.0, size=(300, 1))
        y = 2.0 * X[:, 0] + 5.0

        fodt = make_fastodt(depth=4, grace_period=5, split_confidence=0.01)
        fodt.partial_fit(X, y)
        mae_fodt = float(np.mean(np.abs([fodt.predict(x) for x in X] - y)))

        xfodt = make_xfastodt(depth=4, grace_period=5, split_confidence=0.01)
        xfodt.partial_fit(X, y)
        mae_xfodt = float(np.mean(np.abs(np.asarray(xfodt.predict(X)) - y)))

        assert mae_fodt < float(np.std(y))
        assert mae_xfodt < float(np.std(y))
