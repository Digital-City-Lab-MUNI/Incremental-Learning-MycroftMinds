"""Focused, data-independent regression tests for IncubationBoost residual handling.

Covers the TODO Priority 1 findings:
* growing-tree training targets must be captured BEFORE adding the current
  tree's own contribution (the first tree learns ``y`` itself);
* the incubation target after pruning must be the residual of the retained
  prefix (pre-training predictions), not the stale loop residual that still
  contains the rejected tree's contribution;
* the pruning score must be a nonnegative rolling loss (absolute errors).

Run from the repo root:  python3 -m pytest tests/test_boosting_residuals.py -v
"""

import numpy as np
import pytest

from src.boosting import IncubationBoost
from src.trees.fastodt import FastODT
from src.trees.xfastodt import xFastODT


class FakeTree:
    """Minimal stand-in for FastODT to observe IncubationBoost's residual logic."""

    def __init__(self, prediction=0.0, filled=True, errors=None, **kwargs):
        self.prediction = prediction
        self.filled = filled
        self.errors = list(errors) if errors else []
        self.trained_targets = []

    def predict(self, X):
        return self.prediction

    def is_filled(self):
        return self.filled

    def measure_error(self, y, y_pred):
        return self.errors.pop(0) if self.errors else 0.0

    def partial_fit(self, X, y):
        self.trained_targets.append(float(np.asarray(y).ravel()[0]))


class RecordingIncubator(FakeTree):
    """Base-learner factory that records the residual passed to incubate()."""

    captured = []

    def __init__(self, **kwargs):
        super().__init__(prediction=0.0, filled=False)
        self.__class__.captured = []

    def partial_fit(self, X, y):
        self.__class__.captured.append(float(np.asarray(y).ravel()[0]))


def make_model(**kwargs):
    params = dict(length=5, learning_rate=1.0, depth=5, grace_period=100, verbose=False)
    params.update(kwargs)
    return IncubationBoost(**params)


class TestGrowingTreeTarget:
    def test_first_tree_learns_full_target(self):
        """Two constant targets of 10: the first tree must predict 10, not 5."""
        model = make_model()
        model.partial_fit([[0.0], [0.0]], [10.0, 10.0])
        assert len(model.chain) == 1
        assert model.chain[0].root.samples_seen == 2
        assert model.chain[0].predict(np.array([0.0])) == pytest.approx(10.0)

    def test_unfilled_first_tree_trains_on_y_not_y_minus_own_prediction(self):
        """Growing tree at position 0: target is y (pre-contribution residual)."""
        model = make_model(learning_rate=0.5)
        growing = FakeTree(prediction=3.0, filled=False)
        model.chain = [growing]
        model.base_learner = RecordingIncubator
        model._partial_fit([0.0], 10.0)
        assert growing.trained_targets == [10.0]

    def test_growing_tree_trains_on_preceding_prefix_residual_with_learning_rate(self):
        """Growing tree at n>0: target excludes its own (lr-scaled) contribution."""
        model = make_model(learning_rate=0.5)
        filled = FakeTree(prediction=4.0, filled=True, errors=[1.0])
        growing = FakeTree(prediction=2.0, filled=False)
        model.chain = [filled, growing]
        model.base_learner = RecordingIncubator
        model._partial_fit([0.0], 10.0)
        # preceding prefix predicts 4 -> training target 10 - 4 = 6
        # (own contribution 2 * 0.5 = 1 must NOT be subtracted)
        assert growing.trained_targets == [6.0]


class TestIncubationTargetAfterPruning:
    def test_rejected_tree_contribution_excluded(self):
        """y=10, retained prefix predicts 6, rejected tree contributes -2.

        Incubation must receive 10 - 6 = 4, not the stale loop residual 6.
        """
        model = make_model()
        retained = FakeTree(prediction=6.0, filled=True, errors=[1.0])  # improves over inf
        rejected = FakeTree(prediction=-2.0, filled=True, errors=[5.0])  # worse -> break
        model.chain = [retained, rejected]
        model.base_learner = RecordingIncubator
        model._partial_fit([0.0], 10.0)
        assert model.chain == [retained]  # greedy prefix pruning
        assert RecordingIncubator.captured == [4.0]

    def test_no_pruning_uses_retained_chain_residual(self):
        """All mature prefixes improve: incubation target equals the loop residual
        of the full retained chain (learning rate applied to later trees)."""
        model = make_model(learning_rate=0.5)
        t0 = FakeTree(prediction=6.0, filled=True, errors=[1.0])
        t1 = FakeTree(prediction=2.0, filled=True, errors=[0.5])
        model.chain = [t0, t1]
        model.base_learner = RecordingIncubator
        model._partial_fit([0.0], 10.0)
        # chain retained fully; y_pred_strong = 6 + 0.5*2 = 7 -> target 3
        assert model.chain == [t0, t1]
        assert RecordingIncubator.captured == [3.0]

    def test_pruning_score_uses_pretraining_predictions(self):
        """Filled nodes are never trained in the loop, so their predictions are
        pre-training by construction; verify error receives y - prefix_pred."""
        model = make_model()
        seen = []

        class Spy(FakeTree):
            def measure_error(self, y, y_pred):
                seen.append((y, y_pred))
                return 1.0

        t0 = Spy(prediction=6.0, filled=True)
        model.chain = [t0]
        model.base_learner = RecordingIncubator
        model._partial_fit([0.0], 10.0)
        assert seen == [(10.0, 6.0)]


class TestPredictLearningRate:
    def test_predict_applies_learning_rate_to_later_trees(self):
        model = make_model(learning_rate=0.5)
        model.chain = [FakeTree(prediction=4.0), FakeTree(prediction=2.0)]
        assert model.predict([0.0]) == pytest.approx(4.0 + 0.5 * 2.0)


class TestPruningLoss:
    @pytest.mark.parametrize("tree_cls", [FastODT, xFastODT])
    def test_overprediction_penalized_nonnegative(self, tree_cls):
        """Target 0, prediction 100 must give a nonnegative score (was -5 with a
        zero-initialized signed window)."""
        tree = tree_cls(depth=2, grace_period=100)
        score = tree.measure_error(0.0, 100.0)
        assert score == pytest.approx(100.0)
        assert score >= 0.0

    @pytest.mark.parametrize("tree_cls", [FastODT, xFastODT])
    def test_opposite_sign_errors_cannot_cancel(self, tree_cls):
        """Errors +10 and -10 (raw residuals) must not average to 0."""
        tree = tree_cls(depth=2, grace_period=100)
        tree.measure_error(10.0, 0.0)  # raw residual +10
        score = tree.measure_error(-10.0, 0.0)  # raw residual -10
        assert score == pytest.approx(10.0)

    @pytest.mark.parametrize("tree_cls", [FastODT, xFastODT])
    def test_window_capped_at_20(self, tree_cls):
        tree = tree_cls(depth=2, grace_period=100)
        score = 0.0
        for _ in range(25):
            score = tree.measure_error(1.0, 0.0)
        assert len(tree.error_rates) == 20
        assert score == pytest.approx(1.0)


class TestEndToEndSmoke:
    def test_boosting_learns_deterministic_stream(self):
        """Small data-independent smoke run: y = 2x + 5 on a fixed grid."""
        rng = np.random.default_rng(0)
        X = rng.uniform(0.0, 10.0, size=(200, 1))
        y = 2.0 * X[:, 0] + 5.0
        model = IncubationBoost(
            length=3, learning_rate=0.5, depth=4, grace_period=10,
            split_confidence=0.01, tie_threshold=0.05, verbose=False, random_state=42,
        )
        model.partial_fit(X[:150], y[:150])
        preds = np.array([model.predict(x) for x in X[150:]])
        mae = float(np.mean(np.abs(preds - y[150:])))
        assert np.all(np.isfinite(preds))
        assert mae < float(np.std(y))  # better than predicting the mean
