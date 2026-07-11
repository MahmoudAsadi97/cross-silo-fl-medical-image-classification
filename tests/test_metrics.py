"""Metrics validated against hand computation (independent of sklearn)."""
import pytest

from fl_med.metrics import (
    accuracy,
    balanced_accuracy,
    compute_metrics,
    confusion_matrix,
    macro_f1,
)

Y_TRUE = [0, 0, 1, 2]
Y_PRED = [0, 1, 1, 2]
K = 3


def test_confusion_matrix():
    cm = confusion_matrix(Y_TRUE, Y_PRED, K)
    assert cm.tolist() == [[1, 1, 0], [0, 1, 0], [0, 0, 1]]


def test_balanced_accuracy_hand_computed():
    # recalls: class0 1/2, class1 1/1, class2 1/1 -> mean = 0.8333...
    assert balanced_accuracy(Y_TRUE, Y_PRED, K) == pytest.approx(2.5 / 3)


def test_macro_f1_hand_computed():
    # F1s: 2/3, 2/3, 1 -> mean = 0.7777...
    assert macro_f1(Y_TRUE, Y_PRED, K) == pytest.approx((2 / 3 + 2 / 3 + 1) / 3)


def test_accuracy():
    assert accuracy(Y_TRUE, Y_PRED) == pytest.approx(0.75)


def test_balanced_accuracy_ignores_absent_classes():
    # Only class 0 present in y_true; recall = 1/2 -> balanced acc = 0.5
    assert balanced_accuracy([0, 0], [0, 1], 3) == pytest.approx(0.5)


def test_compute_metrics_bundle_keys():
    out = compute_metrics(Y_TRUE, Y_PRED, K)
    for key in ("accuracy", "balanced_accuracy", "macro_f1",
                "per_class_recall", "per_class_precision", "confusion_matrix", "support"):
        assert key in out
    assert out["support"] == [2, 1, 1]
    assert out["per_class_recall"][2] == pytest.approx(1.0)


def test_perfect_prediction_scores_one():
    y = [0, 1, 2, 3, 4, 5, 6, 7]
    out = compute_metrics(y, y, 8)
    assert out["balanced_accuracy"] == pytest.approx(1.0)
    assert out["macro_f1"] == pytest.approx(1.0)
