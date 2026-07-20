"""Inverse-frequency class weighting (numpy only)."""
import numpy as np
import pytest

from fl_med.losses import inverse_frequency_weights


def test_balanced_counts_give_uniform_weights():
    w = inverse_frequency_weights([10, 10, 10, 10])
    assert np.allclose(w, [1.0, 1.0, 1.0, 1.0])


def test_rare_class_gets_higher_weight():
    w = inverse_frequency_weights([100, 1])  # class 1 is 100x rarer
    assert w[1] > w[0]
    assert w[1] / w[0] == pytest.approx(100.0)


def test_absent_class_gets_zero_weight():
    w = inverse_frequency_weights([50, 0, 50])
    assert w[1] == 0.0


def test_normalization_mean_is_one_over_present():
    w = np.asarray(inverse_frequency_weights([100, 10, 1]))
    assert np.mean(w[w > 0]) == pytest.approx(1.0)
