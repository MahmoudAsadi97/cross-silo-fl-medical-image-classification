"""Heterogeneity metrics (numpy only)."""
import numpy as np
import pytest

from fl_med.data.heterogeneity import (
    emd_1d, hellinger_distance, heterogeneity_report, js_distance,
    kl_divergence, normalized_entropy, shannon_entropy,
)


def test_uniform_entropy_equals_log_k():
    p = np.full(8, 1 / 8)
    assert shannon_entropy(p) == pytest.approx(np.log(8))
    assert normalized_entropy(p) == pytest.approx(1.0)


def test_one_hot_entropy_is_zero():
    p = np.array([1.0, 0, 0, 0])
    assert shannon_entropy(p) == pytest.approx(0.0)
    assert normalized_entropy(p) == pytest.approx(0.0)


def test_divergences_zero_for_identical_distributions():
    p = np.array([0.5, 0.3, 0.2])
    assert kl_divergence(p, p) == pytest.approx(0.0, abs=1e-9)
    assert js_distance(p, p) == pytest.approx(0.0, abs=1e-6)
    assert hellinger_distance(p, p) == pytest.approx(0.0, abs=1e-9)
    assert emd_1d(p, p) == pytest.approx(0.0, abs=1e-9)


def test_metric_bounds():
    p = np.array([1.0, 0.0, 0.0])
    q = np.array([0.0, 0.0, 1.0])
    assert 0.0 <= js_distance(p, q) <= 1.0 + 1e-9
    assert 0.0 <= hellinger_distance(p, q) <= 1.0 + 1e-9
    assert hellinger_distance(p, q) == pytest.approx(1.0)  # disjoint support


def test_report_shapes_and_missing_classes():
    counts = np.array([[10, 10, 10], [30, 0, 0]])  # client1 missing 2 classes
    rep = heterogeneity_report(counts, client_ids=[0, 1], class_names=["a", "b", "c"])
    assert rep["num_clients"] == 2 and rep["num_classes"] == 3
    r0, r1 = rep["per_client"]
    assert r0["num_classes_missing"] == 0
    assert r1["num_classes_missing"] == 2
    assert r1["js_to_global"] > r0["js_to_global"]  # skewed client is further
