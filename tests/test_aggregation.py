"""FedAvg aggregation == sample-weighted mean, verified on toy tensors (numpy).

Backend-agnostic math means these run without torch (brief §5, correctness).
"""
import numpy as np
import pytest

from fl_med.strategies.aggregation import (
    apply_delta,
    mean_of_deltas,
    state_delta,
    weighted_average,
)


def test_weighted_average_matches_hand_computation():
    # Two clients, one 1-D parameter "w".
    s1 = {"w": np.array([0.0, 10.0])}
    s2 = {"w": np.array([2.0, 20.0])}
    # weights 30 and 10 -> normalized 0.75 / 0.25
    out = weighted_average([s1, s2], [30, 10])
    expected = 0.75 * s1["w"] + 0.25 * s2["w"]
    assert np.allclose(out["w"], expected)
    assert np.allclose(out["w"], np.array([0.5, 12.5]))


def test_weighted_average_equal_weights_is_plain_mean():
    states = [{"w": np.array([1.0])}, {"w": np.array([3.0])}, {"w": np.array([5.0])}]
    out = weighted_average(states, [1, 1, 1])
    assert np.allclose(out["w"], np.array([3.0]))


def test_weighted_average_single_client_is_identity():
    s = {"a": np.array([1.0, 2.0]), "b": np.array([[3.0]])}
    out = weighted_average([s], [7])
    assert np.allclose(out["a"], s["a"]) and np.allclose(out["b"], s["b"])


def test_weighted_average_rejects_bad_input():
    with pytest.raises(ValueError):
        weighted_average([], [])
    with pytest.raises(ValueError):
        weighted_average([{"w": np.array([1.0])}], [0])


def test_state_delta_and_apply_delta_roundtrip():
    old = {"w": np.array([1.0, 2.0])}
    new = {"w": np.array([1.5, 2.5])}
    d = state_delta(new, old)
    assert np.allclose(d["w"], np.array([0.5, 0.5]))
    # old + 1.0 * delta == new
    assert np.allclose(apply_delta(old, d, 1.0)["w"], new["w"])
    # server_lr scaling
    assert np.allclose(apply_delta(old, d, 2.0)["w"], np.array([2.0, 3.0]))


def test_mean_of_deltas():
    d1 = {"w": np.array([2.0, 0.0])}
    d2 = {"w": np.array([0.0, 4.0])}
    out = mean_of_deltas([d1, d2])
    assert np.allclose(out["w"], np.array([1.0, 2.0]))
