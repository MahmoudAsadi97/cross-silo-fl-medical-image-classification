"""SCAFFOLD control-variate equations, checked on hand-computed toy cases (numpy).

Locks the option-II update  c_i^+ = c_i - c + (x - y_i)/(K*eta_l)  and the server
aggregation, guarding against the sign/scaling bug in the old prototype.
"""
import numpy as np
import pytest

from fl_med.strategies.scaffold import (
    client_deltas,
    server_update,
    updated_client_control,
    zeros_like_state,
)


def test_zeros_like_state():
    s = {"w": np.array([1.0, 2.0, 3.0])}
    z = zeros_like_state(s)
    assert np.allclose(z["w"], 0.0) and z["w"].shape == s["w"].shape


def test_updated_client_control_hand_computed():
    x = {"w": np.array([10.0])}          # global params
    y = {"w": np.array([8.0])}           # local params after K steps
    c = {"w": np.array([0.0])}           # global control variate
    c_i = {"w": np.array([0.0])}         # local control variate
    K, lr = 2, 0.5                       # scale = 1/(K*lr) = 1.0
    new_c_i = updated_client_control(c_i, c, x, y, num_steps=K, lr=lr)
    # c_i - c + (x - y)/(K*lr) = 0 - 0 + (10-8)*1.0 = 2.0
    assert np.allclose(new_c_i["w"], np.array([2.0]))


def test_updated_client_control_no_movement_is_stable():
    x = {"w": np.array([5.0, -5.0])}
    y = {"w": np.array([5.0, -5.0])}     # client did not move
    c = {"w": np.array([1.0, 1.0])}
    c_i = {"w": np.array([1.0, 1.0])}
    new_c_i = updated_client_control(c_i, c, x, y, num_steps=3, lr=0.1)
    # drift term is zero -> c_i^+ = c_i - c = 0
    assert np.allclose(new_c_i["w"], 0.0)


def test_server_update_single_client_hand_computed():
    x = {"w": np.array([10.0])}
    y = {"w": np.array([8.0])}
    c = {"w": np.array([0.0])}
    c_i = {"w": np.array([0.0])}
    new_c_i = updated_client_control(c_i, c, x, y, num_steps=2, lr=0.5)
    d = client_deltas(c_i, new_c_i, x, y)
    assert np.allclose(d["dy"]["w"], np.array([-2.0]))
    assert np.allclose(d["dc"]["w"], np.array([2.0]))

    out = server_update(x, c, [d["dy"]], [d["dc"]], server_lr=1.0, participation=1.0)
    assert np.allclose(out["global_params"]["w"], np.array([8.0]))   # 10 + mean(dy)
    assert np.allclose(out["c_global"]["w"], np.array([2.0]))        # 0 + mean(dc)


def test_server_update_two_clients_mean_of_deltas():
    x = {"w": np.array([0.0])}
    c = {"w": np.array([0.0])}
    dy = [{"w": np.array([2.0])}, {"w": np.array([-4.0])}]
    dc = [{"w": np.array([1.0])}, {"w": np.array([3.0])}]
    out = server_update(x, c, dy, dc, server_lr=1.0, participation=1.0)
    assert np.allclose(out["global_params"]["w"], np.array([-1.0]))  # mean(2,-4)
    assert np.allclose(out["c_global"]["w"], np.array([2.0]))        # mean(1,3)


def test_updated_client_control_validates_args():
    s = {"w": np.array([1.0])}
    with pytest.raises(ValueError):
        updated_client_control(s, s, s, s, num_steps=0, lr=0.1)
    with pytest.raises(ValueError):
        updated_client_control(s, s, s, s, num_steps=1, lr=0.0)
