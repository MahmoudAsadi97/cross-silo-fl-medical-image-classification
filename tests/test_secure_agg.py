"""Secure aggregation: masks cancel to the exact aggregate; individuals are hidden."""
from collections import OrderedDict

import numpy as np

from fl_med.security.secure_agg import mask_update, secure_sum, secure_weighted_average
from fl_med.strategies.aggregation import weighted_average


def _updates(n=6, seed=0):
    rng = np.random.default_rng(seed)
    return [OrderedDict(w=rng.normal(size=(4, 3)), b=rng.normal(size=(3,))) for _ in range(n)]


def test_secure_sum_recovers_plain_sum():
    ups = _updates()
    ids = list(range(len(ups)))
    masked = [mask_update(ups[i], i, ids, master_seed=1, scale=50.0) for i in ids]
    got = secure_sum(masked)
    plain = {k: sum(u[k] for u in ups) for k in ups[0]}
    for k in plain:
        assert np.allclose(got[k], plain[k], atol=1e-9)


def test_secure_weighted_average_equals_plaintext_fedavg():
    ups = _updates()
    weights = [7947, 2531, 2156, 1448, 525, 281]
    plain = weighted_average(ups, weights)
    sec = secure_weighted_average(ups, weights, master_seed=42, scale=100.0)["aggregate"]
    for k in plain:
        assert np.allclose(sec[k], plain[k], atol=1e-8)


def test_single_masked_update_is_uninformative():
    ups = _updates()
    out = secure_weighted_average(ups, [1] * len(ups), master_seed=3, scale=100.0)
    true0 = out["scaled_contributions"][0]["w"]
    masked0 = out["masked_updates"][0]["w"]
    assert np.linalg.norm(masked0 - true0) > 20 * np.linalg.norm(true0)
