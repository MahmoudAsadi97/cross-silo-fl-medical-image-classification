#!/usr/bin/env python3
"""Torch-free verification of the correctness-critical math.

Runs the same checks as the numpy portions of ``tests/`` but with only numpy +
pyyaml, so it executes anywhere (including a torch-less environment / minimal CI).
Prints a PASS/FAIL table and exits non-zero on any failure. This is the evidence
that FedAvg aggregation, the SCAFFOLD control-variate equations, the metrics, and
the heterogeneity measures are correct -- independent of the GPU stack.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.heterogeneity import (  # noqa: E402
    hellinger_distance, heterogeneity_report, js_distance, kl_divergence,
    normalized_entropy, shannon_entropy,
)
from fl_med.metrics import balanced_accuracy, compute_metrics, macro_f1  # noqa: E402
from fl_med.strategies.aggregation import mean_of_deltas, weighted_average  # noqa: E402
from fl_med.strategies.scaffold import (  # noqa: E402
    client_deltas, server_update, updated_client_control,
)

CHECKS = []


def check(name):
    def deco(fn):
        CHECKS.append((name, fn))
        return fn
    return deco


def approx(a, b, tol=1e-9):
    return abs(float(a) - float(b)) < tol


@check("FedAvg weighted_average == hand-computed weighted mean")
def _():
    out = weighted_average([{"w": np.array([0.0, 10.0])},
                            {"w": np.array([2.0, 20.0])}], [30, 10])
    assert np.allclose(out["w"], [0.5, 12.5])


@check("mean_of_deltas == elementwise mean")
def _():
    out = mean_of_deltas([{"w": np.array([2.0, 0.0])}, {"w": np.array([0.0, 4.0])}])
    assert np.allclose(out["w"], [1.0, 2.0])


@check("SCAFFOLD c_i^+ = c_i - c + (x - y)/(K*lr)")
def _():
    x, y = {"w": np.array([10.0])}, {"w": np.array([8.0])}
    z = {"w": np.array([0.0])}
    new_c = updated_client_control(z, z, x, y, num_steps=2, lr=0.5)
    assert np.allclose(new_c["w"], [2.0])


@check("SCAFFOLD server_update: global += mean(dy), c += mean(dc)")
def _():
    x, c = {"w": np.array([10.0])}, {"w": np.array([0.0])}
    z = {"w": np.array([0.0])}
    y = {"w": np.array([8.0])}
    new_c = updated_client_control(z, c, x, y, num_steps=2, lr=0.5)
    d = client_deltas(z, new_c, x, y)
    out = server_update(x, c, [d["dy"]], [d["dc"]], server_lr=1.0, participation=1.0)
    assert np.allclose(out["global_params"]["w"], [8.0])
    assert np.allclose(out["c_global"]["w"], [2.0])


@check("balanced_accuracy hand-computed == 2.5/3")
def _():
    assert approx(balanced_accuracy([0, 0, 1, 2], [0, 1, 1, 2], 3), 2.5 / 3)


@check("macro_f1 hand-computed == (2/3+2/3+1)/3")
def _():
    assert approx(macro_f1([0, 0, 1, 2], [0, 1, 1, 2], 3), (2 / 3 + 2 / 3 + 1) / 3)


@check("balanced_accuracy ignores classes absent from y_true")
def _():
    assert approx(balanced_accuracy([0, 0], [0, 1], 3), 0.5)


@check("perfect prediction -> balanced_acc == macro_f1 == 1")
def _():
    out = compute_metrics(list(range(8)), list(range(8)), 8)
    assert approx(out["balanced_accuracy"], 1.0) and approx(out["macro_f1"], 1.0)


@check("uniform entropy == log K; normalized == 1")
def _():
    p = np.full(8, 1 / 8)
    assert approx(shannon_entropy(p), np.log(8)) and approx(normalized_entropy(p), 1.0)


@check("divergences vanish for identical distributions")
def _():
    p = np.array([0.5, 0.3, 0.2])
    assert approx(kl_divergence(p, p), 0.0) and approx(js_distance(p, p), 0.0, 1e-6)
    assert approx(hellinger_distance(p, p), 0.0)


@check("Hellinger == 1 for disjoint support; skewed client further from global")
def _():
    assert approx(hellinger_distance(np.array([1.0, 0, 0]), np.array([0, 0, 1.0])), 1.0)
    rep = heterogeneity_report(np.array([[10, 10, 10], [30, 0, 0]]))
    r0, r1 = rep["per_client"]
    assert r1["num_classes_missing"] == 2 and r1["js_to_global"] > r0["js_to_global"]


@check("config: smoke tier -> small_cnn + 2 rounds; overrides win")
def _():
    cfg = resolve_config(REPO / "configs" / "fedavg.yaml", tier="smoke",
                         overrides=["federated.rounds=7"])
    assert cfg["model"]["name"] == "small_cnn" and cfg["federated"]["rounds"] == 7


@check("DP accountant: q=1 analytic anchor eps ~ 4.75")
def _():
    from fl_med.privacy.accounting import compute_epsilon
    e = compute_epsilon(sample_rate=1.0, noise_multiplier=1.0, steps=1, delta=1e-5)
    assert abs(e - 4.75) < 0.15


@check("DP accountant: subsampling + more noise reduce epsilon; steps raise it")
def _():
    from fl_med.privacy.accounting import compute_epsilon
    full = compute_epsilon(sample_rate=1.0, noise_multiplier=1.0, steps=1, delta=1e-5)
    sub = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=1, delta=1e-5)
    lo = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=1000, delta=1e-5)
    hi_noise = compute_epsilon(sample_rate=0.01, noise_multiplier=4.0, steps=1000, delta=1e-5)
    fewer = compute_epsilon(sample_rate=0.01, noise_multiplier=1.0, steps=100, delta=1e-5)
    assert sub < full and hi_noise < lo and fewer < lo


def main() -> int:
    width = max(len(n) for n, _ in CHECKS)
    failures = 0
    for name, fn in CHECKS:
        try:
            fn()
            print(f"PASS  {name:<{width}}")
        except Exception as exc:  # noqa: BLE001
            failures += 1
            print(f"FAIL  {name:<{width}}  -> {exc!r}")
    print(f"\n{len(CHECKS) - failures}/{len(CHECKS)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
