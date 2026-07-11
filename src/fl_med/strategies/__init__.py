"""Federated strategies + a small registry/factory.

Import policy: only pure-python modules are imported at package load, so
``import fl_med.strategies`` works without torch. The Strategy *classes* only
touch torch lazily inside their hooks.
"""
from __future__ import annotations

from typing import Any, Dict

from .aggregation import (
    apply_delta,
    mean_of_deltas,
    state_delta,
    weighted_average,
)
from .base import Strategy
from .fedavg import FedAvg
from .fedprox import FedProx
from .scaffold import Scaffold

REGISTRY = {
    FedAvg.name: FedAvg,
    FedProx.name: FedProx,
    Scaffold.name: Scaffold,
}


def build_strategy(config: Dict[str, Any]) -> Strategy:
    """Instantiate a strategy from a config's ``strategy`` block.

    Expected shape::

        strategy:
          name: fedprox
          mu: 0.1
    """
    strat_cfg = dict(config.get("strategy", {}) or {})
    name = strat_cfg.pop("name", "fedavg")
    if name not in REGISTRY:
        raise KeyError(f"Unknown strategy '{name}'. Known: {sorted(REGISTRY)}")
    return REGISTRY[name](**strat_cfg)


__all__ = [
    "Strategy",
    "FedAvg",
    "FedProx",
    "Scaffold",
    "REGISTRY",
    "build_strategy",
    "weighted_average",
    "state_delta",
    "apply_delta",
    "mean_of_deltas",
]
