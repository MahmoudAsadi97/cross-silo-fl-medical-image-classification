"""Server-side aggregation math, written to be backend-agnostic.

Every function here operates on ``dict[str, Array]`` where ``Array`` supports
``+``, ``-`` and scalar ``*`` -- true for both ``numpy.ndarray`` and
``torch.Tensor``. That lets the exact same code be:

* unit-tested with numpy in a torch-free environment (see ``scripts/verify_core_math.py``),
* executed with torch state-dicts during real training.

This is deliberate: the correctness of federated aggregation is a *math* claim,
so it should be provable without a GPU stack.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Sequence, TypeVar

Array = TypeVar("Array")  # numpy.ndarray or torch.Tensor


def weighted_average(
    states: Sequence[Mapping[str, Array]],
    weights: Sequence[float],
) -> "OrderedDict[str, Array]":
    """Sample-weighted mean of a list of parameter dicts (the FedAvg update).

    ``aggregated[k] = sum_i (weights[i] / sum(weights)) * states[i][k]``.
    """
    if len(states) == 0:
        raise ValueError("states must not be empty")
    if len(states) != len(weights):
        raise ValueError("states and weights must have equal length")
    total = float(sum(weights))
    if total <= 0:
        raise ValueError("sum of weights must be positive")

    keys = list(states[0].keys())
    out: "OrderedDict[str, Array]" = OrderedDict()
    for key in keys:
        acc = None
        for state, weight in zip(states, weights):
            contribution = state[key] * (float(weight) / total)
            acc = contribution if acc is None else acc + contribution
        out[key] = acc
    return out


def state_delta(new_state: Mapping[str, Array], old_state: Mapping[str, Array]) -> Dict[str, Array]:
    """Element-wise ``new - old`` over matching keys."""
    return {k: new_state[k] - old_state[k] for k in new_state.keys()}


def apply_delta(
    state: Mapping[str, Array],
    delta: Mapping[str, Array],
    scale: float = 1.0,
) -> "OrderedDict[str, Array]":
    """Return ``state + scale * delta`` (used for server learning-rate updates)."""
    return OrderedDict((k, state[k] + delta[k] * float(scale)) for k in state.keys())


def mean_of_deltas(deltas: List[Mapping[str, Array]]) -> "OrderedDict[str, Array]":
    """Unweighted mean of a list of delta dicts (used for control-variate updates)."""
    if len(deltas) == 0:
        raise ValueError("deltas must not be empty")
    n = float(len(deltas))
    keys = list(deltas[0].keys())
    out: "OrderedDict[str, Array]" = OrderedDict()
    for key in keys:
        acc = None
        for delta in deltas:
            contribution = delta[key] * (1.0 / n)
            acc = contribution if acc is None else acc + contribution
        out[key] = acc
    return out
