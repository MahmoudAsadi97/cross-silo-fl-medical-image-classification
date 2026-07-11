"""SCAFFOLD (Karimireddy et al., 2020), option-II control variates.

Cross-silo FL (few, persistent clients, full participation) is exactly where
SCAFFOLD's extra per-client state and doubled communication are affordable.

Notation (matches the paper):
    x    global model parameters at round start
    c    global control variate
    c_i  client i's local control variate (persistent across rounds)
    y_i  client i's local parameters after K local SGD steps
    K    number of local steps, eta_l the local learning rate

Local step (applied inside the optimizer loop):     g <- g - c_i + c
Client control update (option II):     c_i^+ = c_i - c + (x - y_i) / (K * eta_l)
Communicated:      dy_i = y_i - x ,   dc_i = c_i^+ - c_i
Server update:     x <- x + eta_g * mean_i(dy_i)
                   c <- c + (|S| / N) * mean_i(dc_i)

The math functions below are backend-agnostic (numpy or torch). The torch-only
gradient correction lives in ``correct_gradients`` behind a lazy import.

NOTE: the earlier prototype in ``archive/src/fl/scaffold.py`` used
``c_i^+ = c_i - c + (y_i - x)`` -- wrong sign *and* missing the 1/(K*eta_l)
scaling. This module fixes both; ``tests/test_scaffold.py`` locks the equations.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, TypeVar

from .aggregation import apply_delta, mean_of_deltas, state_delta

Array = TypeVar("Array")


def zeros_like_state(state: Mapping[str, Array]) -> "OrderedDict[str, Array]":
    """Zero control variates shaped like ``state`` (backend-agnostic via ``* 0``)."""
    return OrderedDict((k, v * 0) for k, v in state.items())


def updated_client_control(
    c_local: Mapping[str, Array],
    c_global: Mapping[str, Array],
    global_params: Mapping[str, Array],
    new_local_params: Mapping[str, Array],
    num_steps: int,
    lr: float,
) -> "OrderedDict[str, Array]":
    """Option-II update: ``c_i^+ = c_i - c + (x - y_i) / (K * eta_l)``."""
    if num_steps <= 0:
        raise ValueError("num_steps (K) must be positive")
    if lr <= 0:
        raise ValueError("lr (eta_l) must be positive")
    scale = 1.0 / (num_steps * lr)
    out: "OrderedDict[str, Array]" = OrderedDict()
    for k in c_local.keys():
        drift = (global_params[k] - new_local_params[k]) * scale
        out[k] = c_local[k] - c_global[k] + drift
    return out


def client_deltas(
    c_local_old: Mapping[str, Array],
    c_local_new: Mapping[str, Array],
    global_params: Mapping[str, Array],
    new_local_params: Mapping[str, Array],
) -> Dict[str, Dict[str, Array]]:
    """Return ``{'dy': y_i - x, 'dc': c_i^+ - c_i}`` communicated to the server."""
    return {
        "dy": state_delta(new_local_params, global_params),
        "dc": state_delta(c_local_new, c_local_old),
    }


def server_update(
    global_params: Mapping[str, Array],
    c_global: Mapping[str, Array],
    dy_list: List[Mapping[str, Array]],
    dc_list: List[Mapping[str, Array]],
    server_lr: float = 1.0,
    participation: float = 1.0,
) -> Dict[str, "OrderedDict[str, Array]"]:
    """Aggregate client deltas into new ``(global_params, c_global)``.

    ``participation = |S| / N`` (1.0 under full participation, the cross-silo case).
    """
    new_global = apply_delta(global_params, mean_of_deltas(dy_list), scale=server_lr)
    new_c = apply_delta(c_global, mean_of_deltas(dc_list), scale=participation)
    return {"global_params": new_global, "c_global": new_c}


def correct_gradients(model, c_local: Mapping[str, Array], c_global: Mapping[str, Array]) -> None:
    """Apply ``g <- g - c_i + c`` in-place to a torch model's ``.grad`` fields.

    Called after ``loss.backward()`` and before ``optimizer.step()``. Keys are
    matched by ``named_parameters`` so control variates are stored per parameter.
    """
    import torch

    with torch.no_grad():
        for name, param in model.named_parameters():
            if param.grad is None or name not in c_local:
                continue
            device = param.grad.device
            param.grad.add_(c_global[name].to(device) - c_local[name].to(device))


# ---------------------------------------------------------------------------
# Strategy wrapper
# ---------------------------------------------------------------------------
from .base import Strategy  # noqa: E402  (kept below the pure math on purpose)


class Scaffold(Strategy):
    """SCAFFOLD strategy. Control-variate lifecycle is driven by the engine, which
    holds ``c_global`` and per-client ``c_local`` and calls the module functions
    above; this class supplies the per-step gradient correction and metadata.
    """

    name = "scaffold"
    needs_global_model = True

    def __init__(self, server_lr: float = 1.0, **kwargs) -> None:
        super().__init__(server_lr=server_lr, **kwargs)
        self.server_lr = float(server_lr)

    def after_backward(self, model, client_state=None) -> None:
        if not client_state:
            return
        c_local = client_state.get("c_local")
        c_global = client_state.get("c_global")
        if c_local is not None and c_global is not None:
            correct_gradients(model, c_local, c_global)

