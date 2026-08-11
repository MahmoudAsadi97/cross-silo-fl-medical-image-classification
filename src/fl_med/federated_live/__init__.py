"""Real (networked) federated learning via Flower.

Turns the in-process simulation into a genuinely distributed system: each client
runs as its own process/machine (e.g. a Raspberry Pi acting as a hospital) and
only model updates cross the network. Same model/data/metrics as ``fl_med.engine``.
"""
from .task import (
    build_model,
    evaluate_model,
    get_ndarrays,
    local_fit,
    local_num_examples,
    set_ndarrays,
)

__all__ = [
    "build_model",
    "evaluate_model",
    "get_ndarrays",
    "local_fit",
    "local_num_examples",
    "set_ndarrays",
]
