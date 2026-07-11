"""FedProx (Li et al., 2020): FedAvg + a proximal term anchoring the local model.

Adds ``(mu/2) * ||w - w_global||^2`` to the local loss, which limits client drift
under heterogeneity. At ``mu = 0`` FedProx must reduce *exactly* to FedAvg -- a
regression guarded by ``tests/test_fedprox.py``.
"""
from __future__ import annotations

from typing import Iterable

from .base import Strategy


def proximal_term(model_params: "Iterable", global_params: "Iterable", mu: float):
    """``(mu/2) * sum ||w - w_global||^2`` as a torch scalar (0.0 when mu == 0)."""
    import torch

    if mu == 0:
        return 0.0
    total = None
    for w, w_g in zip(model_params, global_params):
        term = torch.sum((w - w_g.detach()) ** 2)
        total = term if total is None else total + term
    if total is None:
        return 0.0
    return 0.5 * mu * total


class FedProx(Strategy):
    name = "fedprox"
    needs_global_model = True

    def __init__(self, mu: float = 0.1, **kwargs) -> None:
        super().__init__(mu=mu, **kwargs)
        self.mu = float(mu)

    def extra_loss(self, model, global_model=None):
        if self.mu == 0 or global_model is None:
            return 0.0
        return proximal_term(model.parameters(), global_model.parameters(), self.mu)
