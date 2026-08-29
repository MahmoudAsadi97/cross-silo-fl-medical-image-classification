"""Strategy interface.

The engine runs ONE identical training loop for every strategy; only the hooks
below differ. This is what makes the fair-comparison protocol structural rather
than aspirational -- FedAvg, FedProx and SCAFFOLD share the same client loop,
model init, data splits and compute budget, and diverge only where the algorithm
genuinely differs.

Hooks (all no-ops by default):

* ``extra_loss(model)``        -> scalar torch loss added to CE (FedProx prox term)
* ``after_backward(model)``    -> in-place grad edit before optimizer.step (SCAFFOLD)
* ``on_round_start(...)`` / ``on_client_end(...)`` -> per-round / per-client state
  bookkeeping (SCAFFOLD control variates)

``aggregate`` defaults to the sample-weighted mean (FedAvg); strategies that
change server behaviour override it.
"""
from __future__ import annotations

from typing import Any, Dict, Mapping, Sequence

from .aggregation import weighted_average


class Strategy:
    name: str = "base"
    #: whether the client loop must keep a frozen copy of the global model
    needs_global_model: bool = False

    def __init__(self, **kwargs: Any) -> None:
        self.hparams: Dict[str, Any] = dict(kwargs)

    # ---- client-side hooks -------------------------------------------------
    def extra_loss(self, model, global_model=None):
        """Additional loss term (torch scalar) or ``0.0``. Default: none."""
        return 0.0

    def after_backward(self, model, client_state: Dict[str, Any] | None = None) -> None:
        """In-place gradient modification before the optimizer step. Default: none."""
        return None

    # ---- server-side hook --------------------------------------------------
    def aggregate(
        self,
        client_states: Sequence[Mapping[str, Any]],
        weights: Sequence[float],
        global_state: Mapping[str, Any] | None = None,
    ):
        """Combine client parameter dicts into a new global dict.

        ``global_state`` (the pre-round global) is passed for server-optimizer strategies
        such as FedAdam, which need it to form a pseudo-gradient; the default mean ignores it.
        """
        return weighted_average(client_states, weights)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"{self.__class__.__name__}({self.hparams})"
