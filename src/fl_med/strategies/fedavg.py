"""FedAvg (McMahan et al., 2017): sample-weighted averaging of client models."""
from __future__ import annotations

from .base import Strategy


class FedAvg(Strategy):
    name = "fedavg"
    # Inherits the default weighted-average aggregation and no-op client hooks;
    # FedAvg is the reference against which FedProx (mu=0) must be identical.
