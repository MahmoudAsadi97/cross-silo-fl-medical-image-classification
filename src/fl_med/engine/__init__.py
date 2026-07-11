"""Engine: shared train/eval, federated client + server, and baselines."""
from __future__ import annotations

from .baselines import run_centralized, run_local_only, train_supervised
from .client import local_train
from .server import run_federated
from .train_eval import evaluate, train_one_epoch

__all__ = [
    "train_one_epoch",
    "evaluate",
    "local_train",
    "run_federated",
    "run_centralized",
    "run_local_only",
    "train_supervised",
]
