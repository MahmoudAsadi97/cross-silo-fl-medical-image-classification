"""Additive pairwise-mask secure aggregation (simulation).

Each pair of clients (i, j) shares a secret random mask m_ij (in practice derived
from a Diffie-Hellman shared key; here from a shared seed). Client i adds +m_ij for
every j>i and -m_ij for every j<i to its update. Because the masks are antisymmetric,
they cancel exactly when the server sums all masked updates -- so the server recovers
the true aggregate WITHOUT ever seeing any individual client's update. This is the
core of Bonawitz et al. 2017 (contract reference [1]); dropout recovery / DH key
agreement are omitted as they are orthogonal to the correctness we demonstrate.

Backend-agnostic (numpy or torch), so it verifies without a GPU.

Guarantees demonstrated (see tests + scripts/run_secure_agg.py):
* the unmasked sum equals the plaintext FedAvg aggregate within float tolerance;
* a single masked update is uninformative (far from the true update).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List, Mapping, Sequence

import numpy as np


def _pair_seed(a: int, b: int, key_idx: int, master_seed: int) -> int:
    """Symmetric per-(pair, tensor) seed: identical for clients a and b."""
    lo, hi = (a, b) if a < b else (b, a)
    return (int(master_seed) * 1_000_003 + lo * 97_003 + hi * 389 + key_idx * 7) % (2**32)


def client_mask(
    template: Mapping[str, np.ndarray], client_id: int, client_ids: Sequence[int],
    master_seed: int = 0, scale: float = 1.0,
) -> "OrderedDict[str, np.ndarray]":
    """Sum of antisymmetric pairwise masks for one client (shaped like ``template``)."""
    keys = list(template.keys())
    out = OrderedDict((k, np.zeros_like(np.asarray(template[k], dtype=float))) for k in keys)
    for j in client_ids:
        if j == client_id:
            continue
        sign = 1.0 if client_id < j else -1.0
        for idx, k in enumerate(keys):
            rng = np.random.default_rng(_pair_seed(client_id, j, idx, master_seed))
            out[k] = out[k] + sign * scale * rng.standard_normal(size=np.asarray(template[k]).shape)
    return out


def mask_update(
    update: Mapping[str, np.ndarray], client_id: int, client_ids: Sequence[int],
    master_seed: int = 0, scale: float = 1.0,
) -> "OrderedDict[str, np.ndarray]":
    """Return the client's update with its pairwise masks added."""
    mask = client_mask(update, client_id, client_ids, master_seed, scale)
    return OrderedDict((k, np.asarray(update[k], dtype=float) + mask[k]) for k in update)


def secure_sum(masked_updates: List[Mapping[str, np.ndarray]]) -> "OrderedDict[str, np.ndarray]":
    """Server-side sum of masked updates (masks cancel -> true sum)."""
    keys = list(masked_updates[0].keys())
    return OrderedDict((k, sum(mu[k] for mu in masked_updates)) for k in keys)


def secure_weighted_average(
    updates: Sequence[Mapping[str, np.ndarray]], weights: Sequence[float],
    master_seed: int = 0, scale: float = 1.0,
) -> Dict[str, "object"]:
    """Weighted FedAvg via secure aggregation.

    Each client masks ``(weight_i / sum_w) * update_i``; the server sums the masked
    contributions and the masks cancel, yielding the exact weighted average without
    seeing any individual update. Returns the aggregate + the masked updates (so a
    caller can check they are uninformative).
    """
    ids = list(range(len(updates)))
    total_w = float(sum(weights))
    scaled = [OrderedDict((k, np.asarray(u[k], dtype=float) * (w / total_w)) for k in u)
              for u, w in zip(updates, weights)]
    masked = [mask_update(scaled[i], i, ids, master_seed, scale) for i in ids]
    aggregate = secure_sum(masked)
    return {"aggregate": aggregate, "masked_updates": masked, "scaled_contributions": scaled}
