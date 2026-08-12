"""Byzantine-robust aggregation + model-poisoning attacks -- the INTEGRITY side of security.

Confidentiality is covered elsewhere (DP-SGD, membership inference, secure aggregation).
This module covers the other half: a *malicious* client can send poisoned updates to
corrupt the shared model. Plain FedAvg (a mean) has breakdown point 0 -- a single
crafted update can move the average arbitrarily. Robust aggregators bound each client's
influence:

* ``coordinate_median`` -- per-parameter median across clients (Yin et al., 2018).
* ``trimmed_mean``      -- per-parameter mean after dropping the ``trim`` highest & lowest
                          values (Yin et al., 2018).
* ``krum`` / multi-Krum -- pick the update(s) closest to their neighbours, i.e. most
                          "consistent" with the honest majority (Blanchard et al., 2017).

Every function takes a list of state-dicts (``name -> numpy array``) and returns one
state-dict, so they are drop-in replacements for ``strategies.aggregation.weighted_average``
and are verifiable in pure numpy (see ``scripts/verify_core_math.py``).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import List, Mapping

import numpy as np


def _stack(states: List[Mapping[str, np.ndarray]], key: str) -> np.ndarray:
    """Stack one parameter across clients -> array of shape (n_clients, *param_shape)."""
    return np.stack([np.asarray(s[key], dtype=np.float64) for s in states], axis=0)


# ---- robust aggregators ------------------------------------------------------
def coordinate_median(states: List[Mapping[str, np.ndarray]]) -> "OrderedDict[str, np.ndarray]":
    """Element-wise median across clients for each parameter. Tolerates < 50% malicious."""
    if not states:
        raise ValueError("states must not be empty")
    return OrderedDict((k, np.median(_stack(states, k), axis=0)) for k in states[0].keys())


def trimmed_mean(states: List[Mapping[str, np.ndarray]], trim: int = 1) -> "OrderedDict[str, np.ndarray]":
    """Element-wise mean after removing the ``trim`` largest and ``trim`` smallest values
    per coordinate. Needs ``2*trim < n_clients``. Robust to up to ``trim`` malicious clients."""
    n = len(states)
    if n == 0:
        raise ValueError("states must not be empty")
    if 2 * trim >= n:
        raise ValueError(f"trim={trim} too large for {n} clients (need 2*trim < n)")
    out: "OrderedDict[str, np.ndarray]" = OrderedDict()
    for k in states[0].keys():
        srt = np.sort(_stack(states, k), axis=0)     # sort along the client axis
        out[k] = srt[trim:n - trim].mean(axis=0)     # drop extremes, average the rest
    return out


def _flatten(state: Mapping[str, np.ndarray]) -> np.ndarray:
    """Flatten a whole state-dict into one 1-D float32 vector (for distance math)."""
    return np.concatenate([np.asarray(v, dtype=np.float32).ravel() for v in state.values()])


def krum(states: List[Mapping[str, np.ndarray]], num_malicious: int = 1,
         multi: int = 1) -> "OrderedDict[str, np.ndarray]":
    """Krum / multi-Krum (Blanchard et al., 2017).

    Score each client by the sum of squared distances to its ``n - f - 2`` closest
    neighbours; the update(s) with the smallest score are the most consistent with the
    honest majority and are returned (``multi`` > 1 averages the best ``multi`` -- multi-Krum).
    Distances use the Gram-matrix identity so memory stays O(n^2 + n*D), not O(n^2 * D).
    """
    n = len(states)
    if n == 0:
        raise ValueError("states must not be empty")
    f = int(num_malicious)
    vecs = np.stack([_flatten(s) for s in states], axis=0)            # (n, D) float32
    norms = np.sum(vecs.astype(np.float64) ** 2, axis=1)             # (n,)
    gram = vecs.astype(np.float64) @ vecs.astype(np.float64).T       # (n, n)
    d2 = norms[:, None] + norms[None, :] - 2.0 * gram                # squared distances
    np.fill_diagonal(d2, 0.0)
    m = max(1, n - f - 2)
    scores = np.array([np.sort(d2[i])[1:m + 1].sum() for i in range(n)])  # exclude self (idx 0)
    chosen = np.argsort(scores)[:max(1, multi)]
    keys = list(states[0].keys())
    return OrderedDict(
        (k, np.mean([np.asarray(states[c][k], dtype=np.float64) for c in chosen], axis=0))
        for k in keys
    )


ROBUST_AGGREGATORS = {
    "median": coordinate_median,
    "trimmed_mean": trimmed_mean,
    "krum": krum,
}


# ---- model-poisoning attacks -------------------------------------------------
def poison_sign_flip(local: Mapping[str, np.ndarray], global_state: Mapping[str, np.ndarray],
                     scale: float = 5.0) -> "OrderedDict[str, np.ndarray]":
    """Push AWAY from the honest direction, amplified: ``global - scale*(local - global)``.
    A strong, simple model-poisoning attack that wrecks a plain FedAvg mean."""
    return OrderedDict(
        (k, np.asarray(global_state[k], np.float64)
            - float(scale) * (np.asarray(local[k], np.float64) - np.asarray(global_state[k], np.float64)))
        for k in local.keys()
    )


def poison_scale(local: Mapping[str, np.ndarray], global_state: Mapping[str, np.ndarray],
                 boost: float = 10.0) -> "OrderedDict[str, np.ndarray]":
    """Scaling / model-replacement: amplify the honest delta so it dominates the average:
    ``global + boost*(local - global)``."""
    return OrderedDict(
        (k, np.asarray(global_state[k], np.float64)
            + float(boost) * (np.asarray(local[k], np.float64) - np.asarray(global_state[k], np.float64)))
        for k in local.keys()
    )


POISON_ATTACKS = {
    "sign_flip": poison_sign_flip,
    "scale": poison_scale,
}
