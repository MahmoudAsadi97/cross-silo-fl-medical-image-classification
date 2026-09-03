"""Quantifying non-IID label skew across clients (brief §1 sub-question 1, Phase 1).

Everything operates on a client x class *count matrix* and is pure numpy, so it
runs without torch on the real metadata and is unit-testable. Measures:

* per-client Shannon entropy (nats) and normalized entropy in [0, 1]
* number of missing classes per client
* divergence of each client's label distribution from the global pool:
  KL(client || global), Jensen-Shannon distance, Hellinger distance
* 1-D Earth-Mover / Wasserstein distance over the (ordinal) class index

The JS and Hellinger distances are true metrics in [0, 1]; KL is asymmetric and
reported in nats. EMD over class *index* treats labels as ordered -- a documented
proxy, useful as a single scalar of "how far" a client is from the pool.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np

_EPS = 1e-12


def normalize(counts: np.ndarray) -> np.ndarray:
    counts = np.asarray(counts, dtype=float)
    total = counts.sum(axis=-1, keepdims=True)
    total = np.where(total == 0, 1.0, total)
    return counts / total


def shannon_entropy(p: np.ndarray) -> float:
    p = np.asarray(p, dtype=float)
    p = p[p > 0]
    return float(-np.sum(p * np.log(p)))


def normalized_entropy(p: np.ndarray) -> float:
    """Entropy / log(K): 1.0 == uniform over all K classes, 0.0 == one class."""
    k = len(p)
    if k <= 1:
        return 0.0
    return shannon_entropy(p) / np.log(k)


def kl_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) in nats, with epsilon smoothing on q."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float) + _EPS
    q = q / q.sum()
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def js_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon distance (sqrt of JS divergence, base-2): a metric in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    m = 0.5 * (p + q)

    def _kl2(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / (b[mask] + _EPS)))

    js_div = 0.5 * _kl2(p, m) + 0.5 * _kl2(q, m)
    return float(np.sqrt(max(js_div, 0.0)))


def hellinger_distance(p: np.ndarray, q: np.ndarray) -> float:
    """Hellinger distance: a metric in [0, 1]."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sqrt(0.5 * np.sum((np.sqrt(p) - np.sqrt(q)) ** 2)))


def emd_1d(p: np.ndarray, q: np.ndarray) -> float:
    """1-D Earth-Mover distance over the ordinal class index (sum of |CDF diff|)."""
    p = np.asarray(p, dtype=float)
    q = np.asarray(q, dtype=float)
    return float(np.sum(np.abs(np.cumsum(p) - np.cumsum(q))))


def heterogeneity_report(
    counts: np.ndarray,
    client_ids: Optional[Sequence[int]] = None,
    class_names: Optional[Sequence[str]] = None,
) -> Dict[str, object]:
    """Full per-client heterogeneity table from a client x class count matrix.

    Returns a dict with a per-client list of records and the global distribution.
    """
    counts = np.asarray(counts, dtype=float)
    n_clients, n_classes = counts.shape
    client_ids = list(client_ids) if client_ids is not None else list(range(n_clients))

    global_counts = counts.sum(axis=0)
    global_dist = normalize(global_counts.reshape(1, -1))[0]
    client_dist = normalize(counts)

    records: List[Dict[str, object]] = []
    for i in range(n_clients):
        p = client_dist[i]
        records.append(
            {
                "client_id": client_ids[i],
                "num_samples": int(counts[i].sum()),
                "num_classes_present": int(np.sum(counts[i] > 0)),
                "num_classes_missing": int(np.sum(counts[i] == 0)),
                "entropy_nats": shannon_entropy(p),
                "normalized_entropy": normalized_entropy(p),
                "kl_to_global": kl_divergence(p, global_dist),
                "js_to_global": js_distance(p, global_dist),
                "hellinger_to_global": hellinger_distance(p, global_dist),
                "emd_to_global": emd_1d(p, global_dist),
            }
        )

    return {
        "num_clients": n_clients,
        "num_classes": n_classes,
        "class_names": list(class_names) if class_names else None,
        "global_distribution": global_dist.tolist(),
        "global_entropy_nats": shannon_entropy(global_dist),
        "per_client": records,
    }


def counts_from_dataset(root_dir, num_classes: int = 8, num_clients: int = 6) -> np.ndarray:
    """Scan a split directory into a client x class count matrix (no torch needed)."""
    from .dataset import ISICFederatedFolderDataset

    ds = ISICFederatedFolderDataset(root_dir)
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for _, class_id, client_id in ds.samples:
        counts[client_id, class_id] += 1
    return counts
