"""Independent RDP privacy accountant for the subsampled Gaussian mechanism.

Pure numpy (no torch/opacus), so it runs anywhere and serves as the INDEPENDENT
cross-check that Opacus's reported (epsilon, delta) is correct (brief §5). Uses the
standard Renyi-DP analysis (Mironov 2017; Mironov, Talwar, Zhang 2019) over integer
orders, with the tight RDP->DP conversion (Canonne, Kamath, Steinke 2020) that
recent Opacus also uses. Integer orders give a valid upper bound on epsilon.

Reference behaviour (verified in scripts/verify_core_math.py):
- q=1 reduces to the analytic Gaussian: rdp(alpha) = steps * alpha / (2 sigma^2).
- epsilon decreases as sigma grows, increases with steps, and subsampling (q<1)
  lowers epsilon vs q=1 (privacy amplification).
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence, Tuple

import numpy as np

DEFAULT_ORDERS = list(range(2, 257))


def _log_add(logx: float, logy: float) -> float:
    if logx == -math.inf:
        return logy
    if logy == -math.inf:
        return logx
    a, b = min(logx, logy), max(logx, logy)
    return math.log1p(math.exp(a - b)) + b


def _log_binom(n: int, k: int) -> float:
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def _rdp_gaussian_int(q: float, sigma: float, alpha: int) -> float:
    """RDP of the subsampled Gaussian at integer order ``alpha`` (>= 2)."""
    if q == 0.0:
        return 0.0
    if sigma == 0.0:
        return math.inf
    if q == 1.0:  # no subsampling: exact analytic value
        return alpha / (2.0 * sigma**2)
    log_a = -math.inf
    for i in range(alpha + 1):
        log_coef = _log_binom(alpha, i) + i * math.log(q) + (alpha - i) * math.log1p(-q)
        term = log_coef + (i * i - i) / (2.0 * sigma**2)
        log_a = _log_add(log_a, term)
    return log_a / (alpha - 1)


def compute_rdp(q: float, noise_multiplier: float, steps: int,
                orders: Sequence[int] = tuple(DEFAULT_ORDERS)) -> np.ndarray:
    """RDP at each order after ``steps`` subsampled-Gaussian steps."""
    if not 0.0 <= q <= 1.0:
        raise ValueError("sample rate q must be between 0 and 1")
    if noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be positive")
    if isinstance(steps, bool) or int(steps) != steps or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    if any(int(order) != order or order < 2 for order in orders):
        raise ValueError("RDP orders must be integers greater than or equal to 2")
    return np.array([_rdp_gaussian_int(q, noise_multiplier, int(a)) for a in orders]) * steps


def get_privacy_spent(orders: Sequence[int], rdp: np.ndarray,
                      target_delta: float) -> Tuple[float, int]:
    """Convert RDP -> (epsilon, best order) at ``target_delta`` (tight conversion)."""
    orders = np.asarray(orders, dtype=float)
    rdp = np.asarray(rdp, dtype=float)
    eps = rdp + np.log1p(-1.0 / orders) - (math.log(target_delta) + np.log(orders)) / (orders - 1.0)
    idx = int(np.nanargmin(eps))
    return float(eps[idx]), int(orders[idx])


def compute_epsilon(
    *, sample_rate: float, noise_multiplier: float, steps: int, delta: float,
    orders: Iterable[int] = tuple(DEFAULT_ORDERS),
) -> float:
    """Convenience: epsilon for DP-SGD with the given (q, sigma, steps, delta)."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be between 0 and 1")
    if steps == 0 or sample_rate == 0.0:
        return 0.0
    orders = list(orders)
    rdp = compute_rdp(sample_rate, noise_multiplier, steps, orders)
    eps, _ = get_privacy_spent(orders, rdp, delta)
    return eps
