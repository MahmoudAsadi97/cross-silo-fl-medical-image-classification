"""Opacus DP-SGD engine wrapper (torch; imported lazily).

Local (per-client) DP-SGD: clip each per-sample gradient to ``max_grad_norm`` and
add Gaussian noise scaled by ``noise_multiplier``, giving SAMPLE-level (record-level)
DP for that client's own training. This is NOT client-level DP (which would need
noise on the aggregated server update; see notes in docs/design.md). Opacus cannot take
per-sample gradients through BatchNorm, so the model must be GroupNorm --
``fix_model`` converts any stray BN just in case (brief §4.1, §4.2).
"""
from __future__ import annotations

from typing import Any, Tuple


def fix_model(model):
    """Replace any BatchNorm with a DP-compatible norm (GroupNorm) if needed."""
    from opacus.validators import ModuleValidator

    if not ModuleValidator.is_valid(model):
        model = ModuleValidator.fix(model)
    return model


def make_private(
    *, model, optimizer, data_loader, noise_multiplier: float, max_grad_norm: float,
) -> Tuple[Any, Any, Any, Any]:
    """Attach an Opacus ``PrivacyEngine``. Returns (model, optimizer, loader, engine).

    The returned data_loader uses Poisson subsampling (required for the accounting
    to be valid), so its per-batch size varies around the nominal batch size.
    """
    from opacus import PrivacyEngine

    engine = PrivacyEngine()
    model, optimizer, data_loader = engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=data_loader,
        noise_multiplier=float(noise_multiplier),
        max_grad_norm=float(max_grad_norm),
        poisson_sampling=True,
    )
    return model, optimizer, data_loader, engine


def opacus_epsilon(engine, delta: float) -> float:
    """Epsilon reported by Opacus's own accountant (cross-checked vs accounting.py)."""
    try:
        return float(engine.get_epsilon(delta))
    except Exception:
        return float(engine.accountant.get_epsilon(delta=delta))
