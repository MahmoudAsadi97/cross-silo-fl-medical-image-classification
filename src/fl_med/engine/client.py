"""Federated client: runs local training for a given strategy and returns its update.

Decoupled from data acquisition -- it receives an already-built train loader, so
the same client works with the real dataset, the fixture, or toy tensors in tests.
Supports an optional local DP-SGD path (Opacus) selected by ``privacy_cfg``.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

from ..strategies import scaffold as scaffold_math
from .train_eval import GRAD_CLIP_NORM, train_one_epoch


def clone_state(state_dict) -> "OrderedDict":
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def _build_optimizer(model, optimizer_cfg):
    import torch

    name = optimizer_cfg.get("name", "adam").lower()
    lr = float(optimizer_cfg.get("lr", 1e-3))
    wd = float(optimizer_cfg.get("weight_decay", 0.0))
    if name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr,
                               momentum=optimizer_cfg.get("momentum", 0.0), weight_decay=wd)
    return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)


def local_train(
    *,
    client_id: int,
    model,
    train_loader,
    device,
    strategy,
    optimizer_cfg: Dict[str, Any],
    local_epochs: int,
    global_state: Dict[str, Any],
    global_model=None,
    c_global: Optional[Dict[str, Any]] = None,
    c_local: Optional[Dict[str, Any]] = None,
    num_classes: int = 8,
    max_batches: Optional[int] = None,
    criterion=None,
    privacy_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Train ``model`` locally and return an update dict for the server.

    For SCAFFOLD, pass ``c_global``/``c_local`` (returned dict then also holds the new
    control variate and (dy, dc) deltas). For DP, pass ``privacy_cfg`` with
    ``{enabled, noise_multiplier, max_grad_norm}``.
    """
    model.to(device)
    model.load_state_dict(global_state)
    optimizer = _build_optimizer(model, optimizer_cfg)

    dp_enabled = bool(privacy_cfg and privacy_cfg.get("enabled"))
    grad_clip = GRAD_CLIP_NORM
    if dp_enabled:
        from ..privacy.dp_engine import fix_model, make_private

        model = fix_model(model)                       # ensure DP-compatible norm (GroupNorm)
        model, optimizer, train_loader, _engine = make_private(
            model=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            max_grad_norm=float(privacy_cfg["max_grad_norm"]),
        )
        grad_clip = None                               # Opacus does per-sample clipping

    client_state = {"c_global": c_global, "c_local": c_local} if c_global is not None else None

    history = []
    steps_per_epoch = max_batches if max_batches is not None else len(train_loader)
    for epoch in range(1, local_epochs + 1):
        m = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion=criterion, strategy=strategy, global_model=global_model,
            client_state=client_state, num_classes=num_classes, max_batches=max_batches,
            grad_clip=grad_clip,
        )
        m["local_epoch"] = epoch
        history.append(m)

    # Under DP the model is an Opacus GradSampleModule; extract the underlying
    # module so state-dict keys match the global model for aggregation.
    underlying = getattr(model, "_module", model)
    new_state = clone_state(underlying.state_dict())
    num_samples = len(train_loader.dataset)
    update: Dict[str, Any] = {
        "client_id": client_id,
        "num_samples": num_samples,
        "state_dict": new_state,
        "local_history": history,
        "dp_steps": local_epochs * int(steps_per_epoch),
    }

    if c_global is not None and c_local is not None:  # SCAFFOLD bookkeeping
        num_steps = max(1, local_epochs * int(steps_per_epoch))
        new_c_local = scaffold_math.updated_client_control(
            c_local, c_global, global_state, new_state, num_steps=num_steps, lr=float(optimizer_cfg.get("lr", 1e-3)),
        )
        deltas = scaffold_math.client_deltas(c_local, new_c_local, global_state, new_state)
        update["c_local"] = new_c_local
        update["dy"] = deltas["dy"]
        update["dc"] = deltas["dc"]
    return update
