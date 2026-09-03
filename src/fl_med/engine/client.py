"""Federated client: runs local training for a given strategy and returns its update.

Decoupled from data acquisition -- it receives an already-built train loader, so
the same client works with the real dataset, the fixture, or toy tensors in tests.
Supports an optional local DP-SGD path (Opacus) selected by ``privacy_cfg``, using
Opacus's BatchMemoryManager so a large logical batch (good signal-to-noise + valid
accounting) is split into memory-safe physical micro-batches.
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
    """Train ``model`` locally and return an update dict for the server."""
    model.to(device)
    model.load_state_dict(global_state)
    num_samples = len(train_loader.dataset)

    dp_enabled = bool(privacy_cfg and privacy_cfg.get("enabled"))
    dp_engine = None
    dp_sample_rate = None
    if dp_enabled:
        from ..privacy.dp_engine import fix_model, make_private

        if max_batches is not None:
            raise ValueError(
                "DP-SGD requires a complete local DataLoader pass. "
                "Set federated.max_batches=null so Opacus sampling and privacy accounting agree."
            )

        model = fix_model(model).to(device)            # GroupNorm + no in-place ops
        optimizer = _build_optimizer(model, optimizer_cfg)
        model, optimizer, train_loader, dp_engine = make_private(
            model=model, optimizer=optimizer, data_loader=train_loader,
            noise_multiplier=float(privacy_cfg["noise_multiplier"]),
            max_grad_norm=float(privacy_cfg["max_grad_norm"]),
            accountant=str(privacy_cfg.get("accountant", "rdp")),
            secure_mode=bool(privacy_cfg.get("secure_mode", False)),
        )
        dp_sample_rate = float(
            getattr(train_loader, "sample_rate", 1.0 / max(len(train_loader), 1))
        )
        max_phys = int(privacy_cfg.get("max_physical_batch_size", 16))
    else:
        optimizer = _build_optimizer(model, optimizer_cfg)

    client_state = {"c_global": c_global, "c_local": c_local} if c_global is not None else None

    history = []
    for epoch in range(1, local_epochs + 1):
        if dp_enabled:
            from opacus.utils.batch_memory_manager import BatchMemoryManager

            with BatchMemoryManager(
                data_loader=train_loader, max_physical_batch_size=max_phys, optimizer=optimizer
            ) as mem_loader:
                m = train_one_epoch(
                    model, mem_loader, optimizer, device, criterion=criterion,
                    num_classes=num_classes, max_batches=None, grad_clip=None,
                )
        else:
            m = train_one_epoch(
                model, train_loader, optimizer, device, criterion=criterion,
                strategy=strategy, global_model=global_model, client_state=client_state,
                num_classes=num_classes, max_batches=max_batches, grad_clip=GRAD_CLIP_NORM,
            )
        m["local_epoch"] = epoch
        history.append(m)

    # Under DP the model is an Opacus GradSampleModule; extract the underlying module
    # so state-dict keys match the global model for aggregation.
    underlying = getattr(model, "_module", model)
    new_state = clone_state(underlying.state_dict())
    examples_seen = sum(int(epoch.get("num_examples", 0)) for epoch in history)
    dp_steps = 0
    if dp_enabled and dp_engine is not None:
        # Opacus records one entry per distinct (noise, sample-rate) segment. Reading
        # its history counts the logical optimizer steps that actually consumed budget.
        accountant_history = getattr(dp_engine.accountant, "history", ())
        dp_steps = sum(int(segment[2]) for segment in accountant_history)
    update: Dict[str, Any] = {
        "client_id": client_id,
        "num_samples": int(num_samples),
        "examples_seen": int(examples_seen),
        "state_dict": new_state,
        "local_history": history,
        "dp_steps": int(dp_steps),
    }
    if dp_enabled:
        update["dp_sample_rate"] = float(dp_sample_rate)

    if c_global is not None and c_local is not None:  # SCAFFOLD bookkeeping
        num_steps = max(
            1,
            sum(int(epoch.get("optimizer_steps", 0)) for epoch in history),
        )
        new_c_local = scaffold_math.updated_client_control(
            c_local, c_global, global_state, new_state,
            num_steps=num_steps, lr=float(optimizer_cfg.get("lr", 1e-3)),
        )
        deltas = scaffold_math.client_deltas(c_local, new_c_local, global_state, new_state)
        update["c_local"] = new_c_local
        update["dy"] = deltas["dy"]
        update["dc"] = deltas["dc"]
    return update
