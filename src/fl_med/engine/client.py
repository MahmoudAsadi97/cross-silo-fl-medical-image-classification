"""Federated client: runs local training for a given strategy and returns its update.

Decoupled from data acquisition -- it receives an already-built train loader, so
the same client works with the real dataset, the fixture, or toy tensors in tests.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, Optional

from ..strategies import scaffold as scaffold_math
from .train_eval import train_one_epoch


def clone_state(state_dict) -> "OrderedDict":
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


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
) -> Dict[str, Any]:
    """Train ``model`` locally and return an update dict for the server.

    For SCAFFOLD, pass ``c_global``/``c_local``; the returned dict then also holds
    the new local control variate and the (dy, dc) deltas.
    """
    import torch

    model.to(device)
    model.load_state_dict(global_state)

    opt_name = optimizer_cfg.get("name", "adam").lower()
    lr = float(optimizer_cfg.get("lr", 1e-3))
    wd = float(optimizer_cfg.get("weight_decay", 0.0))
    if opt_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=lr, momentum=optimizer_cfg.get("momentum", 0.0),
            weight_decay=wd,
        )
    else:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)

    client_state = {"c_global": c_global, "c_local": c_local} if c_global is not None else None

    history = []
    steps_per_epoch = max_batches if max_batches is not None else len(train_loader)
    for epoch in range(1, local_epochs + 1):
        m = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion=criterion, strategy=strategy, global_model=global_model,
            client_state=client_state, num_classes=num_classes, max_batches=max_batches,
        )
        m["local_epoch"] = epoch
        history.append(m)

    new_state = clone_state(model.state_dict())
    num_samples = len(train_loader.dataset)
    update: Dict[str, Any] = {
        "client_id": client_id,
        "num_samples": num_samples,
        "state_dict": new_state,
        "local_history": history,
    }

    # SCAFFOLD control-variate bookkeeping (uses the corrected option-II math).
    if c_global is not None and c_local is not None:
        num_steps = max(1, local_epochs * int(steps_per_epoch))
        new_c_local = scaffold_math.updated_client_control(
            c_local, c_global, global_state, new_state, num_steps=num_steps, lr=lr,
        )
        deltas = scaffold_math.client_deltas(c_local, new_c_local, global_state, new_state)
        update["c_local"] = new_c_local
        update["dy"] = deltas["dy"]
        update["dc"] = deltas["dc"]
    return update
