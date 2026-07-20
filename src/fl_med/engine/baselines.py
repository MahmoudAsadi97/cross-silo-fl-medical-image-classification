"""Centralized and local-only baselines (Phase 2).

* Centralized  = all clients' data pooled -> practical upper bound.
* Local-only   = each client trains alone, evaluated on the pooled test set
                 -> lower reference (no collaboration).

Expected sanity ordering on balanced accuracy: majority < local-only < FL <
centralized. ``run_experiment`` checks this once results exist.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

from .train_eval import evaluate, train_one_epoch


def train_supervised(
    *,
    model,
    train_loader,
    test_loader,
    device,
    epochs: int,
    optimizer_cfg: Dict[str, Any],
    num_classes: int = 8,
    max_batches: Optional[int] = None,
    criterion=None,
    logger=None,
    tag: str = "train",
) -> Dict[str, Any]:
    """Plain supervised training with per-epoch eval on the test set."""
    import torch

    model.to(device)
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

    history: List[Dict[str, Any]] = []
    for epoch in range(1, epochs + 1):
        train_m = train_one_epoch(
            model, train_loader, optimizer, device,
            criterion=criterion, num_classes=num_classes, max_batches=max_batches,
        )
        test_m = evaluate(model, test_loader, device, num_classes=num_classes)
        record = {
            "epoch": epoch,
            "train_loss": train_m["loss"],
            "train_balanced_accuracy": train_m["balanced_accuracy"],
            "test_balanced_accuracy": test_m["balanced_accuracy"],
            "test_macro_f1": test_m["macro_f1"],
            "test_accuracy": test_m["accuracy"],
            "test_loss": test_m["loss"],
        }
        history.append(record)
        if logger:
            logger.info(
                "[%s] epoch %d | train_loss=%.4f test_bal_acc=%.4f test_f1=%.4f",
                tag, epoch, train_m["loss"], test_m["balanced_accuracy"],
                test_m["macro_f1"],
            )
    return {"history": history, "final_metrics": history[-1] if history else {}}


def run_centralized(
    *, config, model_builder: Callable[[], Any], train_loader, test_loader,
    device="cpu", criterion=None, logger=None,
) -> Dict[str, Any]:
    tr = config.get("training", {}) or {}
    return train_supervised(
        model=model_builder(), train_loader=train_loader, test_loader=test_loader,
        device=device, epochs=int(tr.get("epochs", 1)),
        optimizer_cfg=config.get("optimizer", {"name": "adam", "lr": 1e-3}),
        num_classes=int(config.get("model", {}).get("num_classes", 8)),
        max_batches=tr.get("max_batches"), criterion=criterion, logger=logger, tag="centralized",
    )


def run_local_only(
    *, config, model_builder, client_ids, client_loader_fn, test_loader,
    device="cpu", criterion=None, logger=None,
) -> Dict[str, Any]:
    """Train one model per client independently; evaluate each on the pooled test set."""
    tr = config.get("training", {}) or {}
    per_client: Dict[int, Any] = {}
    for cid in client_ids:
        train_loader, _ = client_loader_fn(cid)
        res = train_supervised(
            model=model_builder(), train_loader=train_loader, test_loader=test_loader,
            device=device, epochs=int(tr.get("epochs", 1)),
            optimizer_cfg=config.get("optimizer", {"name": "adam", "lr": 1e-3}),
            num_classes=int(config.get("model", {}).get("num_classes", 8)),
            max_batches=tr.get("max_batches"), criterion=criterion,
            logger=logger, tag=f"local/client_{cid}",
        )
        per_client[cid] = res["final_metrics"]
    mean_bal = sum(m.get("test_balanced_accuracy", 0.0) for m in per_client.values()) / max(
        len(per_client), 1
    )
    return {"per_client": per_client, "mean_test_balanced_accuracy": mean_bal}
