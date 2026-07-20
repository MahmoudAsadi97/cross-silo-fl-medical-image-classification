"""Federated server: the round loop, aggregation, drift diagnostics, evaluation.

One loop drives FedAvg / FedProx / SCAFFOLD; the strategy object decides what
differs. Per round we log global test metrics and a *client-drift* measure (mean
L2 norm of each client's update minus the mean update) so we can later test the
hypothesis that drift-correcting strategies actually reduce drift under skew.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List

from ..strategies import scaffold as scaffold_math
from ..strategies.scaffold import Scaffold, zeros_like_state
from .client import local_train
from .train_eval import evaluate


def _state_norm(state) -> float:
    import torch

    total = 0.0
    for v in state.values():
        total += float(torch.sum(v.float() ** 2).item())
    return total**0.5


def _mean_update_deviation(updates: List[Dict[str, Any]], global_state) -> float:
    """Mean over clients of ``|| (w_i - w_global) - mean_j(w_j - w_global) ||_2``."""
    import torch

    keys = list(global_state.keys())
    deltas = [
        {k: (u["state_dict"][k].float() - global_state[k].float()) for k in keys}
        for u in updates
    ]
    mean_delta = {k: sum(d[k] for d in deltas) / len(deltas) for k in keys}
    drifts = []
    for d in deltas:
        sq = sum(float(torch.sum((d[k] - mean_delta[k]) ** 2).item()) for k in keys)
        drifts.append(sq**0.5)
    return float(sum(drifts) / len(drifts))


def run_federated(
    *,
    config: Dict[str, Any],
    model_builder: Callable[[], Any],
    strategy,
    client_ids: List[int],
    client_loader_fn: Callable[[int], Any],
    test_loader,
    device: str = "cpu",
    criterion=None,
    logger=None,
) -> Dict[str, Any]:
    """Run the full federated training loop and return a history dict.

    ``client_loader_fn(client_id)`` returns that client's train loader.
    """
    import torch  # noqa: F401  (ensures torch present; used by helpers)

    fl_cfg = config.get("federated", {}) or {}
    rounds = int(fl_cfg.get("rounds", 2))
    local_epochs = int(fl_cfg.get("local_epochs", 1))
    max_batches = fl_cfg.get("max_batches")
    optimizer_cfg = config.get("optimizer", {"name": "adam", "lr": 1e-3})
    num_classes = int(config.get("model", {}).get("num_classes", 8))
    server_lr = float(getattr(strategy, "server_lr", 1.0))
    is_scaffold = isinstance(strategy, Scaffold)

    global_model = model_builder().to(device)
    global_state = OrderedDict(
        (k, v.detach().cpu().clone()) for k, v in global_model.state_dict().items()
    )

    c_global = zeros_like_state(global_state) if is_scaffold else None
    c_locals: Dict[int, Any] = (
        {cid: zeros_like_state(global_state) for cid in client_ids} if is_scaffold else {}
    )

    frozen_global = None
    history: List[Dict[str, Any]] = []

    for rnd in range(1, rounds + 1):
        if strategy.needs_global_model:
            frozen_global = model_builder().to(device)
            frozen_global.load_state_dict(global_state)
            frozen_global.eval()

        updates: List[Dict[str, Any]] = []
        for cid in client_ids:
            model = model_builder().to(device)
            update = local_train(
                client_id=cid, model=model, train_loader=client_loader_fn(cid),
                device=device, strategy=strategy, optimizer_cfg=optimizer_cfg,
                local_epochs=local_epochs, global_state=global_state,
                global_model=frozen_global,
                c_global=c_global, c_local=c_locals.get(cid),
                num_classes=num_classes, max_batches=max_batches, criterion=criterion,
            )
            updates.append(update)

        drift = _mean_update_deviation(updates, global_state)

        weights = [u["num_samples"] for u in updates]
        if is_scaffold:
            out = scaffold_math.server_update(
                global_state, c_global,
                dy_list=[u["dy"] for u in updates],
                dc_list=[u["dc"] for u in updates],
                server_lr=server_lr,
                participation=len(client_ids) / max(len(client_ids), 1),
            )
            global_state = out["global_params"]
            c_global = out["c_global"]
            for u in updates:
                c_locals[u["client_id"]] = u["c_local"]
        else:
            global_state = strategy.aggregate([u["state_dict"] for u in updates], weights)

        global_model.load_state_dict(global_state)
        test_metrics = evaluate(global_model, test_loader, device, num_classes=num_classes)
        record = {
            "round": rnd,
            "client_drift": drift,
            "global_weight_norm": _state_norm(global_state),
            "test_balanced_accuracy": test_metrics["balanced_accuracy"],
            "test_macro_f1": test_metrics["macro_f1"],
            "test_accuracy": test_metrics["accuracy"],
            "test_loss": test_metrics["loss"],
        }
        history.append(record)
        if logger:
            logger.info(
                "round %d | bal_acc=%.4f macro_f1=%.4f drift=%.4f loss=%.4f",
                rnd, record["test_balanced_accuracy"], record["test_macro_f1"],
                drift, record["test_loss"],
            )

    return {
        "strategy": strategy.name,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "global_state": global_state,
    }
