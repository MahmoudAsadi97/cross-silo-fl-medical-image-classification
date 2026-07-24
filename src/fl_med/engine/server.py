"""Federated server: the round loop, aggregation, drift diagnostics, evaluation.

One loop drives FedAvg / FedProx / SCAFFOLD; the strategy object decides what
differs. Per round we log global test metrics, a *client-drift* measure, and --
when DP is enabled -- per-client cumulative (epsilon, delta) from the independent
RDP accountant (giving the epsilon-vs-round curve; brief §4, sub-questions 6-7).
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

    return float(sum(float(torch.sum(v.float() ** 2).item()) for v in state.values())) ** 0.5


def _mean_update_deviation(updates: List[Dict[str, Any]], global_state) -> float:
    """Mean over clients of ``|| (w_i - w_global) - mean_j(w_j - w_global) ||_2``."""
    import torch

    keys = list(global_state.keys())
    deltas = [{k: (u["state_dict"][k].float() - global_state[k].float()) for k in keys}
              for u in updates]
    mean_delta = {k: sum(d[k] for d in deltas) / len(deltas) for k in keys}
    drifts = []
    for d in deltas:
        sq = sum(float(torch.sum((d[k] - mean_delta[k]) ** 2).item()) for k in keys)
        drifts.append(sq ** 0.5)
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
    """Run the full federated training loop and return a history dict."""
    import torch  # noqa: F401

    fl_cfg = config.get("federated", {}) or {}
    rounds = int(fl_cfg.get("rounds", 2))
    local_epochs = int(fl_cfg.get("local_epochs", 1))
    max_batches = fl_cfg.get("max_batches")
    optimizer_cfg = config.get("optimizer", {"name": "adam", "lr": 1e-3})
    num_classes = int(config.get("model", {}).get("num_classes", 8))
    server_lr = float(getattr(strategy, "server_lr", 1.0))
    is_scaffold = isinstance(strategy, Scaffold)

    privacy_cfg = config.get("privacy", {}) or {}
    dp_enabled = bool(privacy_cfg.get("enabled"))
    batch_size = int(config.get("data", {}).get("batch_size", 32))
    dp_delta = float(privacy_cfg.get("target_delta", 1e-5))
    dp_sigma = float(privacy_cfg.get("noise_multiplier", 1.0))
    cum_steps: Dict[int, int] = {cid: 0 for cid in client_ids}

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
                privacy_cfg=privacy_cfg if dp_enabled else None,
            )
            updates.append(update)

        drift = _mean_update_deviation(updates, global_state)

        weights = [u["num_samples"] for u in updates]
        if is_scaffold:
            out = scaffold_math.server_update(
                global_state, c_global,
                dy_list=[u["dy"] for u in updates], dc_list=[u["dc"] for u in updates],
                server_lr=server_lr, participation=len(client_ids) / max(len(client_ids), 1),
            )
            global_state, c_global = out["global_params"], out["c_global"]
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

        if dp_enabled:  # per-client cumulative (epsilon, delta) via independent accountant
            from ..privacy.accounting import compute_epsilon

            eps_by_client = {}
            for u in updates:
                cid = u["client_id"]
                cum_steps[cid] += int(u.get("dp_steps", 0))
                q_i = min(1.0, batch_size / max(u["num_samples"], 1))
                eps_by_client[cid] = compute_epsilon(
                    sample_rate=q_i, noise_multiplier=dp_sigma,
                    steps=cum_steps[cid], delta=dp_delta,
                )
            record["epsilon_max"] = max(eps_by_client.values())
            record["epsilon_mean"] = sum(eps_by_client.values()) / len(eps_by_client)
            record["delta"] = dp_delta

        history.append(record)
        if logger:
            extra = f" eps_max={record['epsilon_max']:.3f}" if dp_enabled else ""
            logger.info(
                "round %d | bal_acc=%.4f macro_f1=%.4f drift=%.4f loss=%.4f%s",
                rnd, record["test_balanced_accuracy"], record["test_macro_f1"],
                drift, record["test_loss"], extra,
            )

    return {
        "strategy": strategy.name,
        "history": history,
        "final_metrics": history[-1] if history else {},
        "global_state": global_state,
    }
