"""Federated server: the round loop, aggregation, drift diagnostics, evaluation.

One loop drives FedAvg / FedProx / SCAFFOLD; the strategy object decides what
differs. Per round we log global test metrics, a *client-drift* measure, and --
when DP is enabled -- per-client cumulative (epsilon, delta) from the independent
RDP accountant (giving the epsilon-vs-round curve; brief §4, sub-questions 6-7).
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Callable, Dict, List, Optional

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
    event_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
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
    dp_delta = float(privacy_cfg.get("target_delta", 1e-5))
    dp_sigma = float(privacy_cfg.get("noise_multiplier", 1.0))
    cumulative_rdp: Dict[int, Any] = {cid: None for cid in client_ids}
    target_epsilon = privacy_cfg.get("target_epsilon")
    if dp_enabled:
        from ..privacy.accounting import DEFAULT_ORDERS, compute_rdp, get_privacy_spent

    def emit(event: str, **payload: Any) -> None:
        if event_callback is not None:
            event_callback({"event": event, **payload})

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
        emit("round_started", round=rnd, total_rounds=rounds)
        if strategy.needs_global_model:
            frozen_global = model_builder().to(device)
            frozen_global.load_state_dict(global_state)
            frozen_global.eval()

        updates: List[Dict[str, Any]] = []
        for cid in client_ids:
            model = model_builder().to(device)
            train_loader = client_loader_fn(cid)
            if dp_enabled and target_epsilon is not None:
                planned_steps = max(len(train_loader), 1) * local_epochs
                candidate_rdp = compute_rdp(
                    1.0 / max(len(train_loader), 1),
                    dp_sigma,
                    planned_steps,
                    DEFAULT_ORDERS,
                )
                previous = cumulative_rdp[cid]
                if previous is not None:
                    candidate_rdp = previous + candidate_rdp
                candidate_epsilon, _ = get_privacy_spent(
                    DEFAULT_ORDERS, candidate_rdp, dp_delta
                )
                tolerance = max(1e-6, float(target_epsilon) * 1e-6)
                if candidate_epsilon > float(target_epsilon) + tolerance:
                    raise RuntimeError(
                        "The next client update would exceed the frozen target epsilon"
                    )
            emit(
                "client_started",
                round=rnd,
                client_id=cid,
                num_samples=len(train_loader.dataset),
            )
            update = local_train(
                client_id=cid, model=model, train_loader=train_loader,
                device=device, strategy=strategy, optimizer_cfg=optimizer_cfg,
                local_epochs=local_epochs, global_state=global_state,
                global_model=frozen_global,
                c_global=c_global, c_local=c_locals.get(cid),
                num_classes=num_classes, max_batches=max_batches, criterion=criterion,
                privacy_cfg=privacy_cfg if dp_enabled else None,
            )
            updates.append(update)
            client_epsilon = None
            if dp_enabled:
                q_i = float(update["dp_sample_rate"])
                release_rdp = compute_rdp(
                    q_i,
                    dp_sigma,
                    int(update.get("dp_steps", 0)),
                    DEFAULT_ORDERS,
                )
                previous = cumulative_rdp[cid]
                cumulative_rdp[cid] = (
                    release_rdp if previous is None else previous + release_rdp
                )
                client_epsilon, _ = get_privacy_spent(
                    DEFAULT_ORDERS, cumulative_rdp[cid], dp_delta
                )
                if target_epsilon is not None:
                    tolerance = max(1e-6, float(target_epsilon) * 1e-6)
                    if client_epsilon > float(target_epsilon) + tolerance:
                        raise RuntimeError(
                            "Observed privacy spend exceeded the frozen target epsilon"
                        )
            latest_local = update.get("local_history", [{}])[-1]
            emit(
                "client_completed",
                round=rnd,
                client_id=cid,
                num_samples=update["num_samples"],
                examples_seen=update.get("examples_seen"),
                train_loss=latest_local.get("loss"),
                train_balanced_accuracy=latest_local.get("balanced_accuracy"),
                dp_steps=update.get("dp_steps", 0),
                dp_sample_rate=update.get("dp_sample_rate"),
                epsilon=client_epsilon,
                delta=dp_delta if dp_enabled else None,
            )

        drift = _mean_update_deviation(updates, global_state)
        emit("aggregation_started", round=rnd, participating_clients=len(updates))

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
            global_state = strategy.aggregate(
                [u["state_dict"] for u in updates], weights, global_state=global_state)

        global_model.load_state_dict(global_state)
        emit("evaluation_started", round=rnd, total_rounds=rounds)
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
            eps_by_client = {
                cid: get_privacy_spent(DEFAULT_ORDERS, cumulative_rdp[cid], dp_delta)[0]
                for cid in client_ids
            }
            record["epsilon_max"] = max(eps_by_client.values())
            record["epsilon_mean"] = sum(eps_by_client.values()) / len(eps_by_client)
            record["epsilon_by_client"] = {str(k): v for k, v in eps_by_client.items()}
            record["delta"] = dp_delta

        history.append(record)
        emit("round_completed", round=rnd, total_rounds=rounds, metrics=dict(record))
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
