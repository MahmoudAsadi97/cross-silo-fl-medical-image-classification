#!/usr/bin/env python3
"""Single entry point for every experiment.

    python scripts/run_experiment.py --config configs/fedavg.yaml --tier smoke
    python scripts/run_experiment.py --config configs/fedprox.yaml --tier full \
        --seed 0 --output experiments/fedprox_full_seed0 training.local_epochs=2

Dispatch is by the config's ``task`` field (centralized / local_only / federated).
Each run writes ``run_config.yaml`` (provenance), ``metrics.csv``, ``summary.json``
and curve plots into its output directory so every artifact is traceable.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med import CLASS_NAMES  # noqa: E402,F401
from fl_med.config import resolve_config  # noqa: E402
from fl_med.eval import plot_curves, save_history_csv, save_json  # noqa: E402
from fl_med.logging import get_logger, write_run_manifest  # noqa: E402
from fl_med.seeding import set_seed  # noqa: E402


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Run an fl_med experiment")
    p.add_argument("--config", required=True, help="Path to experiment YAML")
    p.add_argument("--tier", default="smoke", choices=["smoke", "dev", "full"])
    p.add_argument("--seed", type=int, default=None, help="Override the single seed")
    p.add_argument("--output", default=None, help="Output dir (default experiments/<name>)")
    p.add_argument("--device", default="cpu")
    p.add_argument(
        "--status-file",
        default=None,
        help="Optional atomic JSON status feed for the local interactive demo",
    )
    p.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    return p.parse_args(argv)


def _json_safe(value):
    """Convert metric payloads to strict JSON without leaking implementation objects."""
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if hasattr(value, "item"):
        return _json_safe(value.item())
    return value


def _atomic_status(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(_json_safe(payload), indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def _default_output(config, tier, seed) -> Path:
    name = config.get("experiment", Path(config["_meta"]["experiment_file"]).stem)
    return REPO / "experiments" / f"{name}_{tier}_seed{seed}"


def main(argv=None) -> int:
    args = parse_args(argv)
    overrides = list(args.overrides)
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")

    config = resolve_config(args.config, tier=args.tier, overrides=overrides)
    seed = int(config.get("seed", 42))
    output_dir = Path(args.output) if args.output else _default_output(config, args.tier, seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("run_experiment", log_file=output_dir / "run.log")
    set_seed(seed)
    write_run_manifest(output_dir, config, extra={"device": args.device})

    task = config.get("task", "federated")
    logger.info("task=%s tier=%s seed=%d output=%s", task, args.tier, seed, output_dir)

    status_path = Path(args.status_file).resolve() if args.status_file else None
    status = {
        "schema_version": 1,
        "run_id": os.environ.get("FL_DEMO_RUN_ID"),
        "mode": "experiment",
        "dataset_kind": os.environ.get("FL_DEMO_DATASET_KIND", "unknown"),
        "status": "preflight",
        "phase": "preflight",
        "round": 0,
        "completed_rounds": 0,
        "total_rounds": int((config.get("federated") or {}).get("rounds", 0)),
        "strategy": str((config.get("strategy") or {}).get("name", task)),
        "history": [],
        "clients": [],
        "events": [],
        "privacy": {
            "enabled": bool((config.get("privacy") or {}).get("enabled", False)),
            "noise_multiplier": (config.get("privacy") or {}).get("noise_multiplier"),
            "max_grad_norm": (config.get("privacy") or {}).get("max_grad_norm"),
            "delta": (config.get("privacy") or {}).get("target_delta"),
            "target_epsilon": (config.get("privacy") or {}).get("target_epsilon"),
            "accountant": (config.get("privacy") or {}).get("accountant", "rdp"),
            "scope": "record_level_dp_sgd_per_center" if (config.get("privacy") or {}).get("enabled") else None,
        },
    }
    client_status: dict[int, dict] = {}
    event_seq = 0

    def publish_status() -> None:
        if status_path is not None:
            status["updated_at"] = datetime.now(timezone.utc).isoformat()
            status["clients"] = [client_status[cid] for cid in sorted(client_status)]
            _atomic_status(status_path, status)

    def on_training_event(event: dict) -> None:
        nonlocal event_seq
        event_seq += 1
        kind = str(event.get("event", "training_event"))
        rnd = int(event.get("round", status.get("round", 0)))
        status["round"] = max(int(status.get("round", 0)), rnd)
        status["status"] = "training"
        status["phase"] = {
            "round_started": "broadcasting",
            "client_started": "local_training",
            "client_completed": "collecting_updates",
            "aggregation_started": "aggregating",
            "evaluation_started": "central_evaluation",
            "round_completed": "round_complete",
        }.get(kind, "training")

        cid = event.get("client_id")
        if cid is not None:
            cid = int(cid)
            rec = client_status.setdefault(cid, {"client_id": cid})
            rec.update({k: event[k] for k in (
                "num_samples", "train_loss", "train_balanced_accuracy",
                "examples_seen", "dp_steps", "dp_sample_rate",
                "epsilon", "delta",
            ) if k in event})
            rec["status"] = "training" if kind == "client_started" else "complete"

        if kind == "round_started":
            for rec in client_status.values():
                rec["status"] = "waiting"
        elif kind == "round_completed":
            metrics = dict(event.get("metrics") or {})
            status["history"].append(metrics)
            status["latest_metrics"] = metrics
            status["completed_rounds"] = rnd
            for rec in client_status.values():
                rec["status"] = "waiting"

        status["events"].append({
            "sequence": event_seq,
            "time": datetime.now(timezone.utc).isoformat(),
            **{k: _json_safe(v) for k, v in event.items() if k != "metrics"},
        })
        status["events"] = status["events"][-80:]
        publish_status()

    publish_status()

    # torch-dependent imports deferred so --help / config resolution need no torch.
    from fl_med.data.loaders import (
        build_centralized_dataloaders, build_client_dataloaders, list_clients,
    )
    from fl_med.models import model_builder_from_config
    from fl_med.engine.baselines import run_centralized, run_local_only
    from fl_med.engine.server import run_federated
    from fl_med.strategies import build_strategy
    from fl_med.data.heterogeneity import counts_from_dataset
    from fl_med.data.paths import resolve_data_root
    from fl_med.losses import build_criterion

    model_builder = model_builder_from_config(config)

    # Class-imbalance-aware loss: inverse-frequency weights from train label counts
    # (label counts are shareable meta-info, so global weights don't leak images).
    criterion = None
    try:
        fixed_counts = (config.get("loss") or {}).get("fixed_class_counts")
        counts = fixed_counts
        if counts is None:
            counts = counts_from_dataset(
                Path(resolve_data_root(config)) / "train",
                num_classes=int(config.get("model", {}).get("num_classes", 8)),
            ).sum(axis=0)
        criterion = build_criterion(config, counts, device=args.device)
        logger.info(
            "loss=%s train_class_counts=%s",
            (config.get("loss") or {}).get("class_weights", "none"),
            counts.tolist() if hasattr(counts, "tolist") else list(counts),
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("weighted criterion unavailable (%s); using unweighted CE", exc)

    if task == "centralized":
        train_loader, test_loader = build_centralized_dataloaders(config)
        result = run_centralized(
            config=config, model_builder=model_builder,
            train_loader=train_loader, test_loader=test_loader,
            device=args.device, criterion=criterion, logger=logger,
        )
        save_history_csv(result["history"], output_dir / "metrics.csv")
        plot_curves(result["history"], "epoch",
                    ["test_balanced_accuracy", "test_macro_f1"],
                    output_dir / "curves.png", title="Centralized",
                    xlabel="epoch", ylabel="score")
        save_json(result["final_metrics"], output_dir / "summary.json")

    elif task == "local_only":
        clients = list_clients(config)
        _, test_loader = build_centralized_dataloaders(config)
        result = run_local_only(
            config=config, model_builder=model_builder, client_ids=clients,
            client_loader_fn=lambda cid: build_client_dataloaders(config, cid),
            test_loader=test_loader, device=args.device, criterion=criterion, logger=logger,
        )
        save_json(result, output_dir / "summary.json")

    elif task == "federated":
        clients = list_clients(config)
        _, test_loader = build_centralized_dataloaders(config)
        result = run_federated(
            config=config, model_builder=model_builder,
            strategy=build_strategy(config), client_ids=clients,
            client_loader_fn=lambda cid: build_client_dataloaders(config, cid)[0],
            test_loader=test_loader, device=args.device, criterion=criterion, logger=logger,
            event_callback=on_training_event if status_path is not None else None,
        )
        save_history_csv(result["history"], output_dir / "metrics.csv")
        plot_curves(result["history"], "round",
                    ["test_balanced_accuracy", "test_macro_f1"],
                    output_dir / "curves.png", title=f"{result['strategy']} (global)",
                    xlabel="round", ylabel="score")
        plot_curves(result["history"], "round", ["client_drift"],
                    output_dir / "drift.png", title=f"{result['strategy']} client drift",
                    xlabel="round", ylabel="mean L2 drift")
        save_json(result["final_metrics"], output_dir / "summary.json")
    else:
        raise SystemExit(f"Unknown task '{task}'")

    logger.info("done -> %s", output_dir)
    if status_path is not None:
        status["status"] = "completed"
        status["phase"] = "completed"
        status["round"] = int(status.get("total_rounds", status.get("round", 0)))
        status["completed_rounds"] = status["round"]
        for rec in client_status.values():
            rec["status"] = "complete"
        event_seq += 1
        status["events"].append({
            "sequence": event_seq,
            "time": datetime.now(timezone.utc).isoformat(),
            "event": "run_completed",
            "round": status["round"],
        })
        publish_status()
    print(f"OK: results in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
