#!/usr/bin/env python3
"""REAL federated SERVER (the neutral coordinator).

Coordinates rounds over the network and aggregates client updates with FedAvg.
Each round it evaluates the global model CENTRALLY on the pooled test set, and
records every client's local fit time -- the data behind the "straggler" view
(a slow edge device, e.g. the Raspberry Pi, holds up the round).

Run on the laptop (the machine with the GPU + the test data):

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
        python scripts/live/server.py --rounds 8 --min-clients 2 --host 0.0.0.0:8080

Then start one client per silo (see scripts/live/client.py). Writes
experiments/live/history.json (accuracy-per-round + per-client timing).

Tested with flwr 1.7-1.12 (start_server / FedAvg API). On flwr >= 1.13 either
`pip install 'flwr==1.11.1'` or adapt to the ServerApp API.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.loaders import build_centralized_test_dataloader  # noqa: E402
from fl_med.federated_live import (  # noqa: E402
    build_model, evaluate_model, get_ndarrays, set_ndarrays,
)
from fl_med.seeding import set_seed  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rounds", type=int, default=8)
    p.add_argument("--min-clients", type=int, default=2, help="wait for this many clients")
    p.add_argument("--host", default="0.0.0.0:8080", help="bind address (0.0.0.0 = all NICs)")
    p.add_argument("--config", default=str(REPO / "configs" / "live_fedavg.yaml"))
    p.add_argument("--tier", default="dev")
    p.add_argument("--device", default="cpu")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader workers for central eval (0 keeps RAM low)")
    p.add_argument("--round-timeout", type=float, default=None,
                   help="seconds to wait for a round before giving up (None = no limit)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--offline-init", action="store_true",
                   help="start without downloading pretrained weights")
    p.add_argument("--init-model", default=None,
                   help="warm-start the global model from a saved .pt (continued FL) so a "
                        "short live run shows real accuracy instead of a from-scratch climb")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(REPO / "experiments" / "live"))
    args = p.parse_args(argv)

    import flwr as fl
    import torch
    from flwr.common import ndarrays_to_parameters

    set_seed(args.seed)

    overrides = []
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    if args.image_size:
        overrides.append(f"data.image_size={args.image_size}")
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    overrides.append(f"seed={args.seed}")
    overrides.append(f"data.num_workers={args.num_workers}")
    # A warm-start checkpoint replaces every model tensor, so downloading
    # ImageNet weights here would add a needless presentation-day network
    # dependency. The architecture is identical with pretrained=False.
    if args.init_model or args.offline_init:
        overrides.append("model.pretrained=false")
    config = resolve_config(Path(args.config), tier=args.tier, overrides=overrides)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    num_classes = int(config.get("model", {}).get("num_classes", 8))
    test_loader = build_centralized_test_dataloader(config)
    eval_model = build_model(config).to(device)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    history: list = []   # per-round central metrics
    timings: list = []   # per-client, per-round fit timing
    latest_clients: list = []   # most recent round's per-client timing (for the live dashboard)
    live_path = outdir / "live_status.json"
    status_sequence = 0
    current_round = 0
    events: list[dict] = []

    def write_live(status, *, phase=None, event=None, event_round=None, **event_data):
        """Atomically publish aggregate status for dashboards and the local control room."""
        nonlocal status_sequence, current_round
        status_sequence += 1
        if event_round is not None:
            current_round = max(current_round, int(event_round))
        if event is not None:
            events.append({
                "sequence": status_sequence,
                "time": datetime.now(timezone.utc).isoformat(),
                "event": event,
                "round": current_round,
                **event_data,
            })
            del events[:-80]
        payload = {
            "schema_version": 1,
            "mode": "networked_flower",
            "sequence": status_sequence,
            "updated_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "phase": phase or status,
            "round": current_round,
            "active_round": current_round,
            "completed_rounds": history[-1]["round"] if history else 0,
            "total_rounds": args.rounds,
            "history": [{
                "round": h["round"],
                "bal_acc": h["bal_acc"],
                "macro_f1": h["macro_f1"],
                "accuracy": h["accuracy"],
                "loss": h["loss"],
            } for h in history],
            "clients": latest_clients,
            "events": events,
        }
        tmp = live_path.with_name(f".{live_path.name}.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(payload, allow_nan=False), encoding="utf-8")
        os.replace(tmp, live_path)

    write_live("waiting", phase="waiting_clients", event="coordinator_ready")

    def evaluate_fn(server_round, parameters, cfg):
        write_live(
            "training", phase="central_evaluation", event="evaluation_started",
            event_round=server_round,
        )
        set_ndarrays(eval_model, parameters)
        m = evaluate_model(eval_model, test_loader, device, num_classes=num_classes)
        history.append({"round": server_round, "bal_acc": m["balanced_accuracy"],
                        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                        "loss": m["loss"]})
        print(f"[server] round {server_round}: central bal_acc={m['balanced_accuracy']:.4f} "
              f"loss={m['loss']:.3f}", flush=True)
        write_live(
            "training", phase="round_complete", event="round_evaluated",
            event_round=server_round,
        )
        return float(m["loss"]), {"bal_acc": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}

    class TimedFedAvg(fl.server.strategy.FedAvg):
        """FedAvg that records each client's reported fit timing per round."""

        def configure_fit(self, server_round, parameters, client_manager):
            write_live(
                "training", phase="local_training", event="round_started",
                event_round=server_round,
            )
            return super().configure_fit(server_round, parameters, client_manager)

        def aggregate_fit(self, server_round, results, failures):
            round_clients = []
            for _, fit_res in results:
                rec = dict(fit_res.metrics or {})
                rec["round"] = server_round
                rec["n"] = int(fit_res.num_examples)
                timings.append(rec)
                round_clients.append({
                    k: rec.get(k)
                    for k in (
                        "client_id", "tag", "device", "fit_seconds", "n", "examples_seen", "train_loss",
                        "train_bal_acc", "freeze_backbone",
                    )
                })
                print(f"[server] round {server_round} client '{rec.get('tag', '?')}' "
                      f"{float(rec.get('fit_seconds', float('nan'))):.1f}s "
                      f"n={rec.get('n')}", flush=True)
            latest_clients[:] = round_clients   # newest round's per-client timing (live dashboard)
            write_live(
                "training", phase="aggregating", event="client_updates_received",
                event_round=server_round, participating_clients=len(round_clients),
            )
            return super().aggregate_fit(server_round, results, failures)

    if args.init_model:
        init_path = Path(args.init_model).expanduser()
        if not init_path.is_file():
            raise FileNotFoundError(
                f"Warm-start checkpoint does not exist: {init_path}. "
                "Refusing to silently start the live demo from random weights."
            )
        eval_model.load_state_dict(torch.load(init_path, map_location=device, weights_only=True))
        print(f"[server] warm-started global model from {init_path}", flush=True)
    init = ndarrays_to_parameters(get_ndarrays(eval_model))
    strategy = TimedFedAvg(
        fraction_fit=1.0, fraction_evaluate=0.0,
        min_fit_clients=args.min_clients, min_available_clients=args.min_clients,
        initial_parameters=init, evaluate_fn=evaluate_fn,
    )

    print(f"[server] listening on {args.host}; waiting for >= {args.min_clients} clients; "
          f"{args.rounds} rounds; device={device}", flush=True)
    fl.server.start_server(
        server_address=args.host,
        config=fl.server.ServerConfig(num_rounds=args.rounds, round_timeout=args.round_timeout),
        strategy=strategy,
    )

    write_live(
        "validating", phase="writing_artifacts", event="training_completed",
        event_round=args.rounds,
    )
    (outdir / "history.json").write_text(
        json.dumps({"history": history, "timings": timings}, indent=2))
    print(f"[server] wrote {outdir / 'history.json'} "
          f"({len(history)} rounds, {len(timings)} client-updates)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
