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
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.loaders import build_centralized_dataloaders  # noqa: E402
from fl_med.federated_live import (  # noqa: E402
    build_model, evaluate_model, get_ndarrays, set_ndarrays,
)


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
    p.add_argument("--init-model", default=None,
                   help="warm-start the global model from a saved .pt (continued FL) so a "
                        "short live run shows real accuracy instead of a from-scratch climb")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(REPO / "experiments" / "live"))
    args = p.parse_args(argv)

    import flwr as fl
    import torch
    from flwr.common import ndarrays_to_parameters

    overrides = []
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    if args.image_size:
        overrides.append(f"data.image_size={args.image_size}")
    if args.batch_size:
        overrides.append(f"data.batch_size={args.batch_size}")
    overrides.append(f"data.num_workers={args.num_workers}")
    config = resolve_config(Path(args.config), tier=args.tier, overrides=overrides)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    num_classes = int(config.get("model", {}).get("num_classes", 8))
    _, test_loader = build_centralized_dataloaders(config)
    eval_model = build_model(config).to(device)

    outdir = Path(args.out)
    outdir.mkdir(parents=True, exist_ok=True)
    history: list = []   # per-round central metrics
    timings: list = []   # per-client, per-round fit timing

    def evaluate_fn(server_round, parameters, cfg):
        set_ndarrays(eval_model, parameters)
        m = evaluate_model(eval_model, test_loader, device, num_classes=num_classes)
        history.append({"round": server_round, "bal_acc": m["balanced_accuracy"],
                        "macro_f1": m["macro_f1"], "accuracy": m["accuracy"],
                        "loss": m["loss"]})
        print(f"[server] round {server_round}: central bal_acc={m['balanced_accuracy']:.4f} "
              f"loss={m['loss']:.3f}", flush=True)
        return float(m["loss"]), {"bal_acc": m["balanced_accuracy"], "macro_f1": m["macro_f1"]}

    class TimedFedAvg(fl.server.strategy.FedAvg):
        """FedAvg that records each client's reported fit timing per round."""

        def aggregate_fit(self, server_round, results, failures):
            for _, fit_res in results:
                rec = dict(fit_res.metrics or {})
                rec["round"] = server_round
                rec.setdefault("n", fit_res.num_examples)
                timings.append(rec)
                print(f"[server] round {server_round} client '{rec.get('tag', '?')}' "
                      f"({rec.get('host', '?')}): {float(rec.get('fit_seconds', float('nan'))):.1f}s "
                      f"n={rec.get('n')}", flush=True)
            return super().aggregate_fit(server_round, results, failures)

    if args.init_model and Path(args.init_model).exists():
        eval_model.load_state_dict(torch.load(args.init_model, map_location=device))
        print(f"[server] warm-started global model from {args.init_model}", flush=True)
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

    (outdir / "history.json").write_text(
        json.dumps({"history": history, "timings": timings}, indent=2))
    print(f"[server] wrote {outdir / 'history.json'} "
          f"({len(history)} rounds, {len(timings)} client-updates)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
