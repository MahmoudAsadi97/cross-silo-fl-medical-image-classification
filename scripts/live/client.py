#!/usr/bin/env python3
"""REAL federated CLIENT (one hospital / silo).

Trains on ONE silo's local data and sends only model updates to the server over
the network -- the raw images never leave this machine. Run one process per
hospital. On the Raspberry Pi this is the "edge hospital"; use --max-batches to
cap work per round so the Pi stays fast.

    # a laptop client (silo 0), server on this LAN:
    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
        python scripts/live/client.py --server 192.168.1.50:8080 --client-id 0 --label laptop-c0

    # the Raspberry Pi (small silo 5, capped for speed):
    DATA_ROOT=~/fl_data/fed_isic2019/raw \
        python scripts/live/client.py --server 192.168.1.50:8080 --client-id 5 \
            --device cpu --max-batches 8 --label pi5

Tested with flwr 1.7-1.12 (start_client API). On flwr >= 1.13 either
`pip install 'flwr==1.11.1'` or adapt to the ClientApp API.
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.loaders import build_client_dataloaders  # noqa: E402
from fl_med.federated_live import (  # noqa: E402
    build_model, get_ndarrays, local_examples_seen, local_fit, local_num_examples, set_ndarrays,
)
from fl_med.seeding import set_seed  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--server", default="127.0.0.1:8080", help="server address host:port")
    p.add_argument("--client-id", type=int, required=True, help="which silo (0..5)")
    p.add_argument("--config", default=str(REPO / "configs" / "live_fedavg.yaml"))
    p.add_argument("--tier", default="dev")
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--max-batches", type=int, default=None,
                   help="cap batches/epoch (keeps a slow device responsive)")
    p.add_argument("--freeze-backbone", action="store_true",
                   help="train only the classifier head (partial-model FL): large speedup "
                        "on weak devices (Pi); architecture unchanged so FedAvg still works")
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader workers (0 avoids process/RAM blow-up when many "
                        "clients share one machine)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--label", default=None, help="friendly tag (e.g. 'pi5', 'laptop-c0')")
    p.add_argument("--seed", type=int, default=42, help="base reproducibility seed")
    p.add_argument(
        "--class-counts", default=None,
        help="comma-separated global train counts used to match centralized class weighting",
    )
    args = p.parse_args(argv)

    import flwr as fl
    import torch

    set_seed(args.seed + args.client_id)

    class_counts = None
    if args.class_counts:
        try:
            class_counts = [int(value) for value in args.class_counts.split(",")]
        except ValueError as exc:
            raise SystemExit("--class-counts must contain comma-separated integers") from exc
        if len(class_counts) != 8 or any(value < 0 for value in class_counts):
            raise SystemExit("--class-counts must contain eight non-negative integers")

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
    # Flower supplies the server's global parameters before local training, so
    # client-side ImageNet weights would immediately be overwritten. Avoiding
    # that download keeps the Raspberry Pi demo fully offline-capable.
    overrides.append("model.pretrained=false")
    config = resolve_config(Path(args.config), tier=args.tier, overrides=overrides)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    train_loader, _ = build_client_dataloaders(config, args.client_id)
    model = build_model(config).to(device)
    tag = args.label or f"client-{args.client_id}"
    partition_size = len(train_loader.dataset)
    examples_per_epoch = local_num_examples(train_loader, args.max_batches)

    class Client(fl.client.NumPyClient):
        def get_parameters(self, cfg):
            return get_ndarrays(model)

        def fit(self, parameters, cfg):
            set_ndarrays(model, parameters)
            t0 = time.perf_counter()
            hist = local_fit(model, train_loader, config, local_epochs=args.local_epochs,
                             device=device, max_batches=args.max_batches,
                             freeze_backbone=args.freeze_backbone,
                             class_counts=class_counts)
            dt = time.perf_counter() - t0
            last = hist[-1]
            examples_seen = local_examples_seen(hist)
            metrics = {
                "client_id": int(args.client_id), "tag": tag, "device": device,
                "fit_seconds": float(dt), "n": int(partition_size),
                "examples_seen": int(examples_seen),
                "train_loss": float(last["loss"]),
                "train_bal_acc": float(last["balanced_accuracy"]),
                "freeze_backbone": bool(args.freeze_backbone),
            }
            print(f"[{tag}] fit done: {dt:.1f}s  partition_n={partition_size}  "
                  f"examples_seen={examples_seen}  "
                  f"loss={last['loss']:.3f}", flush=True)
            return get_ndarrays(model), int(partition_size), metrics

        def evaluate(self, parameters, cfg):
            return 0.0, 1, {}   # central evaluation is done on the server

    print(f"[{tag}] connecting to {args.server}  (device={device}, silo={args.client_id}, "
          f"partition_n={partition_size}, planned_examples_per_epoch={examples_per_epoch}, "
          f"local_epochs={args.local_epochs})", flush=True)
    fl.client.start_client(server_address=args.server, client=Client().to_client())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
