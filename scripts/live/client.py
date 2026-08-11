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
import socket
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.loaders import build_client_dataloaders  # noqa: E402
from fl_med.federated_live import (  # noqa: E402
    build_model, get_ndarrays, local_fit, local_num_examples, set_ndarrays,
)


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
    p.add_argument("--image-size", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--num-workers", type=int, default=0,
                   help="dataloader workers (0 avoids process/RAM blow-up when many "
                        "clients share one machine)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--label", default=None, help="friendly tag (e.g. 'pi5', 'laptop-c0')")
    args = p.parse_args(argv)

    import flwr as fl
    import torch

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
    train_loader, _ = build_client_dataloaders(config, args.client_id)
    model = build_model(config).to(device)
    tag = args.label or f"client{args.client_id}@{socket.gethostname()}"
    n_examples = local_num_examples(train_loader, args.max_batches)

    class Client(fl.client.NumPyClient):
        def get_parameters(self, cfg):
            return get_ndarrays(model)

        def fit(self, parameters, cfg):
            set_ndarrays(model, parameters)
            t0 = time.perf_counter()
            hist = local_fit(model, train_loader, config, local_epochs=args.local_epochs,
                             device=device, max_batches=args.max_batches)
            dt = time.perf_counter() - t0
            last = hist[-1]
            metrics = {
                "tag": tag, "host": socket.gethostname(), "device": device,
                "fit_seconds": float(dt), "n": int(n_examples),
                "train_loss": float(last["loss"]),
                "train_bal_acc": float(last["balanced_accuracy"]),
            }
            print(f"[{tag}] fit done: {dt:.1f}s  n={n_examples}  "
                  f"loss={last['loss']:.3f}", flush=True)
            return get_ndarrays(model), int(n_examples), metrics

        def evaluate(self, parameters, cfg):
            return 0.0, 1, {}   # central evaluation is done on the server

    print(f"[{tag}] connecting to {args.server}  (device={device}, silo={args.client_id}, "
          f"n={n_examples}, local_epochs={args.local_epochs})", flush=True)
    fl.client.start_client(server_address=args.server, client=Client().to_client())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
