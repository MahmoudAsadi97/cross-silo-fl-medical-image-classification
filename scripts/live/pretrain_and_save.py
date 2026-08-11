#!/usr/bin/env python3
"""Pre-train a global model with the in-process simulation and SAVE it, so the
live (networked) run can warm-start from a real model.

Why: a short live run started from scratch can't reach a good accuracy (and a
single laptop GPU can't train many client processes at once). Training here runs
as ONE process -- the exact simulation used for the report -- so there is no GPU
contention or OOM. The live server then loads this model (`--init-model`) and
performs a few rounds of *continued* federated learning: the demo shows genuine
accuracy from round 0, and the point being demonstrated (real distributed FL +
the edge device participating) is what the live run adds on top.

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
        python scripts/live/pretrain_and_save.py --rounds 15 --device cuda
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.data.heterogeneity import counts_from_dataset  # noqa: E402
from fl_med.data.loaders import (  # noqa: E402
    build_centralized_dataloaders, build_client_dataloaders, list_clients,
)
from fl_med.data.paths import resolve_data_root  # noqa: E402
from fl_med.engine.server import run_federated  # noqa: E402
from fl_med.logging import get_logger  # noqa: E402
from fl_med.losses import build_criterion  # noqa: E402
from fl_med.models import model_builder_from_config  # noqa: E402
from fl_med.strategies import build_strategy  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=str(REPO / "configs" / "live_fedavg.yaml"))
    p.add_argument("--tier", default="dev")
    p.add_argument("--rounds", type=int, default=15)
    p.add_argument("--device", default="cuda")
    p.add_argument("--data-root", default=None)
    p.add_argument("--out", default=str(REPO / "experiments" / "live" / "pretrained.pt"))
    args = p.parse_args(argv)

    import torch

    overrides = [f"federated.rounds={args.rounds}", "data.num_workers=0"]
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    config = resolve_config(Path(args.config), tier=args.tier, overrides=overrides)

    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    logger = get_logger("pretrain", log_file=out.parent / "pretrain.log")

    counts = counts_from_dataset(Path(resolve_data_root(config)) / "train").sum(axis=0)
    criterion = build_criterion(config, counts, device=device)
    _, test_loader = build_centralized_dataloaders(config)

    result = run_federated(
        config=config, model_builder=model_builder_from_config(config),
        strategy=build_strategy(config), client_ids=list_clients(config),
        client_loader_fn=lambda cid: build_client_dataloaders(config, cid)[0],
        test_loader=test_loader, device=device, criterion=criterion, logger=logger,
    )
    torch.save(result["global_state"], out)
    ba = result["final_metrics"].get("test_balanced_accuracy", float("nan"))
    print(f"saved pretrained model -> {out}  (final bal_acc={ba:.4f})")
    print("now warm-start the live run:  ... server.py --init-model", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
