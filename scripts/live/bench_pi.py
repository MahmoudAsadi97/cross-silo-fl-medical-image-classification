#!/usr/bin/env python3
"""Self-validating benchmark: how much faster is a federated round with a frozen
backbone (partial-model FL) — and is it CORRECT?

Runs on any machine, but is written for the Raspberry Pi (CPU). It first VERIFIES
the mechanism, then measures the speedup, and prints a PASS/FAIL table:

  correctness checks
    [1] freeze: every backbone tensor is bit-identical after training  (client
        "abstains" on the backbone — required for FedAvg compatibility)
    [2] freeze: the classifier head DID change                          (it learns)
    [3] full:   the backbone DID change                                 (control)
    [4] state-dict keys match a fresh model                             (aggregatable)
  timing
    median seconds per local round: full-model vs frozen-backbone, same batches

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/live/bench_pi.py \
        --client-id 5 --max-batches 8 --repeats 3

Writes experiments/live/bench_pi.json. Exit code != 0 if any check fails.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--client-id", type=int, default=5)
    p.add_argument("--max-batches", type=int, default=8)
    p.add_argument("--repeats", type=int, default=3, help="timed rounds per mode (median reported)")
    p.add_argument("--device", default="cpu")
    p.add_argument("--tier", default="dev")
    p.add_argument("--data-root", default=None)
    p.add_argument("--init-model", default=str(REPO / "experiments" / "live" / "pretrained.pt"))
    args = p.parse_args(argv)

    import torch

    from fl_med.data.loaders import build_client_dataloaders
    from fl_med.federated_live import build_model, local_fit
    from fl_med.federated_live.task import _classifier_head

    overrides = ["data.num_workers=0"]
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    config = resolve_config(REPO / "configs" / "live_fedavg.yaml", tier=args.tier, overrides=overrides)
    device = "cpu" if args.device != "cuda" or not torch.cuda.is_available() else "cuda"

    train_loader, _ = build_client_dataloaders(config, args.client_id)

    def fresh_model():
        m = build_model(config).to(device)
        ip = Path(args.init_model)
        if ip.exists():
            m.load_state_dict(torch.load(ip, map_location=device))
        return m

    def snapshot(m):
        return {k: v.detach().clone() for k, v in m.state_dict().items()}

    def head_param_names(m):
        head = _classifier_head(m)
        head_ids = {id(p) for p in head.parameters()}
        return {n for n, p in m.named_parameters() if id(p) in head_ids}

    def changed(before, after, names):
        """True if ANY tensor in `names` differs between the two snapshots."""
        return any(not torch.equal(before[n], after[n]) for n in names)

    results = {"config": {"client_id": args.client_id, "max_batches": args.max_batches,
                          "tier": args.tier, "device": device,
                          "warm_start": Path(args.init_model).exists()}}
    checks = []

    # ---------- correctness ----------------------------------------------------
    ref = fresh_model()
    ref_keys = list(ref.state_dict().keys())
    hp = head_param_names(ref)
    all_params = {n for n, _ in ref.named_parameters()}
    backbone_params = all_params - hp
    n_head = sum(p.numel() for n, p in ref.named_parameters() if n in hp)
    n_all = sum(p.numel() for p in ref.parameters())
    results["params"] = {"total": int(n_all), "head": int(n_head),
                         "head_fraction": round(n_head / n_all, 6)}
    print(f"model: {n_all:,} params; classifier head = {n_head:,} "
          f"({100*n_head/n_all:.2f}% of the model)")

    # frozen run
    m1 = fresh_model()
    b1 = snapshot(m1)
    local_fit(m1, train_loader, config, local_epochs=1, device=device,
              max_batches=args.max_batches, freeze_backbone=True)
    a1 = snapshot(m1)
    ok1 = not changed(b1, a1, backbone_params)
    ok2 = changed(b1, a1, hp)
    checks.append(("freeze: backbone bit-identical after training", ok1))
    checks.append(("freeze: classifier head changed (it learns)", ok2))

    # full run (control)
    m2 = fresh_model()
    b2 = snapshot(m2)
    local_fit(m2, train_loader, config, local_epochs=1, device=device,
              max_batches=args.max_batches, freeze_backbone=False)
    a2 = snapshot(m2)
    ok3 = changed(b2, a2, backbone_params)
    checks.append(("full: backbone changed (control)", ok3))

    ok4 = list(m1.state_dict().keys()) == ref_keys and list(m2.state_dict().keys()) == ref_keys
    checks.append(("state-dict keys identical -> FedAvg-aggregatable", ok4))

    # ---------- timing ---------------------------------------------------------
    def timed(freeze):
        ts = []
        for _ in range(args.repeats):
            m = fresh_model()
            t0 = time.perf_counter()
            local_fit(m, train_loader, config, local_epochs=1, device=device,
                      max_batches=args.max_batches, freeze_backbone=freeze)
            ts.append(time.perf_counter() - t0)
        return ts

    print(f"\ntiming {args.repeats}+{args.repeats} rounds "
          f"(client {args.client_id}, {args.max_batches} batches, {device}) ...")
    t_full = timed(False)
    t_frozen = timed(True)
    med_full = statistics.median(t_full)
    med_frozen = statistics.median(t_frozen)
    speedup = med_full / med_frozen if med_frozen > 0 else float("nan")
    results["timing"] = {"full_s": [round(x, 2) for x in t_full],
                         "frozen_s": [round(x, 2) for x in t_frozen],
                         "median_full_s": round(med_full, 2),
                         "median_frozen_s": round(med_frozen, 2),
                         "speedup": round(speedup, 2)}

    # ---------- report ---------------------------------------------------------
    print("\n=== correctness ===")
    fails = 0
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
        fails += (not ok)
    results["checks"] = {n: bool(o) for n, o in checks}

    print("\n=== speed (median of %d) ===" % args.repeats)
    print(f"  full model      : {med_full:6.2f} s / round   {t_full}")
    print(f"  frozen backbone : {med_frozen:6.2f} s / round   {t_frozen}")
    print(f"  SPEEDUP         : {speedup:.2f}x")

    out = REPO / "experiments" / "live" / "bench_pi.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {out}")
    return 1 if fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
