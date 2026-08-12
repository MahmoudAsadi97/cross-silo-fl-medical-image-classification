#!/usr/bin/env python3
"""Bonus: Byzantine robustness — a model-poisoning attack vs robust aggregation.

Some clients are MALICIOUS: after honest local training they flip and amplify their
update (a model-poisoning attack) to corrupt the shared model. We compare, under the
same attack, how different server aggregators cope:

* ``fedavg``       — sample-weighted mean (breakdown point 0: one bad client can dominate),
* ``median``       — per-parameter coordinate median,
* ``trimmed_mean`` — per-parameter mean after dropping the extremes,
* ``krum``         — pick the update most consistent with the honest majority.

Expected story: plain FedAvg collapses under attack; the robust aggregators stay close
to the clean (no-attack) baseline. This closes the *integrity* side of the security
analysis (DP + membership inference + secure aggregation cover *confidentiality*).

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
        python scripts/run_robustness.py --device cuda --rounds 12 --malicious 2 --attack sign_flip
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.config import resolve_config  # noqa: E402
from fl_med.eval import save_json  # noqa: E402
from fl_med.logging import get_logger  # noqa: E402
from fl_med.seeding import set_seed  # noqa: E402


def _to_numpy(state):
    return {k: v.detach().cpu().numpy() for k, v in state.items()}


def _load_np(model, np_state):
    import torch
    sd = model.state_dict()
    model.load_state_dict({k: torch.as_tensor(np_state[k], dtype=sd[k].dtype) for k in sd})


def _train_client(global_np, config, cid, device, criterion, max_batches):
    """Honest local training for one client; returns its updated weights as numpy."""
    from fl_med.data.loaders import build_client_dataloaders
    from fl_med.engine.client import _build_optimizer
    from fl_med.engine.train_eval import GRAD_CLIP_NORM, train_one_epoch
    from fl_med.models import build_model

    model = build_model(config).to(device)
    _load_np(model, global_np)
    loader, _ = build_client_dataloaders(config, cid)
    opt = _build_optimizer(model, config.get("optimizer", {"name": "adam", "lr": 1e-3}))
    train_one_epoch(model, loader, opt, device, criterion=criterion, num_classes=8,
                    max_batches=max_batches, grad_clip=GRAD_CLIP_NORM)
    return _to_numpy(model.state_dict())


def _evaluate(global_np, config, device, test_loader):
    from fl_med.engine.train_eval import evaluate
    from fl_med.models import build_model

    model = build_model(config).to(device)
    _load_np(model, global_np)
    return evaluate(model, test_loader, device, num_classes=8)


def run_setting(aggregator, malicious_ids, attack, scale, config, clients, device,
                criterion, test_loader, rounds, max_batches, logger):
    import numpy as np
    from fl_med.models import build_model
    from fl_med.security.robust_agg import POISON_ATTACKS, coordinate_median, krum, trimmed_mean
    from fl_med.strategies.aggregation import weighted_average

    global_np = _to_numpy(build_model(config).state_dict())
    f = max(1, len(malicious_ids))
    curve = []
    for rnd in range(1, rounds + 1):
        states = []
        for cid in clients:
            local = _train_client(global_np, config, cid, device, criterion, max_batches)
            if cid in malicious_ids:
                local = dict(POISON_ATTACKS[attack](local, global_np, scale))
            states.append(local)
        if aggregator == "fedavg":
            new_np = weighted_average(states, [1.0] * len(states))
        elif aggregator == "median":
            new_np = coordinate_median(states)
        elif aggregator == "trimmed_mean":
            new_np = trimmed_mean(states, trim=min(f, (len(states) - 1) // 2))
        elif aggregator == "krum":
            new_np = krum(states, num_malicious=f, multi=1)
        else:
            raise ValueError(aggregator)
        global_np = {k: np.asarray(v) for k, v in new_np.items()}
        ba = _evaluate(global_np, config, device, test_loader)["balanced_accuracy"]
        curve.append(ba)
        logger.info("%-14s round %2d bal_acc=%.4f", aggregator, rnd, ba)
    return {"curve": curve, "best": max(curve), "final": curve[-1]}


def _plot(results, path, n_clients, args):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    styles = {
        "fedavg_clean": ("clean FedAvg (no attack)", "#000000", "--"),
        "fedavg_attacked": ("FedAvg under attack", "#d62728", "-"),
        "median_attacked": ("coordinate-median (robust)", "#2ca02c", "-"),
        "trimmed_mean_attacked": ("trimmed-mean (robust)", "#1f77b4", "-"),
        "krum_attacked": ("Krum (robust)", "#9467bd", "-"),
    }
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for tag, (label, color, ls) in styles.items():
        if tag in results:
            c = results[tag]["curve"]
            ax.plot(range(1, len(c) + 1), c, ls, color=color, marker="o", ms=3, label=label)
    ax.axhline(1 / 8, ls=":", color="gray", lw=1, label="majority floor (1/8)")
    ax.set_xlabel("Federated round")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Byzantine robustness: {args.malicious}/{n_clients} malicious clients "
                 f"({args.attack}, ×{args.scale:g})")
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--malicious", type=int, default=2, help="number of malicious clients")
    p.add_argument("--attack", default="sign_flip", choices=["sign_flip", "scale"])
    p.add_argument("--scale", type=float, default=5.0, help="attack amplification")
    p.add_argument("--max-batches", type=int, default=25)
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    import torch

    from fl_med.data.heterogeneity import counts_from_dataset
    from fl_med.data.loaders import build_centralized_dataloaders, list_clients
    from fl_med.data.paths import resolve_data_root
    from fl_med.losses import build_criterion

    overrides = ["data.num_workers=2", f"seed={args.seed}"]
    data_root = args.data_root or os.environ.get("DATA_ROOT")
    if data_root:
        overrides.append(f"data.root={data_root}")
    config = resolve_config(REPO / "configs" / "fedavg.yaml", tier="dev", overrides=overrides)
    device = args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    set_seed(args.seed)

    outdir = REPO / "experiments" / "robustness"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("robustness", log_file=outdir / "run.log")

    counts = counts_from_dataset(Path(resolve_data_root(config)) / "train").sum(axis=0)
    criterion = build_criterion(config, counts, device=device)
    clients = list_clients(config)
    _, test_loader = build_centralized_dataloaders(config)
    malicious_ids = clients[-args.malicious:] if args.malicious > 0 else []
    logger.info("clients=%s  malicious=%s  attack=%s scale=%s", clients, malicious_ids,
                args.attack, args.scale)

    results = {"_meta": {"clients": clients, "malicious": malicious_ids,
                         "attack": args.attack, "scale": args.scale, "rounds": args.rounds}}
    results["fedavg_clean"] = run_setting("fedavg", [], args.attack, args.scale, config, clients,
                                          device, criterion, test_loader, args.rounds,
                                          args.max_batches, logger)
    for agg in ["fedavg", "median", "trimmed_mean", "krum"]:
        results[f"{agg}_attacked"] = run_setting(agg, malicious_ids, args.attack, args.scale,
                                                  config, clients, device, criterion, test_loader,
                                                  args.rounds, args.max_batches, logger)

    save_json(results, outdir / "robustness_results.json")
    _plot(results, REPO / "reports" / "figures" / "robustness.png", len(clients), args)
    print("\n=== Byzantine robustness — balanced accuracy (best over rounds) ===")
    for tag in ["fedavg_clean", "fedavg_attacked", "median_attacked", "trimmed_mean_attacked", "krum_attacked"]:
        r = results[tag]
        print(f"  {tag:22} best={r['best']:.4f}  final={r['final']:.4f}")
    print(f"\nfigure -> reports/figures/robustness.png ; json -> {outdir/'robustness_results.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
