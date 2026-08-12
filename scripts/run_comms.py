#!/usr/bin/env python3
"""Bonus: communication cost vs accuracy with top-k gradient sparsification.

FL's real-world bottleneck is communication — acute for an edge device like the Pi.
Top-k sparsification sends only the largest-magnitude coordinates of each client update
(value + index) and zeros the rest, cutting bytes/round. We sweep the kept fraction and
plot accuracy against communication, quantifying how much bandwidth can be saved for how
little accuracy. Complements the Pi straggler finding (slow AND bandwidth-limited edge).

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/run_comms.py --device cuda --rounds 12
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import OrderedDict
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


def topk_sparsify(delta, frac):
    """Keep the top ``frac`` of each PARAMETER's coordinates by |magnitude| (layer-wise),
    so no layer — especially the small classifier head — is starved of updates."""
    import numpy as np
    if frac >= 1.0:
        return delta, 1.0
    out, kept, total = OrderedDict(), 0, 0
    for key, v in delta.items():
        v = np.asarray(v)
        flat = np.abs(v).ravel()
        n = flat.size
        total += n
        k = max(1, int(n * frac))
        if k >= n:
            out[key] = v
            kept += n
            continue
        thresh = np.partition(flat, n - k)[n - k]
        mask = np.abs(v) >= thresh
        out[key] = v * mask
        kept += int(mask.sum())
    return out, kept / total


def run_level(frac, config, clients, device, criterion, test_loader, rounds, max_batches,
              logger, init_np=None):
    import numpy as np
    from fl_med.models import build_model

    global_np = ({k: np.array(v) for k, v in init_np.items()} if init_np is not None
                 else _to_numpy(build_model(config).state_dict()))
    n_params = int(sum(np.asarray(v).size for v in global_np.values()))
    curve, kept_fracs = [], []
    for rnd in range(1, rounds + 1):
        deltas = []
        for cid in clients:
            local = _train_client(global_np, config, cid, device, criterion, max_batches)
            delta = OrderedDict((k, np.asarray(local[k]) - np.asarray(global_np[k])) for k in local)
            sdelta, kf = topk_sparsify(delta, frac)
            deltas.append(sdelta)
            kept_fracs.append(kf)
        mean_delta = OrderedDict((k, np.mean([d[k] for d in deltas], axis=0)) for k in deltas[0])
        global_np = OrderedDict((k, np.asarray(global_np[k]) + mean_delta[k]) for k in global_np)
        ba = _evaluate(global_np, config, device, test_loader)["balanced_accuracy"]
        curve.append(ba)
        logger.info("frac=%.3f round %2d bal_acc=%.4f", frac, rnd, ba)
    kept = float(np.mean(kept_fracs))
    dense_bytes = n_params * 4                                   # float32 dense
    sparse_bytes = int(kept * n_params) * 8 if frac < 1.0 else dense_bytes  # value + index
    return {"curve": curve, "best": max(curve), "final": curve[-1], "kept_frac": kept,
            "bytes_per_client_round": int(sparse_bytes),
            "compression": dense_bytes / max(sparse_bytes, 1), "n_params": n_params}


def _plot(results, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    levels = [k for k in results if k != "_meta"]
    levels.sort(key=lambda k: results[k]["compression"])
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.3))
    for k in levels:
        c = results[k]["curve"]
        ax1.plot(range(1, len(c) + 1), c, marker="o", ms=3,
                 label=f"keep {float(k):.0%}  ({results[k]['compression']:.0f}× smaller)")
    ax1.axhline(1 / 8, ls=":", color="gray", lw=1, label="majority floor")
    ax1.set_xlabel("Federated round"); ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("Accuracy vs round, per compression level"); ax1.legend(fontsize=8)

    comps = [results[k]["compression"] for k in levels]
    bests = [results[k]["best"] for k in levels]
    ax2.plot(comps, bests, "o-", color="#1f6fb2")
    for k in levels:
        ax2.annotate(f"keep {float(k):.0%}", (results[k]["compression"], results[k]["best"]),
                     fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax2.set_xscale("log"); ax2.set_xlabel("Compression ratio (× smaller upload)")
    ax2.set_ylabel("Best balanced accuracy")
    ax2.set_title("Communication–accuracy trade-off")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=12)
    p.add_argument("--levels", default="1.0,0.1,0.01", help="kept fractions, comma-separated")
    p.add_argument("--max-batches", type=int, default=25)
    p.add_argument("--init-model", default=None,
                   help="warm-start global model (.pt) so every level starts converged -> "
                        "measures the PURE compression effect, not the warm-up delay")
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

    outdir = REPO / "experiments" / "comms"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("comms", log_file=outdir / "run.log")

    counts = counts_from_dataset(Path(resolve_data_root(config)) / "train").sum(axis=0)
    criterion = build_criterion(config, counts, device=device)
    clients = list_clients(config)
    _, test_loader = build_centralized_dataloaders(config)

    init_np = None
    if args.init_model and Path(args.init_model).exists():
        from fl_med.models import build_model
        m = build_model(config)
        m.load_state_dict(torch.load(args.init_model, map_location="cpu"))
        init_np = _to_numpy(m.state_dict())
        logger.info("warm-started from %s", args.init_model)

    levels = [float(x) for x in args.levels.split(",")]
    results = {"_meta": {"levels": levels, "rounds": args.rounds, "warm_start": bool(init_np)}}
    for frac in levels:
        results[str(frac)] = run_level(frac, config, clients, device, criterion, test_loader,
                                       args.rounds, args.max_batches, logger, init_np=init_np)

    save_json(results, outdir / "comms_results.json")
    _plot(results, REPO / "reports" / "figures" / "comms.png")
    print("\n=== Communication vs accuracy (top-k sparsification) ===")
    for frac in levels:
        r = results[str(frac)]
        print(f"  keep {frac:6.1%}  best bal_acc={r['best']:.4f}  "
              f"{r['compression']:.0f}x smaller upload  ({r['bytes_per_client_round']/1e6:.2f} MB/round/client)")
    print(f"\nfigure -> reports/figures/comms.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
