#!/usr/bin/env python3
"""Phase 5: membership-inference attack on a non-private vs a DP FedAvg model.

Trains two global models with an identical budget (GroupNorm ResNet, weighted CE) --
one non-private, one with DP-SGD -- then runs a loss-threshold MIA on each:
members = pooled train (data the model saw), non-members = pooled test. Reports the
attack AUC for each (AUC ~ 0.5 == no leak) and plots them. Expectation: the
non-private model leaks (AUC > 0.5) and DP reduces the leak toward 0.5.

    DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/run_mia.py --device cuda --rounds 10
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

IMAGE_SIZE = 64


def _eval_loader(root, split, batch_size=128):
    from torch.utils.data import DataLoader

    from fl_med.data.dataset import ISICFederatedFolderDataset
    from fl_med.data.transforms import get_eval_transforms

    ds = ISICFederatedFolderDataset(Path(root) / split, transform=get_eval_transforms(IMAGE_SIZE))
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2)


def _train_global(config, device, logger):
    from fl_med.data.heterogeneity import counts_from_dataset
    from fl_med.data.loaders import (
        build_centralized_dataloaders, build_client_dataloaders, list_clients,
    )
    from fl_med.data.paths import resolve_data_root
    from fl_med.losses import build_criterion
    from fl_med.models import build_model, model_builder_from_config
    from fl_med.strategies import build_strategy
    from fl_med.engine.server import run_federated

    counts = counts_from_dataset(Path(resolve_data_root(config)) / "train").sum(axis=0)
    criterion = build_criterion(config, counts, device=device)
    clients = list_clients(config)
    _, test_loader = build_centralized_dataloaders(config)
    result = run_federated(
        config=config, model_builder=model_builder_from_config(config),
        strategy=build_strategy(config), client_ids=clients,
        client_loader_fn=lambda cid: build_client_dataloaders(config, cid)[0],
        test_loader=test_loader, device=device, criterion=criterion, logger=logger,
    )
    model = build_model(config).to(device)
    model.load_state_dict(result["global_state"])
    return model, result["final_metrics"]


def _plot(results, path):
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    tags = list(results)
    aucs = [results[t]["attack_auc"] for t in tags]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    ax.bar(tags, aucs, color=["#d62728" if "non" in t else "#2ca02c" for t in tags])
    ax.axhline(0.5, ls="--", color="k", lw=1, label="chance (no leak)")
    ax.set_ylabel("Membership-inference attack AUC")
    ax.set_title("DP reduces membership leakage")
    ax.set_ylim(0.4, max(0.75, max(aucs) * 1.1))
    for i, a in enumerate(aucs):
        ax.text(i, a, f"{a:.3f}", ha="center", va="bottom")
    ax.legend()
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    from fl_med.data.paths import resolve_data_root

    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--sigma", type=float, default=1.0, help="DP noise multiplier")
    p.add_argument("--local-epochs", type=int, default=4, help="local epochs/round (overfit)")
    p.add_argument("--data-root", default=None)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    data_root = args.data_root or os.environ.get("DATA_ROOT")
    # Overfit the target so a membership signal exists: no augmentation + several
    # local epochs/round. Batch 32 (smaller -> memorises faster; DP OOM-safe via BMM).
    base = [f"seed={args.seed}", "data.batch_size=32", f"federated.rounds={args.rounds}",
            "federated.max_batches=null", "data.num_workers=2", "data.augment=false",
            f"federated.local_epochs={args.local_epochs}"]
    if data_root:
        base.append(f"data.root={data_root}")

    outdir = REPO / "experiments" / "mia"
    outdir.mkdir(parents=True, exist_ok=True)
    logger = get_logger("run_mia", log_file=outdir / "run.log")
    set_seed(args.seed)

    from fl_med.security.attacks.mia import membership_inference

    variants = [("non_private", ["privacy.enabled=false"]),
                (f"dp_sigma{args.sigma}", ["privacy.enabled=true",
                                           f"privacy.noise_multiplier={args.sigma}"])]
    results = {}
    for tag, priv in variants:
        logger.info("=== training %s ===", tag)
        config = resolve_config(REPO / "configs" / "dp_fedavg.yaml", tier="dev", overrides=base + priv)
        model, final = _train_global(config, args.device, logger)
        root = resolve_data_root(config)
        atk = membership_inference(model, _eval_loader(root, "train"),
                                   _eval_loader(root, "test"), args.device)
        atk["test_balanced_accuracy"] = final.get("test_balanced_accuracy")
        atk["epsilon_max"] = final.get("epsilon_max")
        results[tag] = atk
        logger.info("%s: attack_auc=%.4f member_loss=%.3f nonmember_loss=%.3f bal_acc=%.3f eps=%s",
                    tag, atk["attack_auc"], atk["member_loss_mean"], atk["nonmember_loss_mean"],
                    atk.get("test_balanced_accuracy") or float("nan"), atk.get("epsilon_max"))

    save_json(results, outdir / "mia_results.json")
    _plot(results, REPO / "reports" / "figures" / "mia_auc.png")
    print("\n=== Membership-inference attack AUC (0.5 = no leak) ===")
    for tag, r in results.items():
        ba = r.get("test_balanced_accuracy")
        print(f"  {tag:16} AUC={r['attack_auc']:.4f}  bal_acc={ba:.3f}  eps={r.get('epsilon_max')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
