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
import sys
from pathlib import Path

# Make ``src`` importable without an editable install (useful in CI/environment).
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med import CLASS_NAMES  # noqa: E402
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
    p.add_argument("overrides", nargs="*", help="dotted.key=value overrides")
    return p.parse_args(argv)


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

    # torch-dependent imports are deferred so --help / config resolution need no torch.
    from fl_med.data.loaders import (
        build_centralized_dataloaders, build_client_dataloaders, list_clients,
    )
    from fl_med.models import model_builder_from_config
    from fl_med.engine.baselines import run_centralized, run_local_only
    from fl_med.engine.server import run_federated
    from fl_med.strategies import build_strategy

    model_builder = model_builder_from_config(config)

    if task == "centralized":
        train_loader, test_loader = build_centralized_dataloaders(config)
        result = run_centralized(
            config=config, model_builder=model_builder,
            train_loader=train_loader, test_loader=test_loader,
            device=args.device, logger=logger,
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
            test_loader=test_loader, device=args.device, logger=logger,
        )
        save_json(result, output_dir / "summary.json")

    elif task == "federated":
        clients = list_clients(config)
        _, test_loader = build_centralized_dataloaders(config)
        result = run_federated(
            config=config, model_builder=model_builder,
            strategy=build_strategy(config), client_ids=clients,
            client_loader_fn=lambda cid: build_client_dataloaders(config, cid)[0],
            test_loader=test_loader, device=args.device, logger=logger,
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
    print(f"OK: results in {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
