from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import build_client_dataloaders
from src.models.resnet import build_resnet18
from src.training.engine import train_one_epoch, evaluate
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def main():
    training_cfg = load_yaml(PROJECT_ROOT / "configs/training/centralized.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs/model/resnet18.yaml")
    base_cfg = load_yaml(PROJECT_ROOT / "configs/experiments/base_experiment.yaml")

    seed = int(base_cfg["experiment"]["seed"])
    set_seed(seed, deterministic=bool(base_cfg["experiment"]["deterministic"]))

    device_name = training_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    batch_size = int(training_cfg["batch_size"])
    num_workers = int(base_cfg["experiment"]["num_workers"])
    image_size = int(base_cfg["data"]["image_size"])
    num_classes = int(model_cfg["num_classes"])
    epochs = int(training_cfg["epochs"])
    lr = float(training_cfg["learning_rate"])
    weight_decay = float(training_cfg["weight_decay"])

    local_root = PROJECT_ROOT / "results" / "local_baseline"
    local_root.mkdir(parents=True, exist_ok=True)

    overall_logger = get_logger("local_baseline", str(local_root / "local_baseline.log"))
    overall_logger.info("Starting local-only baseline training")
    overall_logger.info(f"Using device: {device}")

    client_results = []

    for client_id in range(6):
        client_dir = local_root / f"client_{client_id}"
        ckpt_dir = client_dir / "checkpoints"
        log_dir = client_dir / "logs"
        history_path = client_dir / "history.json"
        summary_path = client_dir / "summary.json"

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        log_dir.mkdir(parents=True, exist_ok=True)

        logger = get_logger(
            f"local_client_{client_id}",
            str(log_dir / "train.log"),
        )

        logger.info(f"Starting local training for client {client_id}")

        train_loader, test_loader = build_client_dataloaders(
            client_id=client_id,
            image_size=image_size,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = build_resnet18(
            num_classes=num_classes,
            pretrained=bool(model_cfg["pretrained"]),
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        history = {
            "train": [],
            "test": [],
        }

        best_macro_f1 = -1.0
        best_epoch = None
        best_ckpt_path = ckpt_dir / "best_model_by_macro_f1.pt"

        for epoch in range(1, epochs + 1):
            logger.info(f"Epoch {epoch}/{epochs} | lr={get_current_lr(optimizer):.8f}")

            train_metrics = train_one_epoch(
                model=model,
                dataloader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
            )

            test_metrics = evaluate(
                model=model,
                dataloader=test_loader,
                criterion=criterion,
                device=device,
            )

            history["train"].append({"epoch": epoch, **train_metrics})
            history["test"].append({"epoch": epoch, **test_metrics})

            logger.info(
                "Train | "
                f"loss={train_metrics['loss']:.4f} "
                f"acc={train_metrics['accuracy']:.4f} "
                f"bacc={train_metrics['balanced_accuracy']:.4f} "
                f"f1={train_metrics['macro_f1']:.4f}"
            )
            logger.info(
                "Test  | "
                f"loss={test_metrics['loss']:.4f} "
                f"acc={test_metrics['accuracy']:.4f} "
                f"bacc={test_metrics['balanced_accuracy']:.4f} "
                f"f1={test_metrics['macro_f1']:.4f}"
            )

            if test_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = test_metrics["macro_f1"]
                best_epoch = epoch
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "train_metrics": train_metrics,
                        "test_metrics": test_metrics,
                        "client_id": client_id,
                    },
                    best_ckpt_path,
                )
                logger.info(f"Saved best macro-F1 checkpoint to {best_ckpt_path}")

            with history_path.open("w", encoding="utf-8") as f:
                json.dump(history, f, indent=2)

        best_test_record = max(history["test"], key=lambda x: x["macro_f1"])
        final_train_record = history["train"][-1]
        final_test_record = history["test"][-1]

        summary = {
            "client_id": client_id,
            "epochs": epochs,
            "device": str(device),
            "best_epoch_by_macro_f1": best_epoch,
            "best_test_metrics_by_macro_f1": best_test_record,
            "final_train_metrics": final_train_record,
            "final_test_metrics": final_test_record,
            "checkpoint": str(best_ckpt_path),
        }

        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        client_results.append(
            {
                "client_id": client_id,
                "best_epoch": best_epoch,
                "best_accuracy": best_test_record["accuracy"],
                "best_balanced_accuracy": best_test_record["balanced_accuracy"],
                "best_macro_f1": best_test_record["macro_f1"],
                "final_accuracy": final_test_record["accuracy"],
                "final_balanced_accuracy": final_test_record["balanced_accuracy"],
                "final_macro_f1": final_test_record["macro_f1"],
            }
        )

        overall_logger.info(
            f"Finished client {client_id} | "
            f"best_macro_f1={best_test_record['macro_f1']:.4f} | "
            f"best_accuracy={best_test_record['accuracy']:.4f}"
        )

    df = pd.DataFrame(client_results).sort_values("client_id")
    df.to_csv(local_root / "client_results.csv", index=False)

    aggregate = {
        "mean_best_accuracy": float(df["best_accuracy"].mean()),
        "std_best_accuracy": float(df["best_accuracy"].std()),
        "mean_best_balanced_accuracy": float(df["best_balanced_accuracy"].mean()),
        "std_best_balanced_accuracy": float(df["best_balanced_accuracy"].std()),
        "mean_best_macro_f1": float(df["best_macro_f1"].mean()),
        "std_best_macro_f1": float(df["best_macro_f1"].std()),
    }

    with (local_root / "aggregate_results.json").open("w", encoding="utf-8") as f:
        json.dump(aggregate, f, indent=2)

    overall_logger.info("Local-only baseline completed")
    overall_logger.info(f"Aggregate results: {aggregate}")


if __name__ == "__main__":
    main()
