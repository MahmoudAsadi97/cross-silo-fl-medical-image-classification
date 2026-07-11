from __future__ import annotations

import json
import shutil
from pathlib import Path
import sys

import torch
import torch.nn as nn
import torch.optim as optim
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import build_centralized_dataloaders
from src.models.resnet import build_resnet18
from src.training.engine import train_one_epoch, evaluate
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_scheduler(optimizer, scheduler_cfg):
    if not scheduler_cfg:
        return None

    name = scheduler_cfg.get("name", "").lower()
    if name == "reduce_on_plateau":
        mode = scheduler_cfg.get("mode", "max")
        factor = float(scheduler_cfg.get("factor", 0.5))
        patience = int(scheduler_cfg.get("patience", 2))
        min_lr = float(scheduler_cfg.get("min_lr", 1e-6))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            min_lr=min_lr,
        )

    raise ValueError(f"Unsupported scheduler: {name}")


def get_current_lr(optimizer):
    return float(optimizer.param_groups[0]["lr"])


def is_improvement(current, best, mode: str, min_delta: float) -> bool:
    if best is None:
        return True
    if mode == "max":
        return current > (best + min_delta)
    if mode == "min":
        return current < (best - min_delta)
    raise ValueError(f"Unsupported mode: {mode}")


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

    scheduler_cfg = training_cfg.get("scheduler", {})
    early_cfg = training_cfg.get("early_stopping", {})
    early_enabled = bool(early_cfg.get("enabled", False))
    early_monitor = early_cfg.get("monitor", "macro_f1")
    early_mode = early_cfg.get("mode", "max")
    early_patience = int(early_cfg.get("patience", 5))
    early_min_delta = float(early_cfg.get("min_delta", 0.001))

    run_dir = PROJECT_ROOT / "results" / "centralized_baseline"

    if run_dir.exists():
        shutil.rmtree(run_dir)

    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    plot_dir = run_dir / "plots"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    plot_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("centralized_baseline", str(log_dir / "train.log"))
    logger.info("Starting centralized baseline training")
    logger.info(f"Using device: {device}")
    logger.info("Old centralized results were removed and will be replaced.")

    train_loader, test_loader = build_centralized_dataloaders(
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
    scheduler = build_scheduler(optimizer, scheduler_cfg)

    history = {
        "train": [],
        "test": [],
    }

    best_test_acc = -1.0
    best_test_macro_f1 = -1.0
    best_test_balanced_acc = -1.0

    best_acc_checkpoint_path = ckpt_dir / "best_model_by_accuracy.pt"
    best_f1_checkpoint_path = ckpt_dir / "best_model_by_macro_f1.pt"
    best_bacc_checkpoint_path = ckpt_dir / "best_model_by_balanced_accuracy.pt"

    best_acc_epoch = None
    best_f1_epoch = None
    best_bacc_epoch = None

    early_best_value = None
    early_bad_epochs = 0
    stopped_early = False
    stopped_epoch = None

    for epoch in range(1, epochs + 1):
        current_lr = get_current_lr(optimizer)
        logger.info(f"Epoch {epoch}/{epochs} | lr={current_lr:.8f}")

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

        history["train"].append({"epoch": epoch, "lr": current_lr, **train_metrics})
        history["test"].append({"epoch": epoch, "lr": current_lr, **test_metrics})

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

        checkpoint_payload = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "train_metrics": train_metrics,
            "test_metrics": test_metrics,
        }

        if test_metrics["accuracy"] > best_test_acc:
            best_test_acc = test_metrics["accuracy"]
            best_acc_epoch = epoch
            torch.save(checkpoint_payload, best_acc_checkpoint_path)
            logger.info(f"Saved best accuracy checkpoint to {best_acc_checkpoint_path}")

        if test_metrics["macro_f1"] > best_test_macro_f1:
            best_test_macro_f1 = test_metrics["macro_f1"]
            best_f1_epoch = epoch
            torch.save(checkpoint_payload, best_f1_checkpoint_path)
            logger.info(f"Saved best macro-F1 checkpoint to {best_f1_checkpoint_path}")

        if test_metrics["balanced_accuracy"] > best_test_balanced_acc:
            best_test_balanced_acc = test_metrics["balanced_accuracy"]
            best_bacc_epoch = epoch
            torch.save(checkpoint_payload, best_bacc_checkpoint_path)
            logger.info(f"Saved best balanced-accuracy checkpoint to {best_bacc_checkpoint_path}")

        monitor_name = scheduler_cfg.get("monitor", "macro_f1")
        if scheduler is not None:
            scheduler_value = float(test_metrics[monitor_name])
            scheduler.step(scheduler_value)
            logger.info(
                f"Scheduler step on {monitor_name}={scheduler_value:.4f} | new_lr={get_current_lr(optimizer):.8f}"
            )

        if early_enabled:
            monitored_value = float(test_metrics[early_monitor])
            if is_improvement(monitored_value, early_best_value, early_mode, early_min_delta):
                early_best_value = monitored_value
                early_bad_epochs = 0
                logger.info(
                    f"Early stopping monitor improved: {early_monitor}={monitored_value:.4f}"
                )
            else:
                early_bad_epochs += 1
                logger.info(
                    f"No significant improvement in {early_monitor}. "
                    f"bad_epochs={early_bad_epochs}/{early_patience}"
                )

            if early_bad_epochs >= early_patience:
                stopped_early = True
                stopped_epoch = epoch
                logger.info(
                    f"Early stopping triggered at epoch {epoch} based on {early_monitor}"
                )
                with history_path.open("w", encoding="utf-8") as f:
                    json.dump(history, f, indent=2)
                break

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    summary = {
        "run_name": "centralized_baseline",
        "device": str(device),
        "epochs_requested": epochs,
        "epochs_completed": len(history["train"]),
        "batch_size": batch_size,
        "image_size": image_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "scheduler": scheduler_cfg,
        "early_stopping": {
            "enabled": early_enabled,
            "monitor": early_monitor,
            "mode": early_mode,
            "patience": early_patience,
            "min_delta": early_min_delta,
            "stopped_early": stopped_early,
            "stopped_epoch": stopped_epoch,
        },
        "best_by_accuracy": {
            "epoch": best_acc_epoch,
            "accuracy": best_test_acc,
            "checkpoint": str(best_acc_checkpoint_path),
        },
        "best_by_macro_f1": {
            "epoch": best_f1_epoch,
            "macro_f1": best_test_macro_f1,
            "checkpoint": str(best_f1_checkpoint_path),
        },
        "best_by_balanced_accuracy": {
            "epoch": best_bacc_epoch,
            "balanced_accuracy": best_test_balanced_acc,
            "checkpoint": str(best_bacc_checkpoint_path),
        },
        "final_train_metrics": history["train"][-1] if history["train"] else None,
        "final_test_metrics": history["test"][-1] if history["test"] else None,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("Training completed")
    logger.info(f"Best test accuracy: {best_test_acc:.4f} at epoch {best_acc_epoch}")
    logger.info(f"Best test macro-F1: {best_test_macro_f1:.4f} at epoch {best_f1_epoch}")
    logger.info(f"Best test balanced accuracy: {best_test_balanced_acc:.4f} at epoch {best_bacc_epoch}")
    logger.info(f"History saved to: {history_path}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
