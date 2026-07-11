from __future__ import annotations

import json
import random
from pathlib import Path
import sys

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import build_centralized_dataloaders
from src.evaluation.metrics import compute_classification_metrics
from src.fl.fedavg import train_fedavg_client, weighted_average_state_dicts
from src.models.resnet import build_resnet18
from src.training.engine import evaluate
from src.utils.logger import get_logger
from src.utils.reproducibility import set_seed


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def select_clients(num_clients: int, participation_fraction: float):
    all_clients = list(range(num_clients))
    num_selected = max(1, int(round(num_clients * participation_fraction)))
    return sorted(random.sample(all_clients, num_selected))


def main():
    fl_cfg = load_yaml(PROJECT_ROOT / "configs/fl/fedavg.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs/model/resnet18.yaml")
    base_cfg = load_yaml(PROJECT_ROOT / "configs/experiments/base_experiment.yaml")

    seed = int(fl_cfg.get("seed", base_cfg["experiment"]["seed"]))
    set_seed(seed, deterministic=bool(base_cfg["experiment"]["deterministic"]))

    device_name = fl_cfg.get("device", "cuda")
    if device_name == "cuda" and not torch.cuda.is_available():
        device_name = "cpu"
    device = torch.device(device_name)

    num_clients = int(fl_cfg["num_clients"])
    participation_fraction = float(fl_cfg["participation_fraction"])
    rounds = int(fl_cfg["rounds"])
    local_epochs = int(fl_cfg["local_epochs"])
    batch_size = int(fl_cfg["batch_size"])
    lr = float(fl_cfg["learning_rate"])
    weight_decay = float(fl_cfg["weight_decay"])

    num_workers = int(base_cfg["experiment"]["num_workers"])
    image_size = int(base_cfg["data"]["image_size"])
    num_classes = int(model_cfg["num_classes"])

    run_dir = PROJECT_ROOT / "results" / "fedavg_baseline"
    ckpt_dir = run_dir / "checkpoints"
    log_dir = run_dir / "logs"
    history_path = run_dir / "history.json"
    summary_path = run_dir / "summary.json"

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = get_logger("fedavg_baseline", str(log_dir / "train.log"))
    logger.info("Starting FedAvg baseline")
    logger.info(f"Using device: {device}")

    _, global_test_loader = build_centralized_dataloaders(
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    def model_builder():
        return build_resnet18(
            num_classes=num_classes,
            pretrained=bool(model_cfg["pretrained"]),
        )

    global_model = model_builder().to(device)
    global_state_dict = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

    history = []
    best_macro_f1 = -1.0
    best_round = None
    best_checkpoint_path = ckpt_dir / "best_model_by_macro_f1.pt"

    for round_idx in range(1, rounds + 1):
        selected_clients = select_clients(num_clients, participation_fraction)
        logger.info(f"Round {round_idx}/{rounds} | selected_clients={selected_clients}")

        client_updates = []
        round_client_logs = []

        for client_id in selected_clients:
            logger.info(f"Training client {client_id}")

            client_result = train_fedavg_client(
                client_id=client_id,
                global_state_dict=global_state_dict,
                model_builder=model_builder,
                device=device,
                image_size=image_size,
                batch_size=batch_size,
                num_workers=num_workers,
                local_epochs=local_epochs,
                learning_rate=lr,
                weight_decay=weight_decay,
            )

            client_updates.append(client_result)

            final_local_metrics = client_result["local_history"][-1]
            round_client_logs.append(
                {
                    "client_id": client_id,
                    "num_samples": client_result["num_samples"],
                    "final_local_metrics": final_local_metrics,
                }
            )

            logger.info(
                f"Client {client_id} | samples={client_result['num_samples']} | "
                f"local_loss={final_local_metrics['loss']:.4f} | "
                f"local_acc={final_local_metrics['accuracy']:.4f} | "
                f"local_bacc={final_local_metrics['balanced_accuracy']:.4f} | "
                f"local_f1={final_local_metrics['macro_f1']:.4f}"
            )

        aggregated_state = weighted_average_state_dicts(client_updates)
        global_model.load_state_dict(aggregated_state, strict=True)
        global_state_dict = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}

        global_metrics = evaluate(
            model=global_model,
            dataloader=global_test_loader,
            criterion=torch.nn.CrossEntropyLoss(),
            device=device,
        )

        round_record = {
            "round": round_idx,
            "selected_clients": selected_clients,
            "num_selected_clients": len(selected_clients),
            "global_test_metrics": global_metrics,
            "client_logs": round_client_logs,
        }
        history.append(round_record)

        logger.info(
            "Global Test | "
            f"loss={global_metrics['loss']:.4f} "
            f"acc={global_metrics['accuracy']:.4f} "
            f"bacc={global_metrics['balanced_accuracy']:.4f} "
            f"f1={global_metrics['macro_f1']:.4f}"
        )

        if global_metrics["macro_f1"] > best_macro_f1:
            best_macro_f1 = global_metrics["macro_f1"]
            best_round = round_idx

            torch.save(
                {
                    "round": round_idx,
                    "model_state_dict": global_model.state_dict(),
                    "global_test_metrics": global_metrics,
                    "selected_clients": selected_clients,
                },
                best_checkpoint_path,
            )
            logger.info(f"Saved best macro-F1 checkpoint to {best_checkpoint_path}")

        with history_path.open("w", encoding="utf-8") as f:
            json.dump(history, f, indent=2)

    final_metrics = history[-1]["global_test_metrics"] if history else None

    summary = {
        "run_name": "fedavg_baseline",
        "device": str(device),
        "num_clients": num_clients,
        "participation_fraction": participation_fraction,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": weight_decay,
        "seed": seed,
        "best_round_by_macro_f1": best_round,
        "best_macro_f1": best_macro_f1,
        "best_checkpoint": str(best_checkpoint_path),
        "final_global_test_metrics": final_metrics,
    }

    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    logger.info("FedAvg training completed")
    logger.info(f"Best macro-F1: {best_macro_f1:.4f} at round {best_round}")
    logger.info(f"History saved to: {history_path}")
    logger.info(f"Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
