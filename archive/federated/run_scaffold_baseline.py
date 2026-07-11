from __future__ import annotations

import json
from pathlib import Path
import sys
import random
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.resnet import build_resnet18
from src.data.loaders import build_centralized_dataloaders
from src.fl.scaffold import train_scaffold_client, clone_state_dict, zero_like
from src.training.engine import evaluate


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_yaml(PROJECT_ROOT / "configs/fl/scaffold.yaml")
    model_cfg = load_yaml(PROJECT_ROOT / "configs/model/resnet18.yaml")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_clients = cfg["num_clients"]
    rounds = cfg["rounds"]

    def model_builder():
        return build_resnet18(num_classes=model_cfg["num_classes"], pretrained=True)

    global_model = model_builder().to(device)
    global_state = clone_state_dict(global_model.state_dict())

    # Control variates
    c_global = zero_like(global_state)
    c_locals = [zero_like(global_state) for _ in range(num_clients)]

    _, test_loader = build_centralized_dataloaders(
        image_size=224,
        batch_size=32,
        num_workers=4,
    )

    history = []
    best_f1 = 0

    for r in range(1, rounds + 1):
        updates = []

        for client_id in range(num_clients):
            new_state, new_c = train_scaffold_client(
                client_id,
                global_state,
                c_global,
                c_locals[client_id],
                model_builder,
                device,
                224,
                32,
                4,
                1,
                0.001,
                0.0001,
            )

            updates.append(new_state)
            c_locals[client_id] = new_c

        # Aggregate weights
        new_global = clone_state_dict(global_state)
        for k in new_global.keys():
            new_global[k] = sum(u[k] for u in updates) / len(updates)

        global_model.load_state_dict(new_global)
        global_state = clone_state_dict(new_global)

        metrics = evaluate(global_model, test_loader, torch.nn.CrossEntropyLoss(), device)

        history.append(metrics)

        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]

        print(f"Round {r}: F1={metrics['macro_f1']:.4f}")

    print("Best F1:", best_f1)


if __name__ == "__main__":
    main()
