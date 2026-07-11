from __future__ import annotations

from collections import OrderedDict
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.data.loaders import build_client_dataloaders


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def load_state_dict_to_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, strict=True)


def compute_classification_metrics(y_true, y_pred):
    import numpy as np
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
    }


def train_fedprox_client(
    client_id: int,
    global_state_dict: Dict[str, torch.Tensor],
    model_builder,
    device,
    image_size: int,
    batch_size: int,
    num_workers: int,
    local_epochs: int,
    learning_rate: float,
    weight_decay: float,
    mu: float,
):
    model = model_builder().to(device)
    load_state_dict_to_model(model, global_state_dict)

    global_model = model_builder().to(device)
    load_state_dict_to_model(global_model, global_state_dict)
    global_model.eval()

    train_loader, _ = build_client_dataloaders(
        client_id=client_id,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    local_history: List[Dict] = []

    for local_epoch in range(1, local_epochs + 1):
        model.train()

        running_total_loss = 0.0
        running_ce_loss = 0.0
        running_prox_loss = 0.0
        all_targets = []
        all_predictions = []

        for batch in tqdm(train_loader, desc=f"FedProx Client {client_id}", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(images)
            ce_loss = criterion(outputs, labels)

            prox_term = 0.0
            for param, global_param in zip(model.parameters(), global_model.parameters()):
                prox_term += torch.sum((param - global_param.detach()) ** 2)

            prox_loss = 0.5 * mu * prox_term
            total_loss = ce_loss + prox_loss

            total_loss.backward()
            optimizer.step()

            running_total_loss += total_loss.item() * images.size(0)
            running_ce_loss += ce_loss.item() * images.size(0)
            running_prox_loss += prox_loss.item() * images.size(0)

            preds = torch.argmax(outputs, dim=1)
            all_targets.extend(labels.detach().cpu().tolist())
            all_predictions.extend(preds.detach().cpu().tolist())

        metrics = compute_classification_metrics(all_targets, all_predictions)
        metrics["loss"] = running_total_loss / len(train_loader.dataset)
        metrics["ce_loss"] = running_ce_loss / len(train_loader.dataset)
        metrics["prox_loss"] = running_prox_loss / len(train_loader.dataset)
        metrics["local_epoch"] = local_epoch

        local_history.append(metrics)

    updated_state = clone_state_dict(model.state_dict())
    num_samples = len(train_loader.dataset)

    return {
        "client_id": client_id,
        "num_samples": num_samples,
        "state_dict": updated_state,
        "local_history": local_history,
    }


def weighted_average_state_dicts(client_updates: List[Dict]) -> OrderedDict:
    if len(client_updates) == 0:
        raise ValueError("client_updates must not be empty")

    total_samples = sum(update["num_samples"] for update in client_updates)
    if total_samples <= 0:
        raise ValueError("Total client samples must be positive")

    first_state = client_updates[0]["state_dict"]
    aggregated = OrderedDict()

    for key in first_state.keys():
        weighted_sum = None
        for update in client_updates:
            client_tensor = update["state_dict"][key].float()
            weight = update["num_samples"] / total_samples
            contribution = client_tensor * weight

            if weighted_sum is None:
                weighted_sum = contribution
            else:
                weighted_sum += contribution

        aggregated[key] = weighted_sum

    return aggregated
