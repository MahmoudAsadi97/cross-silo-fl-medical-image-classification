from __future__ import annotations

from collections import OrderedDict
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn
import torch.optim as optim

from src.data.loaders import build_client_dataloaders
from src.training.engine import train_one_epoch


def clone_state_dict(state_dict: Dict[str, torch.Tensor]) -> OrderedDict:
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def load_state_dict_to_model(model: nn.Module, state_dict: Dict[str, torch.Tensor]) -> None:
    model.load_state_dict(state_dict, strict=True)


def train_fedavg_client(
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
):
    model = model_builder().to(device)
    load_state_dict_to_model(model, global_state_dict)

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
        train_metrics = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )
        local_history.append(
            {
                "local_epoch": local_epoch,
                **train_metrics,
            }
        )

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
