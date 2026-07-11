from __future__ import annotations

from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim

from src.data.loaders import build_client_dataloaders


def clone_state_dict(state_dict):
    return OrderedDict((k, v.detach().cpu().clone()) for k, v in state_dict.items())


def zero_like(state_dict):
    return OrderedDict((k, torch.zeros_like(v)) for k, v in state_dict.items())


def train_scaffold_client(
    client_id,
    global_state,
    c_global,
    c_local,
    model_builder,
    device,
    image_size,
    batch_size,
    num_workers,
    local_epochs,
    lr,
    weight_decay,
):

    model = model_builder().to(device)
    model.load_state_dict(global_state)

    train_loader, _ = build_client_dataloaders(
        client_id=client_id,
        image_size=image_size,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    for _ in range(local_epochs):
        for batch in train_loader:
            x = batch["image"].to(device)
            y = batch["label"].to(device)

            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()

            # Apply SCAFFOLD correction
            with torch.no_grad():
                for name, param in model.named_parameters():
                    param.grad += (c_global[name].to(device) - c_local[name].to(device))

            optimizer.step()

    new_state = clone_state_dict(model.state_dict())

    # Update control variates
    delta = OrderedDict()
    for k in global_state.keys():
        delta[k] = new_state[k] - global_state[k]

    new_c_local = OrderedDict()
    for k in c_local.keys():
        new_c_local[k] = c_local[k] - c_global[k] + delta[k]

    return new_state, new_c_local
