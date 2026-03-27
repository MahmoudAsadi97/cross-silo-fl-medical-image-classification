from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from tqdm import tqdm

from src.evaluation.metrics import compute_classification_metrics


def train_one_epoch(
    model,
    dataloader,
    criterion,
    optimizer,
    device,
) -> Dict[str, float]:
    model.train()

    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for batch in tqdm(dataloader, desc="Training", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_predictions)
    metrics["loss"] = epoch_loss
    return metrics


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    criterion,
    device,
) -> Dict[str, float]:
    model.eval()

    running_loss = 0.0
    all_targets: List[int] = []
    all_predictions: List[int] = []

    for batch in tqdm(dataloader, desc="Evaluating", leave=False):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)

        running_loss += loss.item() * images.size(0)

        preds = torch.argmax(outputs, dim=1)
        all_targets.extend(labels.detach().cpu().tolist())
        all_predictions.extend(preds.detach().cpu().tolist())

    epoch_loss = running_loss / len(dataloader.dataset)
    metrics = compute_classification_metrics(all_targets, all_predictions)
    metrics["loss"] = epoch_loss
    return metrics
