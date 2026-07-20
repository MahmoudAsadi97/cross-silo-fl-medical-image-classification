"""Shared local train / eval loops used identically by every strategy and baseline.

The single training loop honours the strategy hooks (FedProx's ``extra_loss`` and
SCAFFOLD's ``after_backward`` gradient correction), so all methods share exactly
the same optimization code -- part of the fair-comparison protocol.

A generous gradient-norm clip (``GRAD_CLIP_NORM``) is applied after the strategy's
gradient edit as a numerical safety net: it never triggers for normal training
(grad norms are O(1-10)) but prevents a runaway (e.g. mis-tuned SCAFFOLD control
variates) from reaching nan and corrupting a run.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from ..metrics import compute_metrics

GRAD_CLIP_NORM = 1e3


def train_one_epoch(
    model,
    loader,
    optimizer,
    device,
    *,
    criterion=None,
    strategy=None,
    global_model=None,
    client_state: Optional[Dict[str, Any]] = None,
    num_classes: int = 8,
    max_batches: Optional[int] = None,
) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    criterion = criterion or nn.CrossEntropyLoss()
    model.train()
    running_loss, seen = 0.0, 0
    y_true, y_pred = [], []

    for i, batch in enumerate(loader):
        if max_batches is not None and i >= max_batches:
            break
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        if strategy is not None:
            extra = strategy.extra_loss(model, global_model)
            if not isinstance(extra, float) or extra != 0.0:
                loss = loss + extra
        loss.backward()
        if strategy is not None:
            strategy.after_backward(model, client_state)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=GRAD_CLIP_NORM)
        optimizer.step()

        running_loss += float(loss.item()) * images.size(0)
        seen += images.size(0)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())

    metrics = compute_metrics(y_true, y_pred, num_classes)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics


def evaluate(model, loader, device, *, criterion=None, num_classes: int = 8) -> Dict[str, Any]:
    import torch
    import torch.nn as nn

    criterion = criterion or nn.CrossEntropyLoss()
    model.eval()
    running_loss, seen = 0.0, 0
    y_true, y_pred, probs = [], [], []

    with torch.no_grad():
        for batch in loader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += float(loss.item()) * images.size(0)
            seen += images.size(0)
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(torch.argmax(outputs, dim=1).detach().cpu().tolist())
            probs.append(torch.softmax(outputs, dim=1).detach().cpu().numpy())

    y_prob = np.concatenate(probs, axis=0) if probs else None
    metrics = compute_metrics(y_true, y_pred, num_classes, y_prob=y_prob)
    metrics["loss"] = running_loss / max(seen, 1)
    return metrics
