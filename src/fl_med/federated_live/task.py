"""Shared task code for REAL (networked) federated learning via Flower.

This reuses the EXACT same model, data loaders, training loop and metrics as the
in-process simulation (``fl_med.engine``). The only thing that changes is the
transport: clients now run as separate processes/machines and exchange model
parameters over the network instead of in a Python loop. A result produced here
is therefore directly comparable to the simulation -- it is the same federated
math, just genuinely distributed.

Torch is imported lazily so this module (and torch-free environments such as CI) can
import it without the GPU stack.
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Any, Dict, List

import numpy as np


# ---- parameter (state_dict) <-> list-of-ndarrays: Flower's wire format --------
def get_ndarrays(model) -> List[np.ndarray]:
    """Model weights as an ordered list of numpy arrays (Flower parameter format)."""
    return [v.detach().cpu().numpy() for v in model.state_dict().values()]


def set_ndarrays(model, ndarrays: List[np.ndarray]) -> None:
    """Load a Flower parameter list back into ``model`` (key order preserved)."""
    import torch

    keys = list(model.state_dict().keys())
    state = OrderedDict((k, torch.tensor(v)) for k, v in zip(keys, ndarrays))
    model.load_state_dict(state, strict=True)


# ---- model / criterion --------------------------------------------------------
def build_model(config: Dict[str, Any]):
    from ..models import build_model as _bm

    return _bm(config)


def _make_criterion(config: Dict[str, Any], train_loader, device):
    """Weighted CE from this client's LOCAL label counts (reusing the engine's
    ``build_criterion``) when ``loss.class_weights`` is 'balanced'/'inverse_frequency',
    else plain CE. Weighting is essential on Fed-ISIC2019: plain CE collapses to the
    majority class (balanced accuracy = 1/K). Only label counts are used -- shareable
    meta-information, never raw images."""
    mode = str((config.get("loss") or {}).get("class_weights", "none")).lower()
    if mode not in ("balanced", "inverse_frequency"):
        import torch.nn as nn

        return nn.CrossEntropyLoss()

    from ..losses import build_criterion

    num_classes = int(config.get("model", {}).get("num_classes", 8))
    ds = train_loader.dataset
    if hasattr(ds, "labels"):
        labels = ds.labels()
    elif hasattr(ds, "samples"):
        labels = [s[1] for s in ds.samples]
    else:
        labels = []
    if not labels:
        import torch.nn as nn

        return nn.CrossEntropyLoss()
    counts = np.zeros(num_classes, dtype=np.float64)
    for y in labels:
        counts[int(y)] += 1.0
    return build_criterion(config, counts, device=device)


# ---- local training / central evaluation (reuse the simulation's engine) ------
def _classifier_head(model):
    """The model's final classification layer (resnet: ``fc``; mobilenet/effnet:
    ``classifier``). Raises if neither exists so misuse fails loudly."""
    head = getattr(model, "fc", None)
    if head is None:
        head = getattr(model, "classifier", None)
    if head is None:
        raise AttributeError("model has neither .fc nor .classifier — cannot freeze backbone")
    return head


def apply_freeze_backbone(model):
    """Freeze every parameter except the classifier head (partial-model FL).

    Why: on a weak device (Raspberry Pi) the expensive part of a training step is
    backpropagation through the deep backbone. Freezing it means autograd only
    tracks the tiny head, so the backward pass all but disappears — a large
    speedup — while the ARCHITECTURE is unchanged, so FedAvg can still average
    this client's weights with everyone else's. The frozen backbone weights are
    returned unchanged (equal to the incoming global), i.e. the client simply
    abstains on the backbone and votes on the head. Returns (n_trainable, n_total).
    """
    head = _classifier_head(model)
    for p in model.parameters():
        p.requires_grad = False
    for p in head.parameters():
        p.requires_grad = True
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    return n_train, n_total


def local_fit(model, train_loader, config: Dict[str, Any], *, local_epochs: int,
              device: str, max_batches=None, freeze_backbone: bool = False) -> List[Dict[str, Any]]:
    """Train ``model`` on this client's local data with the SAME loop the
    simulation uses. ``max_batches`` caps steps/epoch so a slow device (the Pi)
    stays responsive; ``freeze_backbone`` trains only the classifier head
    (partial-model FL — big speedup on weak devices, architecture unchanged).
    Returns the per-epoch metric history."""
    from ..engine.client import _build_optimizer
    from ..engine.train_eval import GRAD_CLIP_NORM, train_one_epoch

    model.to(device)
    if freeze_backbone:
        apply_freeze_backbone(model)
        # optimizer over the trainable (head) parameters only
        opt_cfg = config.get("optimizer", {"name": "adam", "lr": 1e-3})
        import torch

        name = str(opt_cfg.get("name", "adam")).lower()
        lr = float(opt_cfg.get("lr", 1e-3))
        wd = float(opt_cfg.get("weight_decay", 0.0))
        params = [p for p in model.parameters() if p.requires_grad]
        if name == "sgd":
            optimizer = torch.optim.SGD(params, lr=lr,
                                        momentum=opt_cfg.get("momentum", 0.0), weight_decay=wd)
        else:
            optimizer = torch.optim.Adam(params, lr=lr, weight_decay=wd)
    else:
        optimizer = _build_optimizer(model, config.get("optimizer", {"name": "adam", "lr": 1e-3}))
    criterion = _make_criterion(config, train_loader, device)
    num_classes = int(config.get("model", {}).get("num_classes", 8))

    history: List[Dict[str, Any]] = []
    for _ in range(int(local_epochs)):
        m = train_one_epoch(
            model, train_loader, optimizer, device, criterion=criterion,
            num_classes=num_classes, max_batches=max_batches, grad_clip=GRAD_CLIP_NORM,
        )
        history.append(m)
    return history


def evaluate_model(model, test_loader, device: str, num_classes: int = 8) -> Dict[str, Any]:
    from ..engine.train_eval import evaluate

    return evaluate(model, test_loader, device, num_classes=num_classes)


def local_num_examples(train_loader, max_batches=None) -> int:
    """Images actually trained on this round (for FedAvg's sample weighting)."""
    n = len(train_loader.dataset)
    if max_batches is not None:
        bs = int(getattr(train_loader, "batch_size", 1) or 1)
        return min(n, int(max_batches) * bs)
    return n
