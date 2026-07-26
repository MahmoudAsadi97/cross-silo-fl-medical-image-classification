"""Membership-inference attack (loss-threshold, shadow-free).

A trained model tends to assign LOWER loss to samples it was trained on (members)
than to unseen samples (non-members); the gap is a privacy leak. The attack scores
each sample by ``-loss`` and measures how well that separates members from
non-members via ROC AUC. AUC ~ 0.5 means "no better than chance" (no leak); higher
means the model memorised its training data. DP-SGD is expected to push AUC toward
0.5 (brief §5, empirical privacy validation).

``roc_auc`` is a pure-numpy rank-based (Mann-Whitney) AUC so this verifies without
sklearn; ``per_sample_losses`` needs torch only when actually attacking a model.
"""
from __future__ import annotations

from typing import Dict, Optional

import numpy as np


def roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUC = P(score[member] > score[non-member]); ties count as 0.5 (average ranks)."""
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    s = scores[order]
    # average ranks for ties
    ranks_sorted = np.arange(1, len(scores) + 1, dtype=float)
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks_sorted[i:j + 1] = (i + 1 + j + 1) / 2.0
        i = j + 1
    ranks[order] = ranks_sorted
    n_pos = int(labels.sum())
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[labels == 1].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def per_sample_losses(model, loader, device, max_batches: Optional[int] = None) -> np.ndarray:
    """Cross-entropy loss for each sample in ``loader`` (model in eval mode)."""
    import torch
    import torch.nn as nn

    ce = nn.CrossEntropyLoss(reduction="none")
    model.eval()
    out = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if max_batches is not None and i >= max_batches:
                break
            x = batch["image"].to(device)
            y = batch["label"].to(device)
            out.append(ce(model(x), y).detach().cpu().numpy())
    return np.concatenate(out) if out else np.array([])


def attack_auc(member_losses: np.ndarray, nonmember_losses: np.ndarray) -> Dict[str, float]:
    """AUC of the loss-threshold attack + descriptive stats."""
    member_losses = np.asarray(member_losses, dtype=float)
    nonmember_losses = np.asarray(nonmember_losses, dtype=float)
    scores = np.concatenate([-member_losses, -nonmember_losses])   # lower loss -> more member-like
    labels = np.concatenate([np.ones(len(member_losses)), np.zeros(len(nonmember_losses))])
    return {
        "attack_auc": roc_auc(scores, labels),
        "n_member": int(len(member_losses)),
        "n_nonmember": int(len(nonmember_losses)),
        "member_loss_mean": float(np.mean(member_losses)) if len(member_losses) else float("nan"),
        "nonmember_loss_mean": float(np.mean(nonmember_losses)) if len(nonmember_losses) else float("nan"),
    }


def membership_inference(model, member_loader, nonmember_loader, device,
                         max_batches: Optional[int] = None) -> Dict[str, float]:
    """End-to-end: compute per-sample losses on members/non-members and return attack AUC."""
    m = per_sample_losses(model, member_loader, device, max_batches=max_batches)
    n = per_sample_losses(model, nonmember_loader, device, max_batches=max_batches)
    return attack_auc(m, n)
