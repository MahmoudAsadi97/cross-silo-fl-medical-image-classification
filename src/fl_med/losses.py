"""Class-imbalance-aware loss weighting.

Fed-ISIC2019 is severely imbalanced (nevus dominates); an unweighted cross-entropy
collapses to predicting the majority class (balanced accuracy = 1/K). Inverse-
frequency class weights counter this and mirror FLamby's weighted-loss baseline.

Weights come from *label counts only* -- shareable meta-information, not raw
images -- so using global weights in FL does not leak patient data.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def inverse_frequency_weights(counts: Sequence[float], normalize: bool = True) -> List[float]:
    """w_c proportional to 1/count_c. Absent classes get weight 0.

    When ``normalize``, weights are scaled so the mean over present classes is 1,
    which keeps the loss magnitude comparable to the unweighted case.
    """
    counts = np.asarray(counts, dtype=float)
    weights = np.zeros(len(counts), dtype=float)
    present = counts > 0
    weights[present] = 1.0 / counts[present]
    if normalize and present.any():
        weights[present] *= present.sum() / weights[present].sum()
    return weights.tolist()


def build_criterion(config: dict, class_counts: Optional[Sequence[float]] = None, device: str = "cpu"):
    """CrossEntropyLoss, class-weighted when ``loss.class_weights`` is set + counts given."""
    import torch
    import torch.nn as nn

    mode = ((config.get("loss") or {}).get("class_weights") or "none")
    if mode in ("balanced", "inverse_frequency") and class_counts is not None:
        w = torch.tensor(inverse_frequency_weights(class_counts), dtype=torch.float32, device=device)
        return nn.CrossEntropyLoss(weight=w)
    return nn.CrossEntropyLoss()
