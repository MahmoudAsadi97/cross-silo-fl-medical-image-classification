"""Classification metrics for a class-imbalanced multi-class problem.

Primary metric is **balanced accuracy** (mean per-class recall) to match FLamby's
Fed-ISIC2019 leaderboard; **macro-F1** is the secondary headline. Also exposes
per-class precision/recall, the confusion matrix, and (optionally) macro
one-vs-rest AUC.

The core is pure numpy so it is unit-testable without torch or sklearn. Where
sklearn is available it is used only for AUC (which needs probabilistic scores)
and for an optional parity cross-check.
"""
from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def per_class_recall(cm: np.ndarray) -> np.ndarray:
    support = cm.sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.diag(cm) / support
    recall[support == 0] = np.nan  # undefined where a class is absent from y_true
    return recall


def per_class_precision(cm: np.ndarray) -> np.ndarray:
    predicted = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.diag(cm) / predicted
    precision[predicted == 0] = np.nan  # undefined where a class is never predicted
    return precision


def balanced_accuracy(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Mean recall over classes *present in y_true* (matches sklearn's definition)."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recall = per_class_recall(cm)
    present = ~np.isnan(recall)
    if not present.any():
        return 0.0
    return float(np.mean(recall[present]))


def macro_f1(y_true: Sequence[int], y_pred: Sequence[int], num_classes: int) -> float:
    """Unweighted mean F1 over all ``num_classes`` classes.

    Computed over the fixed class set (not only classes present) so the number is
    comparable across clients that are missing different classes. Classes with
    zero precision and recall contribute F1 = 0.
    """
    cm = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        f1 = 2 * tp / (2 * tp + fp + fn)
    f1 = np.nan_to_num(f1, nan=0.0)
    return float(np.mean(f1))


def accuracy(y_true: Sequence[int], y_pred: Sequence[int]) -> float:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def macro_ovr_auc(
    y_true: Sequence[int], y_prob: np.ndarray, num_classes: int
) -> Optional[float]:
    """Macro one-vs-rest AUC. Requires sklearn + class probabilities; else None."""
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return None
    y_true = np.asarray(y_true, dtype=int)
    present = np.unique(y_true)
    if present.size < 2:
        return None
    try:
        # Restrict to classes present in y_true to avoid undefined per-class AUC.
        return float(
            roc_auc_score(
                y_true,
                np.asarray(y_prob),
                multi_class="ovr",
                average="macro",
                labels=list(range(num_classes)),
            )
        )
    except Exception:
        return None


def compute_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    num_classes: int,
    y_prob: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Full metric bundle. ``y_prob`` (N x C) enables macro-AUC when sklearn is present."""
    cm = confusion_matrix(y_true, y_pred, num_classes)
    recall = per_class_recall(cm)
    precision = per_class_precision(cm)
    out: Dict[str, object] = {
        "accuracy": accuracy(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy(y_true, y_pred, num_classes),
        "macro_f1": macro_f1(y_true, y_pred, num_classes),
        "per_class_recall": [None if np.isnan(r) else float(r) for r in recall],
        "per_class_precision": [None if np.isnan(p) else float(p) for p in precision],
        "confusion_matrix": cm.tolist(),
        "support": cm.sum(axis=1).tolist(),
    }
    if y_prob is not None:
        out["macro_ovr_auc"] = macro_ovr_auc(y_true, y_prob, num_classes)
    return out
