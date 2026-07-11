"""Result artifacts: persistence, multi-seed aggregation, stats, and plots.

Kept dependency-light: json/csv persistence and numpy stats always work; scipy
(Wilcoxon) and matplotlib (plots) are used when present and degrade gracefully.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


def save_json(obj: Any, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
    return path


def save_history_csv(history: List[Dict[str, Any]], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not history:
        path.write_text("")
        return path
    keys = list(history[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(history)
    return path


def summarize_seeds(values: Sequence[float]) -> Dict[str, float]:
    """mean / std / n / 95% CI half-width for a list of per-seed scalars."""
    arr = np.asarray(list(values), dtype=float)
    n = arr.size
    if n == 0:
        return {"mean": float("nan"), "std": float("nan"), "n": 0}
    std = float(np.std(arr, ddof=1)) if n > 1 else 0.0
    ci = 1.96 * std / np.sqrt(n) if n > 1 else 0.0
    return {"mean": float(np.mean(arr)), "std": std, "n": int(n), "ci95": float(ci)}


def paired_wilcoxon(a: Sequence[float], b: Sequence[float]) -> Dict[str, Optional[float]]:
    """Paired Wilcoxon signed-rank test across seeds + a simple effect size."""
    a = np.asarray(list(a), dtype=float)
    b = np.asarray(list(b), dtype=float)
    out: Dict[str, Optional[float]] = {
        "mean_diff": float(np.mean(a - b)) if a.size else None,
        "n": int(a.size),
        "statistic": None,
        "p_value": None,
    }
    try:
        from scipy.stats import wilcoxon

        if a.size >= 1 and np.any(a - b != 0):
            stat, p = wilcoxon(a, b)
            out["statistic"], out["p_value"] = float(stat), float(p)
    except Exception:
        pass
    return out


def plot_curves(
    history: List[Dict[str, Any]], x: str, ys: Sequence[str], path: str | Path,
    title: str = "", xlabel: str = "", ylabel: str = "",
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [row[x] for row in history]
    for y in ys:
        ax.plot(xs, [row.get(y) for row in history], marker="o", label=y)
    ax.set_title(title or ", ".join(ys))
    ax.set_xlabel(xlabel or x)
    ax.set_ylabel(ylabel)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def plot_confusion_matrix(
    cm, class_names: Optional[Sequence[str]], path: str | Path, title: str = "Confusion matrix",
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None
    cm = np.asarray(cm, dtype=float)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ticks = range(cm.shape[0])
    labels = class_names if class_names else [str(i) for i in ticks]
    ax.set_xticks(list(ticks))
    ax.set_yticks(list(ticks))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in ticks:
        for j in ticks:
            ax.text(
                j, i, int(cm[i, j]), ha="center", va="center", fontsize=6,
                color="white" if cm[i, j] > thresh else "black",
            )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
