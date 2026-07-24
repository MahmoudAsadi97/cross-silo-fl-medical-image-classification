#!/usr/bin/env python3
"""Phase 2/3 summary figure from experiments/comparison.json.

Left: balanced accuracy (best-epoch) per method with std error bars + the
majority-class floor (1/8). Right: final-round client drift per method (log scale;
baselines have no drift). Saved to reports/figures/phase23_comparison.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
NUM_CLASSES = 8
ORDER = ["local_only", "scaffold", "fedprox", "fedavg", "centralized"]
LABELS = {"local_only": "Local-only", "scaffold": "SCAFFOLD", "fedprox": "FedProx",
          "fedavg": "FedAvg", "centralized": "Centralized"}


def main(argv=None) -> int:
    tier = (argv or ["dev"])[0]
    comp = json.loads((REPO / "experiments" / "comparison.json").read_text())
    rows = {r["method"]: r for r in comp if r.get("tier") == tier}
    methods = [m for m in ORDER if m in rows]
    if not methods:
        print("no rows for tier", tier)
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    acc = [rows[m]["bal_acc_best_mean"] for m in methods]
    err = [rows[m].get("bal_acc_best_std", 0.0) for m in methods]
    drift = [rows[m].get("drift_final_mean") for m in methods]
    labels = [LABELS[m] for m in methods]
    colors = ["#7f7f7f", "#9467bd", "#2ca02c", "#1f77b4", "#d62728"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))

    ax1.bar(labels, acc, yerr=err, capsize=4, color=colors)
    ax1.axhline(1 / NUM_CLASSES, ls="--", color="k", lw=1,
                label=f"majority floor (1/{NUM_CLASSES})")
    ax1.set_ylabel("Balanced accuracy (best epoch)")
    ax1.set_title(f"Accuracy by method ({tier}, mean±std over seeds)")
    ax1.set_ylim(0, max(acc) * 1.25)
    ax1.legend(fontsize=8)
    ax1.tick_params(axis="x", rotation=20)

    fl = [(lab, d, c) for lab, d, c in zip(labels, drift, colors) if d and d == d]
    if fl:
        flab, fd, fc = zip(*fl)
        ax2.bar(flab, fd, color=fc)
        ax2.set_yscale("log")
        ax2.set_ylabel("Client drift (final round, log scale)")
        ax2.set_title("Client drift by strategy")
        ax2.tick_params(axis="x", rotation=20)
        for i, d in enumerate(fd):
            ax2.text(i, d, f"{d:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    out = REPO / "reports" / "figures" / f"phase23_comparison_{tier}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
