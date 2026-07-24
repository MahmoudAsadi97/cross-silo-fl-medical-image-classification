#!/usr/bin/env python3
"""Privacy-utility curve from the DP sweep (Phase 4).

Reads experiments/dp_none_* (matched non-private baseline) and experiments/dp_s*_*
(DP at several noise multipliers), and plots balanced accuracy vs. the per-client
max epsilon. Saved to reports/figures/dp_privacy_utility_<tier>.png.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _summary(path: Path):
    f = path / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def main(argv=None) -> int:
    tier = (argv or ["dev"])[0]
    exp = REPO / "experiments"

    baseline = None
    for d in exp.glob(f"dp_none_{tier}_seed*"):
        s = _summary(d)
        if s:
            baseline = s.get("test_balanced_accuracy")
            break

    pts = []  # (epsilon, balanced_acc, sigma)
    for d in sorted(exp.glob(f"dp_s*_{tier}_seed*")):
        s = _summary(d)
        if not s or "epsilon_max" not in s:
            continue
        sigma = d.name.split("_")[1][1:]
        pts.append((s["epsilon_max"], s["test_balanced_accuracy"], sigma))
    pts.sort()

    if not pts:
        print("no DP runs found; run scripts/run_dp_sweep.sh first")
        return 1

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    eps = [p[0] for p in pts]
    acc = [p[1] for p in pts]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.plot(eps, acc, "o-", color="#1f77b4", label="DP-SGD (FedAvg)")
    for e, a, sig in pts:
        ax.annotate(f"σ={sig}", (e, a), textcoords="offset points", xytext=(5, 5), fontsize=8)
    if baseline is not None:
        ax.axhline(baseline, ls="--", color="#d62728", label=f"non-private ({baseline:.3f})")
    ax.axhline(1 / 8, ls=":", color="k", lw=1, label="majority floor (1/8)")
    ax.set_xlabel("Privacy budget  ε (per-client max, δ=1e-5)")
    ax.set_ylabel("Balanced accuracy")
    ax.set_title(f"Privacy–utility trade-off ({tier})")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out = REPO / "reports" / "figures" / f"dp_privacy_utility_{tier}.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
