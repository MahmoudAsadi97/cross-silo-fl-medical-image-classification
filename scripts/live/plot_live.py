#!/usr/bin/env python3
"""Figures for the REAL distributed run (reads experiments/live/history.json):

  1. live_accuracy.png  -- central balanced accuracy vs federated round.
  2. live_straggler.png -- each client's local fit time per round. A slow edge
     device (the Pi) shows up as the tall line: FedAvg is synchronous, so the
     round is gated by the slowest client -- the classic straggler problem.

Torch-free (numpy + matplotlib only), so it runs anywhere.

    python scripts/live/plot_live.py [path/to/history.json]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    src = Path(argv[0]) if argv else REPO / "experiments" / "live" / "history.json"
    if not src.exists():
        print(f"no history at {src} -- run a live demo first")
        return 1
    data = json.loads(src.read_text())
    history, timings = data.get("history", []), data.get("timings", [])

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figdir = REPO / "reports" / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # (1) accuracy vs round -----------------------------------------------------
    if history:
        rounds = [h["round"] for h in history]
        bal = [h["bal_acc"] for h in history]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(rounds, bal, "o-", color="#1f6fb2", label="global model (central eval)")
        ax.axhline(1 / 8, ls=":", color="k", lw=1, label="majority floor (1/8)")
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Balanced accuracy")
        ax.set_title("Real distributed FedAvg over the network")
        ax.legend()
        fig.tight_layout()
        fig.savefig(figdir / "live_accuracy.png", dpi=130)
        plt.close(fig)

    # (2) per-client fit time per round (straggler view) ------------------------
    if timings:
        by_tag = defaultdict(lambda: ([], []))
        for t in timings:
            xs, ys = by_tag[t.get("tag", "?")]
            xs.append(t.get("round"))
            ys.append(t.get("fit_seconds", float("nan")))
        fig, ax = plt.subplots(figsize=(6, 4))
        for tag, (xs, ys) in sorted(by_tag.items()):
            ax.plot(xs, ys, "o-", label=tag)
        ax.set_xlabel("Federated round")
        ax.set_ylabel("Local fit time (s)")
        ax.set_title("Per-client compute per round (straggler view)")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(figdir / "live_straggler.png", dpi=130)
        plt.close(fig)

        # one-line summary of the straggler gap
        last_by_tag = {}
        for t in timings:
            last_by_tag[t.get("tag", "?")] = t.get("fit_seconds", float("nan"))
        if last_by_tag:
            slow = max(last_by_tag, key=last_by_tag.get)
            fast = min(last_by_tag, key=last_by_tag.get)
            if last_by_tag[fast] > 0:
                ratio = last_by_tag[slow] / last_by_tag[fast]
                print(f"straggler: '{slow}' {last_by_tag[slow]:.1f}s vs "
                      f"'{fast}' {last_by_tag[fast]:.1f}s  ({ratio:.1f}x slower)")

    print("wrote reports/figures/live_accuracy.png, reports/figures/live_straggler.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
