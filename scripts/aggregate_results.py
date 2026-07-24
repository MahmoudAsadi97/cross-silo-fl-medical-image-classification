#!/usr/bin/env python3
"""Aggregate per-seed experiment runs into a comparison table (mean +/- std).

For each ``experiments/<name>_<tier>_seed<k>/`` run it reads the per-round
``metrics.csv`` and reports BOTH the best-epoch and final-epoch balanced accuracy
/ macro-F1 (best is more representative than the noisy final round), plus the
final-round client drift for federated methods. Local-only (no metrics.csv) is
summarised as the mean over its per-client final metrics. Groups by (name, tier),
reports mean/std across seeds, and optionally a paired Wilcoxon between two methods.

Note on "best": chosen on the test set (no materialized val split), so it is a
mild optimistic estimate; final is reported alongside for honesty.
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.eval import paired_wilcoxon, save_json, summarize_seeds  # noqa: E402

RUN_RE = re.compile(r"^(?P<name>.+)_(?P<tier>smoke|dev|full)_seed(?P<seed>\d+)$")


def _read_metrics_csv(path: Path):
    with path.open() as f:
        return list(csv.DictReader(f))


def _run_values(run: Path):
    """Return per-run scalars: best/final balanced-acc + macro-f1, final drift."""
    metrics_csv = run / "metrics.csv"
    summary = run / "summary.json"
    if metrics_csv.exists():
        rows = _read_metrics_csv(metrics_csv)
        if not rows:
            return None
        ba = [float(r["test_balanced_accuracy"]) for r in rows]
        f1 = [float(r["test_macro_f1"]) for r in rows]
        drift = rows[-1].get("client_drift")
        return {
            "best_ba": max(ba), "final_ba": ba[-1],
            "best_f1": max(f1), "final_f1": f1[-1],
            "drift": float(drift) if drift not in (None, "") else None,
        }
    if summary.exists():
        data = json.loads(summary.read_text())
        pcs = data.get("per_client")
        if pcs:  # local-only: mean over clients of their final metrics
            ba = [float(m["test_balanced_accuracy"]) for m in pcs.values()]
            f1 = [float(m.get("test_macro_f1", "nan")) for m in pcs.values()]
            mean_ba, mean_f1 = float(np.mean(ba)), float(np.nanmean(f1))
            return {"best_ba": mean_ba, "final_ba": mean_ba,
                    "best_f1": mean_f1, "final_f1": mean_f1, "drift": None}
    return None


def collect(exp_dir: Path, tier_filter=None):
    groups = defaultdict(lambda: defaultdict(list))
    for run in sorted(exp_dir.glob("*_seed*")):
        m = RUN_RE.match(run.name)
        if not m:
            continue
        if tier_filter and m["tier"] != tier_filter:
            continue
        vals = _run_values(run)
        if not vals:
            continue
        key = (m["name"], m["tier"])
        for k, v in vals.items():
            if v is not None:
                groups[key][k].append(v)
    return groups


ORDER = ["centralized", "fedavg", "fedprox", "scaffold", "local_only"]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default=str(REPO / "experiments"))
    p.add_argument("--tier", default="dev")
    p.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = p.parse_args(argv)

    groups = collect(Path(args.exp_dir), tier_filter=args.tier)
    if not groups:
        print(f"No runs for tier={args.tier} under {args.exp_dir}")
        return 0

    def sort_key(item):
        name = item[0][0]
        return (ORDER.index(name) if name in ORDER else 99, name)

    rows = []
    print(f"\n=== Comparison (tier={args.tier}, mean +/- std over seeds) ===")
    print(f"{'method':<13}{'n':<3}{'bal_acc best':<18}{'bal_acc final':<18}"
          f"{'macro_f1 best':<18}{'drift(final)':<12}")
    for (name, tier), vals in sorted(groups.items(), key=sort_key):
        bb, fb = summarize_seeds(vals["best_ba"]), summarize_seeds(vals["final_ba"])
        bf = summarize_seeds(vals["best_f1"])
        drift = summarize_seeds(vals["drift"]) if vals.get("drift") else {"mean": float("nan")}
        dtxt = "-" if np.isnan(drift["mean"]) else f"{drift['mean']:.2f}"
        print(f"{name:<13}{bb['n']:<3}{bb['mean']:.4f}+/-{bb['std']:.4f}    "
              f"{fb['mean']:.4f}+/-{fb['std']:.4f}    "
              f"{bf['mean']:.4f}+/-{bf['std']:.4f}    {dtxt}")
        rows.append({"method": name, "tier": tier, "n_seeds": bb["n"],
                     "bal_acc_best_mean": bb["mean"], "bal_acc_best_std": bb["std"],
                     "bal_acc_final_mean": fb["mean"], "bal_acc_final_std": fb["std"],
                     "macro_f1_best_mean": bf["mean"], "drift_final_mean": drift["mean"]})
    save_json(rows, Path(args.exp_dir) / "comparison.json")

    if args.compare:
        a, b = args.compare
        va = groups.get((a, args.tier), {}).get("best_ba")
        vb = groups.get((b, args.tier), {}).get("best_ba")
        if va and vb and len(va) == len(vb):
            res = paired_wilcoxon(va, vb)
            print(f"\nPaired Wilcoxon {a} vs {b} (best bal_acc): "
                  f"mean_diff={res['mean_diff']:.4f} p={res['p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
