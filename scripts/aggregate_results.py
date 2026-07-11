#!/usr/bin/env python3
"""Aggregate per-seed experiment runs into a comparison table (mean +/- std).

Scans ``experiments/<name>_<tier>_seed<k>/summary.json``, groups by (name, tier),
and reports mean/std/95%CI of the headline metrics across seeds. Optionally runs a
paired Wilcoxon test between two named methods. Writes ``experiments/comparison.csv``.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.eval import paired_wilcoxon, save_json, summarize_seeds  # noqa: E402

RUN_RE = re.compile(r"^(?P<name>.+)_(?P<tier>smoke|dev|full)_seed(?P<seed>\d+)$")
HEADLINE = ["test_balanced_accuracy", "test_macro_f1", "test_accuracy"]


def collect(exp_dir: Path):
    groups = defaultdict(lambda: defaultdict(list))  # (name,tier) -> metric -> [values]
    per_group_seeds = defaultdict(list)
    for run in sorted(exp_dir.glob("*_seed*")):
        m = RUN_RE.match(run.name)
        summary = run / "summary.json"
        if not m or not summary.exists():
            continue
        data = json.loads(summary.read_text())
        key = (m["name"], m["tier"])
        per_group_seeds[key].append(int(m["seed"]))
        for metric in HEADLINE:
            if metric in data:
                groups[key][metric].append(float(data[metric]))
    return groups, per_group_seeds


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--exp-dir", default=str(REPO / "experiments"))
    p.add_argument("--compare", nargs=2, metavar=("A", "B"),
                   help="two run names to paired-test on balanced accuracy")
    args = p.parse_args(argv)

    exp_dir = Path(args.exp_dir)
    groups, seeds = collect(exp_dir)
    if not groups:
        print(f"No summary.json runs found under {exp_dir}")
        return 0

    rows = []
    print(f"{'method':<22}{'tier':<7}{'seeds':<7}{'balanced_acc (mean±std)':<28}macro_f1")
    for (name, tier), metrics in sorted(groups.items()):
        ba = summarize_seeds(metrics.get("test_balanced_accuracy", []))
        f1 = summarize_seeds(metrics.get("test_macro_f1", []))
        print(f"{name:<22}{tier:<7}{ba['n']:<7}"
              f"{ba['mean']:.4f} ± {ba['std']:.4f}          {f1['mean']:.4f} ± {f1['std']:.4f}")
        rows.append({"method": name, "tier": tier, "n_seeds": ba["n"],
                     "balanced_acc_mean": ba["mean"], "balanced_acc_std": ba["std"],
                     "macro_f1_mean": f1["mean"], "macro_f1_std": f1["std"]})

    save_json(rows, exp_dir / "comparison.json")

    if args.compare:
        a, b = args.compare
        va = next((m.get("test_balanced_accuracy") for (n, _), m in groups.items() if n == a), None)
        vb = next((m.get("test_balanced_accuracy") for (n, _), m in groups.items() if n == b), None)
        if va and vb and len(va) == len(vb):
            res = paired_wilcoxon(va, vb)
            print(f"\nPaired Wilcoxon {a} vs {b}: mean_diff={res['mean_diff']:.4f} "
                  f"p={res['p_value']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
