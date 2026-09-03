#!/usr/bin/env python3
"""Phase 1: quantify + visualize non-IID label skew across the six clients.

Scans a split directory (real data by default, fixture as a fallback) into a
client x class count matrix, computes the heterogeneity report (entropy,
KL/JS/Hellinger to the global pool, EMD, missing classes), and writes a CSV, a
JSON, a per-client stacked class-distribution bar chart, and a JS-distance bar.
Pure numpy/matplotlib -- runs without torch on the real metadata.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med import CLASS_NAMES, NUM_CLASSES  # noqa: E402
from fl_med.data.heterogeneity import counts_from_dataset, heterogeneity_report  # noqa: E402
from fl_med.data.paths import FIXTURE_RAW_DIR, RAW_DIR  # noqa: E402
from fl_med.eval import save_json  # noqa: E402


def _counts(root: Path, num_clients: int) -> np.ndarray:
    return counts_from_dataset(root, num_classes=NUM_CLASSES, num_clients=num_clients)


def _write_csv(report: dict, path: Path) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    rows = report["per_client"]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def _plots(counts: np.ndarray, report: dict, outdir: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    outdir.mkdir(parents=True, exist_ok=True)
    n_clients = counts.shape[0]
    client_ids = [r["client_id"] for r in report["per_client"]]

    # Stacked per-client class distribution (proportions).
    props = counts / np.clip(counts.sum(axis=1, keepdims=True), 1, None)
    fig, ax = plt.subplots(figsize=(8, 5))
    bottom = np.zeros(n_clients)
    for c in range(NUM_CLASSES):
        ax.bar(range(n_clients), props[:, c], bottom=bottom, label=CLASS_NAMES[c])
        bottom += props[:, c]
    ax.set_xticks(range(n_clients))
    ax.set_xticklabels([f"client {i}" for i in client_ids])
    ax.set_ylabel("class proportion")
    ax.set_title("Per-client class distribution (non-IID)")
    ax.legend(fontsize=7, bbox_to_anchor=(1.01, 1), loc="upper left")
    fig.tight_layout()
    fig.savefig(outdir / "client_class_distribution.png", dpi=120)
    plt.close(fig)

    # JS distance to global per client.
    js = [r["js_to_global"] for r in report["per_client"]]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar([f"c{i}" for i in client_ids], js, color="#c0504d")
    ax.set_ylabel("Jensen-Shannon distance to global")
    ax.set_title("Client skew vs. global label distribution")
    fig.tight_layout()
    fig.savefig(outdir / "js_distance_to_global.png", dpi=120)
    plt.close(fig)


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=None, help="split dir with client_*/ (default: real train)")
    p.add_argument("--split", default="train")
    p.add_argument("--clients", type=int, default=6)
    p.add_argument("--outdir", default=str(REPO / "experiments" / "heterogeneity"))
    args = p.parse_args(argv)

    if args.root:
        root = Path(args.root)
    else:
        real = RAW_DIR / args.split
        root = real if real.exists() else FIXTURE_RAW_DIR / args.split
    print(f"Scanning: {root}")

    counts = _counts(root, args.clients)
    report = heterogeneity_report(
        counts, client_ids=list(range(args.clients)), class_names=list(CLASS_NAMES)
    )
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    save_json({"counts": counts.tolist(), **report}, outdir / "heterogeneity.json")
    _write_csv(report, outdir / "heterogeneity_per_client.csv")
    _plots(counts, report, outdir)

    print(f"clients={report['num_clients']} classes={report['num_classes']} "
          f"global_entropy={report['global_entropy_nats']:.3f} nats")
    for r in report["per_client"]:
        print(f"  client {r['client_id']}: n={r['num_samples']:>6} "
              f"missing={r['num_classes_missing']} "
              f"entropy={r['entropy_nats']:.3f} JS={r['js_to_global']:.3f}")
    print(f"Artifacts -> {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
