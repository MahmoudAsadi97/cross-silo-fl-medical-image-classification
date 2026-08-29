#!/usr/bin/env python3
"""Secure-aggregation demonstration (Phase: sub-question 7).

Simulates additive pairwise-mask secure aggregation over Fed-ISIC2019-shaped client
updates and shows: (1) the server recovers the EXACT FedAvg aggregate (masks cancel),
(2) each individual masked update is uninformative. Writes a JSON summary and a
log-scale figure contrasting per-client masking magnitude vs the ~0 aggregate error.
Pure numpy -- no GPU needed.
"""
from __future__ import annotations

import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.eval import save_json  # noqa: E402
from fl_med.security.secure_agg import secure_weighted_average  # noqa: E402
from fl_med.strategies.aggregation import weighted_average  # noqa: E402

CLIENT_SIZES = [7947, 2531, 2156, 1448, 525, 281]  # real Fed-ISIC2019 train sizes


def main() -> int:
    rng = np.random.default_rng(0)
    # ResNet-ish shapes (a conv-like and a classifier-like tensor) per client.
    updates = [OrderedDict(conv=rng.normal(size=(64, 3, 3, 3)), fc=rng.normal(size=(8, 64)))
               for _ in CLIENT_SIZES]

    plain = weighted_average(updates, CLIENT_SIZES)
    out = secure_weighted_average(updates, CLIENT_SIZES, master_seed=2024, scale=100.0)
    sec = out["aggregate"]

    agg_err = max(float(np.max(np.abs(sec[k] - plain[k]))) for k in plain)
    per_client_hidden = []
    for i in range(len(updates)):
        true_i = np.concatenate([out["scaled_contributions"][i][k].ravel() for k in plain])
        masked_i = np.concatenate([out["masked_updates"][i][k].ravel() for k in plain])
        per_client_hidden.append({
            "client": i,
            "true_norm": float(np.linalg.norm(true_i)),
            "masked_deviation": float(np.linalg.norm(masked_i - true_i)),
        })

    summary = {
        "n_clients": len(updates),
        "aggregate_max_abs_error_vs_plaintext": agg_err,
        "aggregate_exact": bool(agg_err < 1e-6),
        "per_client": per_client_hidden,
        "mean_hidden_ratio": float(np.mean([c["masked_deviation"] / max(c["true_norm"], 1e-9)
                                            for c in per_client_hidden])),
    }
    outdir = REPO / "experiments" / "secure_agg"
    outdir.mkdir(parents=True, exist_ok=True)
    save_json(summary, outdir / "secure_agg.json")
    _plot(per_client_hidden, agg_err, REPO / "reports" / "figures" / "secure_agg.png")

    print("=== Secure aggregation (pairwise masks) ===")
    print(f"  aggregate max abs error vs plaintext FedAvg: {agg_err:.2e}  (exact: {agg_err < 1e-6})")
    print(f"  mean per-client hiding ratio: {summary['mean_hidden_ratio']:.0f}x")
    print("  -> server recovers the exact average; no individual update is exposed.")
    return 0


def _plot(per_client, agg_err, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(7, 4.2))
    xs = [f"c{c['client']}" for c in per_client]
    ax.bar(xs, [c["masked_deviation"] for c in per_client], color="#1f77b4",
           label="||masked - true|| per client (hidden)")
    ax.axhline(max(agg_err, 1e-16), color="#2ca02c", lw=2,
               label=f"aggregate error vs plaintext ({agg_err:.0e})")
    ax.set_yscale("log")
    ax.set_ylabel("magnitude (log scale)")
    ax.set_title("Secure aggregation: individuals hidden, aggregate exact")
    ax.legend(fontsize=8)
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
