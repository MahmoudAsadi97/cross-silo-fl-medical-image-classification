from __future__ import annotations

import numpy as np
import pandas as pd
from pathlib import Path

from src.data.paths import REPORTS_DIR


def load_matrix():
    path = REPORTS_DIR / "client_class_matrix.csv"
    df = pd.read_csv(path)
    df = df.set_index(["split", "client_id"])
    return df


def compute_entropy(probs):
    probs = probs[probs > 0]
    return -np.sum(probs * np.log(probs))


def analyze():
    df = load_matrix()

    results = []

    for (split, client_id), row in df.iterrows():
        counts = row.values
        total = counts.sum()

        if total == 0:
            continue

        probs = counts / total
        entropy = compute_entropy(probs)

        results.append({
            "split": split,
            "client_id": client_id,
            "num_samples": int(total),
            "entropy": float(entropy),
            "num_classes": int((counts > 0).sum())
        })

    result_df = pd.DataFrame(results)

    # imbalance ratio (max / min client size)
    imbalance = result_df.groupby("split")["num_samples"].agg(["max", "min"])
    imbalance["imbalance_ratio"] = imbalance["max"] / imbalance["min"]

    print("\n=== Entropy per client ===")
    print(result_df.sort_values(["split", "client_id"]).to_string(index=False))

    print("\n=== Imbalance ratio ===")
    print(imbalance.to_string())

    out_path = REPORTS_DIR / "heterogeneity_metrics.csv"
    result_df.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    analyze()
