#!/usr/bin/env python3
"""Regenerate report figures from artifacts in ``experiments/``.

* re-runs the heterogeneity analysis (Phase 1),
* re-draws per-run curve figures from each ``metrics.csv``,
and copies the resulting PNGs into ``reports/figures/`` so the report always
references freshly-generated images (brief §5, reproducibility).
"""
from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.eval import plot_curves  # noqa: E402

FIGDIR = REPO / "reports" / "figures"


def _read_csv(path: Path):
    with path.open() as f:
        return [
            {k: (float(v) if _isfloat(v) else v) for k, v in row.items()}
            for row in csv.DictReader(f)
        ]


def _isfloat(v: str) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def main() -> int:
    FIGDIR.mkdir(parents=True, exist_ok=True)

    # Phase 1 heterogeneity (regenerate + copy).
    import subprocess

    subprocess.run([sys.executable, str(REPO / "scripts" / "analyze_heterogeneity.py")], check=False)
    het = REPO / "experiments" / "heterogeneity"
    for png in het.glob("*.png"):
        shutil.copy(png, FIGDIR / png.name)

    # Per-run curves from metrics.csv.
    for metrics_csv in (REPO / "experiments").glob("*/metrics.csv"):
        history = _read_csv(metrics_csv)
        if not history:
            continue
        run = metrics_csv.parent.name
        xkey = "round" if "round" in history[0] else "epoch"
        ys = [k for k in ("test_balanced_accuracy", "test_macro_f1") if k in history[0]]
        if ys:
            plot_curves(history, xkey, ys, FIGDIR / f"{run}_curves.png",
                        title=run, xlabel=xkey, ylabel="score")
    print(f"Figures -> {FIGDIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
