from __future__ import annotations

import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))


def load_history(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def extract_metric(history_split, metric_name):
    epochs = [item["epoch"] for item in history_split]
    values = [item[metric_name] for item in history_split]
    return epochs, values


def make_plot(epochs, train_values, test_values, metric_name, out_path: Path):
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_values, label="train")
    plt.plot(epochs, test_values, label="test")
    plt.xlabel("Epoch")
    plt.ylabel(metric_name)
    plt.title(f"Centralized Baseline - {metric_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"Saved plot: {out_path}")


def main():
    run_dir = PROJECT_ROOT / "results" / "centralized_baseline"
    history_path = run_dir / "history.json"
    plot_dir = run_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    history = load_history(history_path)

    for metric_name in ["loss", "accuracy", "balanced_accuracy", "macro_f1"]:
        train_epochs, train_values = extract_metric(history["train"], metric_name)
        test_epochs, test_values = extract_metric(history["test"], metric_name)

        out_path = plot_dir / f"{metric_name}.png"
        make_plot(train_epochs, train_values, test_values, metric_name, out_path)


if __name__ == "__main__":
    main()
