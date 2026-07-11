from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import pandas as pd

from src.data.paths import RAW_DIR, REPORTS_DIR


def count_images_in_dir(path: Path) -> int:
    return len([p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}])


def parse_class_name(class_dir_name: str) -> int:
    return int(class_dir_name.replace("class_", ""))


def parse_client_name(client_dir_name: str) -> int:
    return int(client_dir_name.replace("client_", ""))


def collect_split_stats(split_dir: Path, split_name: str) -> pd.DataFrame:
    rows: List[Dict] = []

    if not split_dir.exists():
        return pd.DataFrame(columns=["split", "client_id", "class_id", "count"])

    for client_dir in sorted([p for p in split_dir.iterdir() if p.is_dir()]):
        client_id = parse_client_name(client_dir.name)

        for class_dir in sorted([p for p in client_dir.iterdir() if p.is_dir()]):
            class_id = parse_class_name(class_dir.name)
            count = count_images_in_dir(class_dir)

            rows.append(
                {
                    "split": split_name,
                    "client_id": client_id,
                    "class_id": class_id,
                    "count": count,
                }
            )

    return pd.DataFrame(rows)


def build_all_stats() -> pd.DataFrame:
    train_df = collect_split_stats(RAW_DIR / "train", "train")
    test_df = collect_split_stats(RAW_DIR / "test", "test")
    return pd.concat([train_df, test_df], ignore_index=True)


def add_distribution_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    split_client_totals = df.groupby(["split", "client_id"])["count"].transform("sum")
    df["client_total_in_split"] = split_client_totals
    df["class_fraction_within_client"] = df["count"] / df["client_total_in_split"]
    return df


def save_outputs(df: pd.DataFrame) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    full_path = REPORTS_DIR / "client_class_counts.csv"
    df.to_csv(full_path, index=False)

    client_split_totals = (
        df.groupby(["split", "client_id"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "num_samples"})
    )
    client_split_totals.to_csv(REPORTS_DIR / "client_split_totals.csv", index=False)

    pivot = df.pivot_table(
        index=["split", "client_id"],
        columns="class_id",
        values="count",
        fill_value=0,
        aggfunc="sum",
    )
    pivot.to_csv(REPORTS_DIR / "client_class_matrix.csv")

    print(f"Saved: {full_path}")
    print(f"Saved: {REPORTS_DIR / 'client_split_totals.csv'}")
    print(f"Saved: {REPORTS_DIR / 'client_class_matrix.csv'}")


def print_summary(df: pd.DataFrame) -> None:
    print("=== Client Split Totals ===")
    totals = (
        df.groupby(["split", "client_id"], as_index=False)["count"]
        .sum()
        .rename(columns={"count": "num_samples"})
        .sort_values(["split", "client_id"])
    )
    print(totals.to_string(index=False))

    print("\n=== Number of classes present per client ===")
    classes_present = (
        df.groupby(["split", "client_id"])["class_id"]
        .nunique()
        .reset_index(name="num_classes_present")
        .sort_values(["split", "client_id"])
    )
    print(classes_present.to_string(index=False))

    print("\n=== Overall split totals ===")
    split_totals = df.groupby("split")["count"].sum().reset_index(name="num_samples")
    print(split_totals.to_string(index=False))


def main() -> None:
    df = build_all_stats()
    df = add_distribution_columns(df)
    save_outputs(df)
    print_summary(df)


if __name__ == "__main__":
    main()
