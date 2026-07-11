import os
import json
import csv
from collections import defaultdict

DATA_ROOT = "data"
OUTPUT_DIR = os.path.join(DATA_ROOT, "metadata")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def inspect_dataset(root):
    if not os.path.exists(root):
        raise FileNotFoundError(f"Dataset path not found: {root}")

    client_stats = defaultdict(lambda: {"samples": 0, "by_split": {}})

    fed_root = os.path.join(root, "fed_isic2019", "raw")

    for split in ["train", "val", "test"]:
        split_path = os.path.join(fed_root, split)

        if not os.path.exists(split_path):
            continue

        for client in sorted(os.listdir(split_path)):
            client_path = os.path.join(split_path, client)

            if not os.path.isdir(client_path):
                continue

            count = 0
            for dirpath, _, filenames in os.walk(client_path):
                count += sum(
                    1 for f in filenames if f.lower().endswith(IMAGE_EXTENSIONS)
                )

            client_stats[client]["samples"] += count
            client_stats[client]["by_split"][split] = count

    return dict(sorted(client_stats.items()))


def save_json(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "client_sizes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    return out_path


def save_csv(stats, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "client_sizes.csv")

    splits = ["train", "val", "test"]
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id", "total_samples", "train_samples", "val_samples", "test_samples"])

        for client, info in stats.items():
            row = [
                client,
                info["samples"],
                info["by_split"].get("train", 0),
                info["by_split"].get("val", 0),
                info["by_split"].get("test", 0),
            ]
            writer.writerow(row)

    return out_path


def print_summary(stats):
    print("\n=== Dataset Summary ===")
    for client, info in stats.items():
        split_info = ", ".join(
            f"{split}={info['by_split'].get(split, 0)}"
            for split in ["train", "val", "test"]
        )
        print(f"{client}: total={info['samples']} ({split_info})")


def main():
    stats = inspect_dataset(DATA_ROOT)
    print_summary(stats)

    json_path = save_json(stats, OUTPUT_DIR)
    csv_path = save_csv(stats, OUTPUT_DIR)

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")


if __name__ == "__main__":
    main()