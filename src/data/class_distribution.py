import os
import json
import csv
from collections import defaultdict

DATA_ROOT = "data"
RAW_ROOT = os.path.join(DATA_ROOT, "fed_isic2019", "raw")
OUTPUT_DIR = os.path.join(DATA_ROOT, "metadata")
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png")


def count_images_in_dir(path):
    return sum(
        1
        for f in os.listdir(path)
        if os.path.isfile(os.path.join(path, f)) and f.lower().endswith(IMAGE_EXTENSIONS)
    )


def get_client_class_distribution(split_root):
    """
    Expected structure:
    split_root/
        client_0/
            class_a/
                img1.jpg
            class_b/
                img2.jpg
    """
    result = {}

    if not os.path.exists(split_root):
        return result

    for client in sorted(os.listdir(split_root)):
        client_path = os.path.join(split_root, client)
        if not os.path.isdir(client_path):
            continue

        class_counts = {}
        for class_name in sorted(os.listdir(client_path)):
            class_path = os.path.join(client_path, class_name)
            if not os.path.isdir(class_path):
                continue

            class_counts[class_name] = count_images_in_dir(class_path)

        result[client] = class_counts

    return result


def merge_split_distributions(train_dist, val_dist, test_dist):
    clients = sorted(set(train_dist) | set(val_dist) | set(test_dist))
    merged = {}

    for client in clients:
        merged[client] = {
            "train": train_dist.get(client, {}),
            "val": val_dist.get(client, {}),
            "test": test_dist.get(client, {}),
        }

    return merged


def build_label_mapping(merged_dist):
    labels = sorted({
        label
        for client_info in merged_dist.values()
        for split_info in client_info.values()
        for label in split_info.keys()
    })
    return {label: idx for idx, label in enumerate(labels)}


def save_json(data, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return path


def save_csv(merged_dist, label_mapping, filename):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, filename)

    labels = list(label_mapping.keys())

    header = ["client_id", "split"] + labels

    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)

        for client, split_info in merged_dist.items():
            for split in ["train", "val", "test"]:
                row = [client, split]
                counts = split_info.get(split, {})
                row.extend(counts.get(label, 0) for label in labels)
                writer.writerow(row)

    return path


def print_summary(merged_dist):
    print("\n=== Class Distribution Summary ===")
    for client, split_info in merged_dist.items():
        print(f"\n{client}")
        for split in ["train", "val", "test"]:
            counts = split_info.get(split, {})
            total = sum(counts.values())
            print(f"  {split}: total={total}, classes={len(counts)}")
            for label, count in counts.items():
                print(f"    - {label}: {count}")


def main():
    train_root = os.path.join(RAW_ROOT, "train")
    val_root = os.path.join(RAW_ROOT, "val")
    test_root = os.path.join(RAW_ROOT, "test")

    train_dist = get_client_class_distribution(train_root)
    val_dist = get_client_class_distribution(val_root)
    test_dist = get_client_class_distribution(test_root)

    merged_dist = merge_split_distributions(train_dist, val_dist, test_dist)
    label_mapping = build_label_mapping(merged_dist)

    print_summary(merged_dist)

    json_path = save_json(merged_dist, "client_class_distribution.json")
    mapping_path = save_json(label_mapping, "label_mapping.json")
    csv_path = save_csv(merged_dist, label_mapping, "client_class_distribution.csv")

    print(f"\nSaved JSON: {json_path}")
    print(f"Saved CSV : {csv_path}")
    print(f"Saved map : {mapping_path}")


if __name__ == "__main__":
    main()