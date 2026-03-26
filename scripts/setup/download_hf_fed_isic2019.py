from pathlib import Path
import json
import sys

from datasets import load_dataset

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.paths import FED_ISIC2019_ROOT, RAW_DIR, REPORTS_DIR, ensure_data_directories


def main() -> None:
    ensure_data_directories()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading dataset from Hugging Face...")
    ds = load_dataset("flwrlabs/fed-isic2019")

    print(ds)

    summary = {}
    for split_name, split_ds in ds.items():
        summary[split_name] = {
            "num_rows": len(split_ds),
            "features": list(split_ds.features.keys()),
        }

    split_info_path = REPORTS_DIR / "hf_dataset_overview.json"
    with split_info_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("\nSaved dataset overview to:", split_info_path)

    for split_name, split_ds in ds.items():
        print(f"\n=== Split: {split_name} ===")
        print("Rows:", len(split_ds))
        print("Columns:", list(split_ds.features.keys()))

        sample = split_ds[0]
        print("Sample keys:", list(sample.keys()))

        out_dir = RAW_DIR / split_name
        out_dir.mkdir(parents=True, exist_ok=True)

        records = []
        for idx, item in enumerate(split_ds):
            center = item.get("center")
            label = item.get("label")
            image = item.get("image")

            center_str = str(center)
            label_str = str(label)

            image_dir = out_dir / f"client_{center_str}" / f"class_{label_str}"
            image_dir.mkdir(parents=True, exist_ok=True)

            image_path = image_dir / f"{split_name}_{idx:06d}.jpg"
            image.save(image_path)

            records.append(
                {
                    "index": idx,
                    "split": split_name,
                    "center": center,
                    "label": label,
                    "relative_image_path": str(image_path.relative_to(FED_ISIC2019_ROOT)),
                }
            )

            if (idx + 1) % 500 == 0:
                print(f"[{split_name}] Saved {idx + 1}/{len(split_ds)} images")

        metadata_path = REPORTS_DIR / f"{split_name}_metadata.json"
        with metadata_path.open("w", encoding="utf-8") as f:
            json.dump(records, f, indent=2)

        print(f"Saved {len(records)} records to {metadata_path}")

    print("\nDone.")
    print("Dataset exported under:", RAW_DIR)


if __name__ == "__main__":
    main()
