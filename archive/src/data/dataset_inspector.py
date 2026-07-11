from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

from src.data.paths import RAW_DIR, REPORTS_DIR, ensure_data_directories


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def list_files_recursive(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return [p for p in root.rglob("*") if p.is_file()]


def list_image_files(root: Path) -> List[Path]:
    return [p for p in list_files_recursive(root) if p.suffix.lower() in IMAGE_EXTENSIONS]


def immediate_subdirs(root: Path) -> List[Path]:
    if not root.exists():
        return []
    return sorted([p for p in root.iterdir() if p.is_dir()])


def detect_metadata_files(root: Path) -> List[Path]:
    candidates = []
    for ext in [".csv", ".json", ".txt", ".xlsx"]:
        candidates.extend(root.rglob(f"*{ext}"))
    return sorted(candidates)


def build_dataset_summary() -> Dict:
    ensure_data_directories()

    raw_exists = RAW_DIR.exists()
    all_files = list_files_recursive(RAW_DIR)
    image_files = list_image_files(RAW_DIR)
    subdirs = immediate_subdirs(RAW_DIR)
    metadata_files = detect_metadata_files(RAW_DIR)

    summary = {
        "raw_dir": str(RAW_DIR),
        "raw_exists": raw_exists,
        "total_files": len(all_files),
        "total_image_files": len(image_files),
        "top_level_subdirectories": [p.name for p in subdirs],
        "num_top_level_subdirectories": len(subdirs),
        "metadata_files": [str(p.relative_to(RAW_DIR)) for p in metadata_files],
    }

    if subdirs:
        per_subdir = {}
        for subdir in subdirs:
            per_subdir[subdir.name] = {
                "total_files": len(list_files_recursive(subdir)),
                "image_files": len(list_image_files(subdir)),
                "subdirectories": [p.name for p in immediate_subdirs(subdir)],
            }
        summary["per_top_level_subdirectory"] = per_subdir

    return summary


def save_summary(summary: Dict) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "dataset_summary.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    return out_path


def print_summary(summary: Dict) -> None:
    print("=== Fed-ISIC2019 Dataset Summary ===")
    print("Raw directory:", summary["raw_dir"])
    print("Exists:", summary["raw_exists"])
    print("Total files:", summary["total_files"])
    print("Total image files:", summary["total_image_files"])
    print("Top-level subdirectories:", summary["top_level_subdirectories"])
    print("Metadata files:", summary["metadata_files"])

    if "per_top_level_subdirectory" in summary:
        print("\\n--- Per top-level subdirectory ---")
        for name, info in summary["per_top_level_subdirectory"].items():
            print(f"{name}:")
            print(f"  total_files={info['total_files']}")
            print(f"  image_files={info['image_files']}")
            print(f"  subdirectories={info['subdirectories']}")


def main() -> None:
    summary = build_dataset_summary()
    out_path = save_summary(summary)
    print_summary(summary)
    print("\\nSaved summary to:", out_path)


if __name__ == "__main__":
    main()
