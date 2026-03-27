from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data.loaders import build_centralized_dataloaders


def main():
    train_loader, test_loader = build_centralized_dataloaders(
        image_size=224,
        batch_size=8,
        num_workers=2,
    )

    print("Train batches:", len(train_loader))
    print("Test batches:", len(test_loader))

    train_batch = next(iter(train_loader))
    test_batch = next(iter(test_loader))

    print("\n=== Train batch ===")
    print("Image shape:", train_batch["image"].shape)
    print("Labels:", train_batch["label"])
    print("Client IDs:", train_batch["client_id"])
    print("First image path:", train_batch["image_path"][0])

    print("\n=== Test batch ===")
    print("Image shape:", test_batch["image"].shape)
    print("Labels:", test_batch["label"])
    print("Client IDs:", test_batch["client_id"])
    print("First image path:", test_batch["image_path"][0])


if __name__ == "__main__":
    main()
