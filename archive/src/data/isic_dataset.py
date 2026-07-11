from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_class_id(class_dir_name: str) -> int:
    return int(class_dir_name.replace("class_", ""))


def parse_client_id(client_dir_name: str) -> int:
    return int(client_dir_name.replace("client_", ""))


class ISICFederatedFolderDataset(Dataset):
    def __init__(self, root_dir: str | Path, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, int]] = []
        self._build_index()

    def _add_samples_for_client_dir(self, client_dir: Path, client_id: int) -> None:
        for class_dir in sorted([p for p in client_dir.iterdir() if p.is_dir()]):
            if not class_dir.name.startswith("class_"):
                continue

            class_id = parse_class_id(class_dir.name)

            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((image_path, class_id, client_id))

    def _build_index(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")

        subdirs = sorted([p for p in self.root_dir.iterdir() if p.is_dir()])

        if len(subdirs) == 0:
            raise RuntimeError(f"No subdirectories found under: {self.root_dir}")

        # Case 1: root_dir is train/ or test/ and contains client_* folders
        if all(p.name.startswith("client_") for p in subdirs):
            for client_dir in subdirs:
                client_id = parse_client_id(client_dir.name)
                self._add_samples_for_client_dir(client_dir, client_id)

        # Case 2: root_dir is already a single client folder and contains class_* folders
        elif all(p.name.startswith("class_") for p in subdirs):
            client_id = parse_client_id(self.root_dir.name)
            self._add_samples_for_client_dir(self.root_dir, client_id)

        else:
            raise RuntimeError(
                f"Unexpected folder structure under {self.root_dir}. "
                f"Expected client_* or class_* directories."
            )

        if len(self.samples) == 0:
            raise RuntimeError(f"No image samples found under: {self.root_dir}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        image_path, class_id, client_id = self.samples[index]

        image = Image.open(image_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return {
            "image": image,
            "label": class_id,
            "client_id": client_id,
            "image_path": str(image_path),
        }
