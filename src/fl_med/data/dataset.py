"""Folder-backed dataset for the official Fed-ISIC2019 splits.

Expected layout (either a split root holding ``client_*`` dirs, or a single
client dir holding ``class_*`` dirs)::

    <root>/client_<i>/class_<j>/<image>.jpg

Each item is a dict ``{image, label, client_id, image_path}``. Torch is imported
lazily so this module (and the label-index it builds) is importable without torch;
``__getitem__`` needs torch/PIL only when actually iterated.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}


def parse_class_id(name: str) -> int:
    return int(name.replace("class_", ""))


def parse_client_id(name: str) -> int:
    return int(name.replace("client_", ""))


def _try_import_dataset_base():
    try:
        from torch.utils.data import Dataset

        return Dataset
    except ImportError:  # allow import without torch; construction still works
        return object


class ISICFederatedFolderDataset(_try_import_dataset_base()):  # type: ignore[misc]
    def __init__(self, root_dir: str | Path, transform=None) -> None:
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples: List[Tuple[Path, int, int]] = []
        self._build_index()

    def _add_client_dir(self, client_dir: Path, client_id: int) -> None:
        for class_dir in sorted(p for p in client_dir.iterdir() if p.is_dir()):
            if not class_dir.name.startswith("class_"):
                continue
            class_id = parse_class_id(class_dir.name)
            for img in sorted(class_dir.iterdir()):
                if img.is_file() and img.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((img, class_id, client_id))

    def _build_index(self) -> None:
        if not self.root_dir.exists():
            raise FileNotFoundError(f"Dataset root does not exist: {self.root_dir}")
        subdirs = sorted(p for p in self.root_dir.iterdir() if p.is_dir())
        if not subdirs:
            raise RuntimeError(f"No subdirectories under: {self.root_dir}")

        if all(p.name.startswith("client_") for p in subdirs):
            for client_dir in subdirs:
                self._add_client_dir(client_dir, parse_client_id(client_dir.name))
        elif all(p.name.startswith("class_") for p in subdirs):
            self._add_client_dir(self.root_dir, parse_client_id(self.root_dir.name))
        else:
            raise RuntimeError(
                f"Unexpected structure under {self.root_dir}; expected client_*/ or class_*/"
            )
        if not self.samples:
            raise RuntimeError(f"No images found under: {self.root_dir}")

    def labels(self) -> List[int]:
        return [label for _, label, _ in self.samples]

    def client_ids(self) -> List[int]:
        return [cid for _, _, cid in self.samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        from PIL import Image

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
