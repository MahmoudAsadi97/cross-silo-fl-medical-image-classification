"""Config/tier-aware DataLoader construction.

Data root is chosen by tier (smoke -> committed fixture, dev/full -> real data),
loaders are seeded for reproducibility, and augmentation is train-only. Set
``data.augment: false`` to train WITHOUT augmentation (used e.g. by the membership-
inference attack, which needs the target model to overfit). Torch imported lazily.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

from ..seeding import make_generator, seed_worker
from .dataset import ISICFederatedFolderDataset
from .paths import resolve_data_root
from .transforms import get_eval_transforms, get_train_transforms


def _cfg(config: dict) -> dict:
    data = dict(config.get("data", {}) or {})
    data.setdefault("image_size", 200)
    data.setdefault("batch_size", 32)
    data.setdefault("num_workers", 0)
    data.setdefault("augment", True)
    data.setdefault("seed", config.get("seed", 42))
    return data


def _train_transform(data: dict):
    if data.get("augment", True):
        return get_train_transforms(data["image_size"])
    return get_eval_transforms(data["image_size"])   # no-augmentation target (MIA)


def _make_loader(dataset, *, batch_size, shuffle, num_workers, seed):
    from torch.utils.data import DataLoader

    kwargs = dict(batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    if num_workers and num_workers > 0:
        kwargs["worker_init_fn"] = seed_worker
        kwargs["pin_memory"] = True
    if shuffle:
        kwargs["generator"] = make_generator(seed)
    return DataLoader(dataset, **kwargs)


def build_centralized_dataloaders(config: dict) -> Tuple[object, object]:
    """Pool all clients' train data; evaluate on the pooled held-out test split."""
    data = _cfg(config)
    root = Path(resolve_data_root(config))
    train_ds = ISICFederatedFolderDataset(root / "train", transform=_train_transform(data))
    test_ds = ISICFederatedFolderDataset(root / "test", transform=get_eval_transforms(data["image_size"]))
    train_loader = _make_loader(train_ds, batch_size=data["batch_size"], shuffle=True,
                                num_workers=data["num_workers"], seed=data["seed"])
    test_loader = _make_loader(test_ds, batch_size=data["batch_size"], shuffle=False,
                               num_workers=data["num_workers"], seed=data["seed"])
    return train_loader, test_loader


def build_client_dataloaders(config: dict, client_id: int) -> Tuple[object, object]:
    """Per-client train loader + that client's local test loader."""
    data = _cfg(config)
    root = Path(resolve_data_root(config))
    train_ds = ISICFederatedFolderDataset(
        root / "train" / f"client_{client_id}", transform=_train_transform(data))
    test_ds = ISICFederatedFolderDataset(
        root / "test" / f"client_{client_id}", transform=get_eval_transforms(data["image_size"]))
    train_loader = _make_loader(train_ds, batch_size=data["batch_size"], shuffle=True,
                                num_workers=data["num_workers"], seed=data["seed"] + client_id)
    test_loader = _make_loader(test_ds, batch_size=data["batch_size"], shuffle=False,
                               num_workers=data["num_workers"], seed=data["seed"])
    return train_loader, test_loader


def list_clients(config: dict) -> list:
    """Client ids present in the (tier-resolved) train split."""
    root = Path(resolve_data_root(config)) / "train"
    return sorted(
        int(p.name.replace("client_", "")) for p in root.iterdir()
        if p.is_dir() and p.name.startswith("client_")
    )
