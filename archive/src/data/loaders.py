from __future__ import annotations

from torch.utils.data import DataLoader

from src.data.isic_dataset import ISICFederatedFolderDataset
from src.data.paths import RAW_DIR
from src.data.transforms import get_train_transforms, get_eval_transforms


def build_centralized_datasets(image_size: int = 224):
    train_dataset = ISICFederatedFolderDataset(
        root_dir=RAW_DIR / "train",
        transform=get_train_transforms(image_size=image_size),
    )

    test_dataset = ISICFederatedFolderDataset(
        root_dir=RAW_DIR / "test",
        transform=get_eval_transforms(image_size=image_size),
    )

    return train_dataset, test_dataset


def build_centralized_dataloaders(
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_dataset, test_dataset = build_centralized_datasets(image_size=image_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader


def build_client_datasets(client_id: int, image_size: int = 224):
    train_dataset = ISICFederatedFolderDataset(
        root_dir=RAW_DIR / "train" / f"client_{client_id}",
        transform=get_train_transforms(image_size=image_size),
    )

    test_dataset = ISICFederatedFolderDataset(
        root_dir=RAW_DIR / "test" / f"client_{client_id}",
        transform=get_eval_transforms(image_size=image_size),
    )

    return train_dataset, test_dataset


def build_client_dataloaders(
    client_id: int,
    image_size: int = 224,
    batch_size: int = 32,
    num_workers: int = 4,
):
    train_dataset, test_dataset = build_client_datasets(
        client_id=client_id,
        image_size=image_size,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, test_loader
