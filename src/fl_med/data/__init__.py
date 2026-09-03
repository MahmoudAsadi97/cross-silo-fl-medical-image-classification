"""Data subpackage: dataset, tier-aware loaders, heterogeneity metrics, fixture."""
from __future__ import annotations

from . import heterogeneity, paths
from .dataset import ISICFederatedFolderDataset
from .fixture import fixture_class_counts, generate_fixture
from .heterogeneity import counts_from_dataset, heterogeneity_report
from .loaders import (
    build_centralized_dataloaders,
    build_centralized_test_dataloader,
    build_client_dataloaders,
    list_clients,
)

__all__ = [
    "paths",
    "heterogeneity",
    "ISICFederatedFolderDataset",
    "build_centralized_dataloaders",
    "build_centralized_test_dataloader",
    "build_client_dataloaders",
    "list_clients",
    "heterogeneity_report",
    "counts_from_dataset",
    "generate_fixture",
    "fixture_class_counts",
]
