"""Canonical on-disk locations for the Fed-ISIC2019 data and the tiny fixture."""
from __future__ import annotations

from pathlib import Path

# repo_root/src/fl_med/data/paths.py -> parents[3] == repo root
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DATA_ROOT = PROJECT_ROOT / "data"

# Real dataset (gitignored). Layout: FED_ISIC2019_ROOT/raw/{train,test}/client_<i>/class_<j>/*.jpg
FED_ISIC2019_ROOT = DATA_ROOT / "fed_isic2019"
RAW_DIR = FED_ISIC2019_ROOT / "raw"
METADATA_DIR = FED_ISIC2019_ROOT / "metadata"
REPORTS_DIR = FED_ISIC2019_ROOT / "reports"

# Tiny synthetic fixture (committed) with the SAME layout, for smoke tests + CI.
FIXTURE_ROOT = DATA_ROOT / "fixtures" / "fed_isic2019_tiny"
FIXTURE_RAW_DIR = FIXTURE_ROOT / "raw"


def data_root_for_tier(tier: str) -> Path:
    """smoke -> committed fixture; dev/full -> the real dataset."""
    return FIXTURE_RAW_DIR if tier == "smoke" else RAW_DIR


def resolve_data_root(config: dict) -> Path:
    """Pick the raw-data root from config: explicit ``data.root`` wins, else by tier."""
    explicit = (config.get("data") or {}).get("root")
    if explicit:
        return Path(explicit)
    tier = (config.get("_meta") or {}).get("tier", "smoke")
    return data_root_for_tier(tier)
