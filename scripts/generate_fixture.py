#!/usr/bin/env python3
"""Generate the committed tiny synthetic fixture (train/ + test/ trees).

Run once; the output under ``data/fixtures/fed_isic2019_tiny`` is committed so
smoke tests and CI never need the real multi-GB dataset.
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from fl_med.data.fixture import generate_fixture  # noqa: E402
from fl_med.data.paths import FIXTURE_RAW_DIR  # noqa: E402


def main() -> int:
    summary = generate_fixture(FIXTURE_RAW_DIR)
    print(f"Fixture written to {FIXTURE_RAW_DIR}")
    print(f"  train images: {summary['train']}")
    print(f"  test  images: {summary['test']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
