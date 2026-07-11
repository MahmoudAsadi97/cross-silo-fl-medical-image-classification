"""Generate a tiny synthetic Fed-ISIC2019-shaped fixture.

The fixture mirrors the real directory layout and the real non-IID character
(different clients get different class mixes, some classes missing) but uses a
handful of procedurally-generated RGB images per class. It is committed to the
repo so that smoke tests, CI, and the full pipeline run WITHOUT the multi-GB
real dataset. Class-conditional color tints give the tiny CNN a weak-but-real
signal to learn, so the smoke run exercises a genuine train/eval path.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np

# Per-client class -> number of train images. Deliberately non-IID: client 0 has
# all classes, others are skewed and miss classes (mirrors the real splits).
_DEFAULT_PLAN: Dict[int, Dict[int, int]] = {
    0: {0: 4, 1: 4, 2: 3, 3: 2, 4: 2, 5: 1, 6: 1, 7: 1},
    1: {1: 6, 4: 3, 0: 1},                       # nevus-heavy, few classes
    2: {0: 3, 2: 3, 4: 2, 7: 2},
    3: {1: 4, 5: 2, 6: 2},
    4: {0: 2, 3: 2, 4: 2},
    5: {1: 3, 2: 2},                             # smallest, most skewed
}


def _class_tint(class_id: int) -> np.ndarray:
    """Deterministic base RGB tint per class (gives the smoke model real signal)."""
    rng = np.random.default_rng(1000 + class_id)
    return rng.integers(40, 215, size=3)


def _make_image(class_id: int, salt: int, size: int = 32) -> "np.ndarray":
    rng = np.random.default_rng(class_id * 10_000 + salt)
    base = _class_tint(class_id)
    noise = rng.integers(-25, 26, size=(size, size, 3))
    img = np.clip(base.reshape(1, 1, 3) + noise, 0, 255).astype(np.uint8)
    return img


def generate_fixture(
    root: str | Path,
    plan: Dict[int, Dict[int, int]] | None = None,
    test_fraction: float = 0.5,
    image_size: int = 32,
    seed: int = 0,
) -> Dict[str, int]:
    """Write train/ and test/ trees under ``root``. Returns a small summary."""
    from PIL import Image

    plan = plan or _DEFAULT_PLAN
    root = Path(root)
    counts = {"train": 0, "test": 0}

    for client_id, class_plan in plan.items():
        for class_id, n_train in class_plan.items():
            n_test = max(1, int(round(n_train * test_fraction)))
            for split, n in (("train", n_train), ("test", n_test)):
                out_dir = root / split / f"client_{client_id}" / f"class_{class_id}"
                out_dir.mkdir(parents=True, exist_ok=True)
                for k in range(n):
                    salt = seed * 100 + (0 if split == "train" else 50) + k
                    arr = _make_image(class_id, salt, image_size)
                    Image.fromarray(arr, "RGB").save(
                        out_dir / f"{split}_{class_id:02d}_{k:02d}.png"
                    )
                    counts[split] += 1
    return counts


def fixture_class_counts(plan: Dict[int, Dict[int, int]] | None = None) -> "np.ndarray":
    """The fixture's client x class train-count matrix (for heterogeneity tests)."""
    plan = plan or _DEFAULT_PLAN
    n_clients = max(plan) + 1
    counts = np.zeros((n_clients, 8), dtype=np.int64)
    for client_id, class_plan in plan.items():
        for class_id, n in class_plan.items():
            counts[client_id, class_id] = n
    return counts


if __name__ == "__main__":  # pragma: no cover
    from .paths import FIXTURE_RAW_DIR

    summary = generate_fixture(FIXTURE_RAW_DIR)
    print(f"Fixture written to {FIXTURE_RAW_DIR}: {summary}")
