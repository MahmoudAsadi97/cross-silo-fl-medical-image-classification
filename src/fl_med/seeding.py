"""Global, reproducible seeding.

``set_seed`` covers python ``random``, numpy, and (if installed) torch + CUDA.
Torch is imported lazily so the pure-python core stays importable without it.
``seed_worker`` is a DataLoader ``worker_init_fn`` that gives each worker a
deterministic, distinct seed.
"""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42, deterministic: bool = True) -> int:
    """Seed all RNGs. Returns the seed for convenient logging."""
    os.environ["PYTHONHASHSEED"] = str(seed)
    # Make cuBLAS matmuls deterministic (also silences torch's per-backward warning
    # when deterministic algorithms are enabled on CUDA >= 10.2). Must be set before
    # the CUDA context initializes, so set_seed is called early in every entrypoint.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            # Opt-in deterministic algorithms; warn_only avoids hard failures on
            # ops without a deterministic implementation (e.g. some pooling).
            try:
                torch.use_deterministic_algorithms(True, warn_only=True)
            except TypeError:  # older torch without warn_only
                pass
        else:
            torch.backends.cudnn.deterministic = False
            torch.backends.cudnn.benchmark = True
    except ImportError:
        pass  # torch not installed: python/numpy seeding still applies.

    return seed


def seed_worker(worker_id: int) -> None:  # pragma: no cover - runs in subprocess
    """DataLoader worker_init_fn for reproducible shuffling/augmentation."""
    base = np.random.get_state()[1][0]
    worker_seed = (int(base) + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_generator(seed: int = 42):
    """Return a seeded ``torch.Generator`` (for DataLoader ``generator=``)."""
    import torch

    g = torch.Generator()
    g.manual_seed(seed)
    return g
