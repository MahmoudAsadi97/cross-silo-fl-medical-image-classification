"""fl_med: cross-silo federated learning for medical image classification.

The package is organised so that the *pure-python core* (config, seeding,
metrics, and the aggregation / control-variate math in ``strategies``) imports
without any heavy dependencies. Anything requiring the torch stack lives behind
lazy imports inside functions, so the core can be unit-tested on a machine with
only numpy available (see ``tests/`` and ``scripts/verify_core_math.py``).
"""
from __future__ import annotations

__version__ = "0.1.0"

# Number of diagnostic classes in Fed-ISIC2019.
NUM_CLASSES = 8

# Human-readable class names (index == class_id in the on-disk ``class_<id>`` folders).
# Source: FLamby Fed-ISIC2019 / ISIC2019 challenge label set.
CLASS_NAMES = (
    "Melanoma",              # 0  MEL
    "Melanocytic nevus",     # 1  NV
    "Basal cell carcinoma",  # 2  BCC
    "Actinic keratosis",     # 3  AK
    "Benign keratosis",      # 4  BKL
    "Dermatofibroma",        # 5  DF
    "Vascular lesion",       # 6  VASC
    "Squamous cell carcinoma",  # 7  SCC
)

# The six natural cross-silo clients (4 source hospitals; one contributes 3
# clients due to 3 imaging technologies).
NUM_CLIENTS = 6

__all__ = ["__version__", "NUM_CLASSES", "CLASS_NAMES", "NUM_CLIENTS"]
