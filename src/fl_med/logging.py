"""Logging + run provenance.

Two responsibilities:

* ``get_logger`` -- a console (+ optional file) logger with a consistent format.
* ``write_run_manifest`` -- dump ``run_config.yaml`` capturing the resolved
  config, git commit, tier, hardware, and library versions into a run's output
  directory. Every experiment must call this so any artifact traces back to the
  exact code + config that produced it (brief §5, reproducibility).
"""
from __future__ import annotations

import logging
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

from .config import save_yaml


def get_logger(name: str, log_file: Optional[str | Path] = None) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    if logger.handlers:  # already configured
        return logger

    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(name)s | %(message)s")
    stream = logging.StreamHandler()
    stream.setFormatter(fmt)
    logger.addHandler(stream)

    if log_file is not None:
        path = Path(log_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(path, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


def _git_commit() -> Optional[str]:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return None


def _git_dirty() -> Optional[bool]:
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return len(out.decode().strip()) > 0
    except Exception:
        return None


def collect_hardware() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }
    try:
        import os

        info["cpu_count"] = os.cpu_count()
    except Exception:
        pass
    try:
        import torch

        # torch.__version__ is a TorchVersion (str subclass) that yaml.safe_dump
        # cannot represent -> coerce to plain str.
        info["torch"] = str(torch.__version__)
        info["cuda_available"] = bool(torch.cuda.is_available())
        if torch.cuda.is_available():
            info["gpu"] = str(torch.cuda.get_device_name(0))
            info["gpu_count"] = int(torch.cuda.device_count())
    except ImportError:
        info["torch"] = None
    return info


def _yaml_safe(obj):
    """Recursively coerce a structure into YAML-safe primitives.

    yaml.safe_dump only handles exact built-in types; str/int subclasses (e.g.
    torch's TorchVersion) and numpy scalars raise RepresenterError. This makes the
    run manifest robust to whatever ends up in a resolved config.
    """
    if obj is None or isinstance(obj, bool):
        return obj
    if isinstance(obj, str):
        return str(obj)
    if isinstance(obj, int):
        return int(obj)
    if isinstance(obj, float):
        return float(obj)
    if isinstance(obj, dict):
        return {str(k): _yaml_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_yaml_safe(v) for v in obj]
    # numpy scalars and anything exotic -> python scalar or str.
    item = getattr(obj, "item", None)
    if callable(item):
        try:
            return _yaml_safe(item())
        except Exception:
            pass
    return str(obj)


def write_run_manifest(
    output_dir: str | Path,
    config: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
) -> Path:
    """Write ``run_config.yaml`` into ``output_dir`` and return its path."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "tier": config.get("_meta", {}).get("tier"),
        "git_commit": _git_commit(),
        "git_dirty": _git_dirty(),
        "hardware": collect_hardware(),
        "config": config,
    }
    if extra:
        manifest["extra"] = extra

    path = output_dir / "run_config.yaml"
    save_yaml(_yaml_safe(manifest), path)
    return path
