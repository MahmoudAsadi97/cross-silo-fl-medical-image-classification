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

        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        info["torch"] = None
    return info


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
    save_yaml(manifest, path)
    return path
