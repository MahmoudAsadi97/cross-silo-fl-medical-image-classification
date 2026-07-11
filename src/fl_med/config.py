"""Config loading and the smoke/dev/full tiering protocol.

A config is plain nested dicts loaded from YAML. Resolution order (later wins):

    1. ``_base.yaml``            -- shared defaults (found next to the experiment file)
    2. the experiment YAML       -- e.g. ``fedavg.yaml``
    3. ``tiers.<tier>`` block     -- tier-specific overrides inside the merged config
    4. dotted CLI overrides       -- e.g. ``training.rounds=1``

Everything is a flag: switching ``--tier`` never requires editing code. The
resolved config carries ``_meta`` (tier + source files) so runs are traceable.
"""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

VALID_TIERS = ("smoke", "dev", "full")


def load_yaml(path: str | Path) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def save_yaml(config: Dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(config, f, sort_keys=False)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into a copy of ``base`` (override wins)."""
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if key in out and isinstance(out[key], dict) and isinstance(value, dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _coerce(value: str) -> Any:
    """Best-effort scalar coercion for CLI overrides (int/float/bool/None/str)."""
    low = value.lower()
    if low in ("true", "false"):
        return low == "true"
    if low in ("none", "null"):
        return None
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            pass
    return value


def apply_override(config: Dict[str, Any], dotted_key: str, value: Any) -> None:
    """Set ``a.b.c = value`` in-place, creating intermediate dicts as needed."""
    keys = dotted_key.split(".")
    node = config
    for key in keys[:-1]:
        node = node.setdefault(key, {})
        if not isinstance(node, dict):
            raise ValueError(f"Override path '{dotted_key}' traverses a non-dict at '{key}'")
    node[keys[-1]] = value


def parse_overrides(pairs: Optional[List[str]]) -> Dict[str, Any]:
    """Turn ``['training.rounds=1', 'seed=0']`` into a flat dict with coerced values."""
    result: Dict[str, Any] = {}
    for pair in pairs or []:
        if "=" not in pair:
            raise ValueError(f"Override '{pair}' must be of the form key.path=value")
        key, raw = pair.split("=", 1)
        result[key.strip()] = _coerce(raw.strip())
    return result


def resolve_config(
    experiment_path: str | Path,
    tier: str = "smoke",
    overrides: Optional[List[str]] = None,
    base_name: str = "_base.yaml",
) -> Dict[str, Any]:
    """Resolve a fully-merged config for a given tier.

    Parameters
    ----------
    experiment_path : path to the experiment YAML (e.g. ``configs/fedavg.yaml``).
    tier            : one of ``smoke`` / ``dev`` / ``full``.
    overrides       : optional list of ``dotted.key=value`` strings.
    base_name       : name of the shared base file located beside the experiment.
    """
    if tier not in VALID_TIERS:
        raise ValueError(f"tier must be one of {VALID_TIERS}, got {tier!r}")

    experiment_path = Path(experiment_path)
    base_path = experiment_path.parent / base_name

    merged: Dict[str, Any] = load_yaml(base_path) if base_path.exists() else {}
    experiment = load_yaml(experiment_path)
    merged = deep_merge(merged, experiment)

    # Apply the selected tier block, then drop the raw tiers table.
    tiers = merged.pop("tiers", {}) or {}
    if tier in tiers:
        merged = deep_merge(merged, tiers[tier])

    # CLI overrides win over everything.
    for dotted_key, value in parse_overrides(overrides).items():
        apply_override(merged, dotted_key, value)

    merged["_meta"] = {
        "tier": tier,
        "base_file": str(base_path) if base_path.exists() else None,
        "experiment_file": str(experiment_path),
        "overrides": list(overrides or []),
    }
    return merged


def get(config: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    """Read ``a.b.c`` from a nested config, returning ``default`` if absent."""
    node: Any = config
    for key in dotted_key.split("."):
        if not isinstance(node, dict) or key not in node:
            return default
        node = node[key]
    return node
