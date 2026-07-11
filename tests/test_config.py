"""Config resolution + tiering (numpy/yaml only)."""
from pathlib import Path

from fl_med.config import apply_override, deep_merge, get, resolve_config

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_deep_merge_nested():
    base = {"a": {"x": 1, "y": 2}, "b": 3}
    over = {"a": {"y": 20, "z": 30}}
    out = deep_merge(base, over)
    assert out == {"a": {"x": 1, "y": 20, "z": 30}, "b": 3}
    assert base["a"]["y"] == 2  # base untouched


def test_apply_override_and_get():
    cfg = {"training": {"epochs": 5}}
    apply_override(cfg, "training.epochs", 1)
    apply_override(cfg, "federated.rounds", 2)
    assert get(cfg, "training.epochs") == 1
    assert get(cfg, "federated.rounds") == 2
    assert get(cfg, "missing.key", "default") == "default"


def test_resolve_config_applies_smoke_tier():
    cfg = resolve_config(CONFIGS / "fedavg.yaml", tier="smoke")
    assert cfg["_meta"]["tier"] == "smoke"
    assert cfg["model"]["name"] == "small_cnn"      # smoke tier override
    assert cfg["federated"]["rounds"] == 2
    assert cfg["strategy"]["name"] == "fedavg"
    assert "tiers" not in cfg                         # tiers table stripped


def test_resolve_config_full_tier_uses_resnet():
    cfg = resolve_config(CONFIGS / "fedprox.yaml", tier="full")
    assert cfg["model"]["name"] == "resnet18"
    assert cfg["strategy"]["mu"] == 0.1
    assert cfg["data"]["image_size"] == 200


def test_cli_override_wins():
    cfg = resolve_config(CONFIGS / "fedavg.yaml", tier="smoke",
                         overrides=["federated.rounds=7", "seed=123"])
    assert cfg["federated"]["rounds"] == 7
    assert cfg["seed"] == 123
