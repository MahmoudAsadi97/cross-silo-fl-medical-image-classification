"""Controller validation and privacy-planning tests (no training runtime required)."""
from __future__ import annotations

import importlib.util
import json
import threading
import time
import urllib.request
from functools import partial
from http.server import ThreadingHTTPServer
from pathlib import Path

import pytest


REPO = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "demo_control_server", REPO / "scripts" / "demo" / "control_server.py"
)
assert SPEC and SPEC.loader
CONTROL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CONTROL)

REAL_CENTER_SIZES = [7947, 2531, 2156, 1448, 525, 281]


def test_target_epsilon_calibrates_noise_for_complete_schedule():
    plan = CONTROL.privacy_plan(
        target_epsilon=8.0,
        delta=1e-5,
        rounds=30,
        local_epochs=1,
        batch_size=128,
        client_sizes=REAL_CENTER_SIZES,
    )
    assert plan["noise_multiplier"] == pytest.approx(2.2242, abs=0.01)
    assert plan["epsilon_max"] <= 8.0 + 1e-9
    assert set(plan["epsilon_by_client"]) == {"0", "1", "2", "3", "4", "5"}
    assert plan["accuracy_prediction"] is None


def test_tighter_target_requires_more_noise():
    common = dict(
        delta=1e-5,
        rounds=10,
        local_epochs=1,
        batch_size=128,
        client_sizes=REAL_CENTER_SIZES,
    )
    tight = CONTROL.privacy_plan(target_epsilon=2.0, **common)
    loose = CONTROL.privacy_plan(target_epsilon=10.0, **common)
    assert tight["noise_multiplier"] > loose["noise_multiplier"]


def test_delta_limit_uses_largest_center():
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.privacy_plan(
            target_epsilon=8.0,
            delta=2e-4,
            rounds=3,
            local_epochs=1,
            batch_size=128,
            client_sizes=REAL_CENTER_SIZES,
        )
    assert error.value.code == "invalid_delta"


def test_networked_mode_rejects_dp():
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.validate_run(
            {"mode": "networked", "strategy": "dp_fedavg"}, REAL_CENTER_SIZES
        )
    assert error.value.code == "live_dp_not_supported"


@pytest.mark.parametrize("key", ["mode", "strategy", "device"])
def test_choice_parameters_reject_non_strings(key):
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.validate_run({key: []}, REAL_CENTER_SIZES)
    assert error.value.code == "invalid_parameter"


def test_network_seed_is_preserved():
    spec = CONTROL.validate_run(
        {"mode": "networked", "strategy": "fedavg", "seed": 73}, REAL_CENTER_SIZES
    )
    assert spec["seed"] == 73


def test_dp_requires_full_local_pass():
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.validate_run(
            {"strategy": "dp_fedavg", "max_batches": 4}, REAL_CENTER_SIZES
        )
    assert error.value.code == "dp_requires_full_pass"


@pytest.mark.parametrize("key,value", [("command", "id"), ("output", "../elsewhere")])
def test_unknown_process_inputs_are_rejected(key, value):
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.validate_run({key: value}, REAL_CENTER_SIZES)
    assert error.value.code == "unknown_parameter"


def test_host_validation_is_exact():
    assert CONTROL.valid_host("127.0.0.1:8765", 8765)
    assert CONTROL.valid_host("localhost:8765", 8765)
    assert not CONTROL.valid_host("127.0.0.1:8765.attacker.example", 8765)
    assert not CONTROL.valid_host("attacker.example", 8765)


def test_atomic_json_always_leaves_valid_document(tmp_path):
    path = tmp_path / "status.json"
    CONTROL.atomic_json(path, {"status": "training", "round": 1})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "status": "training",
        "round": 1,
    }
    assert not list(tmp_path.glob(".*.tmp"))


def test_private_status_suppresses_unbudgeted_training_metrics(tmp_path):
    status_path = tmp_path / "status.json"
    CONTROL.atomic_json(status_path, {
        "status": "training",
        "clients": [{
            "client_id": 0,
            "host": "private-host",
            "path": "/private/path",
            "status": "complete",
            "n": 10,
            "num_samples": 10,
            "examples_seen": 10,
            "train_loss": 1.2,
            "train_bal_acc": 0.4,
            "train_balanced_accuracy": 0.4,
            "dp_steps": 2,
            "dp_sample_rate": 0.5,
        }],
        "events": [{
            "event": "client_completed",
            "client_id": 0,
            "num_samples": 10,
            "examples_seen": 10,
            "train_loss": 1.2,
            "train_balanced_accuracy": 0.4,
            "host": "private-host",
            "path": "/private/path",
            "metrics": {"private": True},
            "dp_steps": 2,
        }],
    })
    manager = CONTROL.RunManager(tmp_path, "fixture", [10] * 6)
    manager.run = {
        "run_id": "test-run",
        "status_file": status_path,
        "spec": {
            "mode": "experiment",
            "rounds": 1,
            "strategy": "dp_fedavg",
            "privacy": {"enabled": True},
        },
        "started_monotonic": time.monotonic(),
    }
    public = manager.status()
    assert public["clients"][0]["dp_steps"] == 2
    forbidden = {
        "n", "num_samples", "examples_seen", "train_loss", "train_bal_acc",
        "train_balanced_accuracy", "host", "path", "metrics",
    }
    assert forbidden.isdisjoint(public["clients"][0])
    assert forbidden.isdisjoint(public["events"][0])


def test_subprocess_environment_does_not_forward_credentials(monkeypatch):
    monkeypatch.setenv("GITHUB_TOKEN", "not-forwarded")
    monkeypatch.setenv("PATH", "/safe/path")
    environment = CONTROL.subprocess_environment()
    assert environment["PATH"] == "/safe/path"
    assert "GITHUB_TOKEN" not in environment


def test_data_validation_rejects_out_of_range_classes(tmp_path):
    for split in ("train", "test"):
        for client_id in range(6):
            class_id = 8 if split == "train" and client_id == 0 else 0
            folder = tmp_path / split / f"client_{client_id}" / f"class_{class_id}"
            folder.mkdir(parents=True)
            (folder / "sample.png").write_bytes(b"index-only fixture")
    with pytest.raises(CONTROL.RequestProblem) as error:
        CONTROL.validate_data_root(tmp_path)
    assert error.value.code == "data_unavailable"


def test_http_health_and_static_security_contract(tmp_path, monkeypatch):
    (tmp_path / "index.html").write_text("<!doctype html><title>demo</title>", encoding="utf-8")

    class Manager:
        dataset_kind = "fixture"
        client_sizes = [1] * 6

        @staticmethod
        def _active():
            return False

    monkeypatch.setattr(CONTROL, "dependency_usable", lambda _name: False)
    monkeypatch.setattr(CONTROL, "flower_runtime_usable", lambda: False)
    monkeypatch.setattr(CONTROL, "cuda_available", lambda: False)
    handler = partial(CONTROL.DemoHandler, directory=str(tmp_path))
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    server.manager = Manager()
    server.demo_token = "test-token"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_port}"
        with urllib.request.urlopen(base + "/", timeout=3) as response:
            assert response.status == 200
            assert "default-src 'self'" in response.headers["Content-Security-Policy"]
        with urllib.request.urlopen(base + "/api/v1/health", timeout=3) as response:
            health = json.load(response)
            assert response.status == 200
            assert health["dataset"]["kind"] == "fixture"
            assert health["training_profile"]["model"] == "SmallCNN + GroupNorm"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=3)
