#!/usr/bin/env python3
"""Loopback-only control service for the interactive federated-learning demo."""
from __future__ import annotations

import argparse
import importlib
import importlib.metadata
import json
import math
import os
import secrets
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
import webbrowser
from datetime import datetime, timezone
from functools import lru_cache, partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

REPO = Path(__file__).resolve().parents[2]
DIST = REPO / "dist"
RUNS_ROOT = REPO / "experiments" / "site_runs"
FIXTURE_ROOT = REPO / "data" / "fixtures" / "fed_isic2019_tiny" / "raw"
FED_ISIC2019_TRAIN_SIZES = [7947, 2531, 2156, 1448, 525, 281]
sys.path.insert(0, str(REPO / "src"))

from fl_med.privacy.accounting import compute_epsilon  # noqa: E402

EXPERIMENT_CONFIGS = {
    "fedavg": "fedavg.yaml",
    "fedprox": "fedprox.yaml",
    "scaffold": "scaffold.yaml",
    "fedadam": "fedadam.yaml",
    "dp_fedavg": "dp_fedavg.yaml",
}
ACTIVE_STATES = {"preparing", "preflight", "waiting", "training", "validating", "stopping"}


class RequestProblem(ValueError):
    def __init__(self, code: str, message: str, status: int = HTTPStatus.BAD_REQUEST):
        super().__init__(message)
        self.code = code
        self.status = int(status)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


@lru_cache(maxsize=None)
def dependency_usable(name: str) -> bool:
    """Return true only when a runtime dependency can actually be imported."""
    try:
        importlib.import_module(name)
        return True
    except Exception:  # noqa: BLE001
        return False


def flower_runtime_usable() -> bool:
    if not dependency_usable("flwr"):
        return False
    try:
        major, minor, *_ = (
            int(part) for part in importlib.metadata.version("flwr").split(".")
        )
        return major == 1 and 7 <= minor <= 12
    except (ValueError, importlib.metadata.PackageNotFoundError):
        return False


def subprocess_environment() -> dict[str, str]:
    """Pass only runtime settings needed by local training subprocesses."""
    allowed = {
        "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "HOME",
        "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR", "TMP", "TEMP", "VIRTUAL_ENV",
        "CONDA_PREFIX", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "SYSTEMROOT", "WINDIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def cuda_available() -> bool:
    if not dependency_usable("torch"):
        return False
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:  # noqa: BLE001
        return False


def available_loopback_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def validate_data_root(path: Path) -> list[int]:
    from fl_med.data.dataset import ISICFederatedFolderDataset
    from PIL import Image

    sizes = []
    expected = {f"client_{client_id}" for client_id in range(6)}
    try:
        for split in ("train", "test"):
            split_root = path / split
            observed = {entry.name for entry in split_root.iterdir() if entry.is_dir()}
            if observed != expected:
                raise ValueError(f"{split} must contain exactly client_0 through client_5")
        for client_id in range(6):
            train = ISICFederatedFolderDataset(path / "train" / f"client_{client_id}")
            test = ISICFederatedFolderDataset(path / "test" / f"client_{client_id}")
            for dataset in (train, test):
                if any(label < 0 or label >= 8 for _, label, _ in dataset.samples):
                    raise ValueError("class IDs must be between 0 and 7")
                if any(observed_id != client_id for _, _, observed_id in dataset.samples):
                    raise ValueError("client IDs must match their partition directory")
                probes = {dataset.samples[0][0], dataset.samples[-1][0]}
                for image_path in probes:
                    with Image.open(image_path) as image:
                        image.verify()
            sizes.append(len(train))
    except (FileNotFoundError, OSError, RuntimeError, ValueError) as exc:
        raise RequestProblem(
            "data_unavailable", "The six train/test center partitions are unavailable", 412
        ) from exc
    return sizes


def bounded_int(payload: dict, key: str, default: int, low: int, high: int) -> int:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int) or not low <= value <= high:
        raise RequestProblem("invalid_parameter", f"{key} must be an integer from {low} to {high}")
    return value


def bounded_float(payload: dict, key: str, default: float, low: float, high: float) -> float:
    value = payload.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise RequestProblem("invalid_parameter", f"{key} must be numeric")
    value = float(value)
    if not math.isfinite(value) or not low <= value <= high:
        raise RequestProblem("invalid_parameter", f"{key} must be between {low} and {high}")
    return value


def privacy_plan(
    *, target_epsilon: float, delta: float, rounds: int, local_epochs: int,
    batch_size: int, client_sizes: list[int],
) -> dict:
    if not 0.0 < delta < 1.0 / max(client_sizes):
        raise RequestProblem("invalid_delta", "delta must be positive and smaller than 1/N for every center")

    batches_by_client = [max(1, math.ceil(samples / batch_size)) for samples in client_sizes]

    def client_epsilon(sigma: float, steps_per_epoch: int) -> float:
        return compute_epsilon(
            sample_rate=1.0 / steps_per_epoch,
            noise_multiplier=sigma,
            steps=steps_per_epoch * rounds * local_epochs,
            delta=delta,
        )

    def epsilon_by_client(sigma: float) -> dict[str, float]:
        values = {}
        for client_id, steps_per_epoch in enumerate(batches_by_client):
            values[str(client_id)] = client_epsilon(sigma, steps_per_epoch)
        return values

    low, high = 0.1, 32.0
    worst_batches = min(batches_by_client)
    if client_epsilon(high, worst_batches) > target_epsilon:
        raise RequestProblem("privacy_target_unreachable", "Target epsilon needs more noise than this demo permits")
    for _ in range(40):
        mid = (low + high) / 2.0
        if client_epsilon(mid, worst_batches) > target_epsilon:
            low = mid
        else:
            high = mid
    per_client = epsilon_by_client(high)
    while max(per_client.values()) > target_epsilon:
        high *= 1.000001
        per_client = epsilon_by_client(high)
    return {
        "scope": "record_level_dp_sgd_per_center",
        "target_epsilon": target_epsilon,
        "delta": delta,
        "noise_multiplier": high,
        "batch_size": batch_size,
        "epsilon_by_client": per_client,
        "epsilon_max": max(per_client.values()),
        "accuracy_prediction": None,
    }


def validate_run(payload: dict, client_sizes: list[int]) -> dict:
    allowed = {
        "mode", "strategy", "rounds", "local_epochs", "max_batches", "seed", "device",
        "target_epsilon", "delta", "clip_norm", "freeze_edge", "warm_start",
    }
    unknown = sorted(set(payload) - allowed)
    if unknown:
        raise RequestProblem("unknown_parameter", f"Unknown parameter: {unknown[0]}")

    mode = payload.get("mode", "experiment")
    if not isinstance(mode, str) or mode not in {"experiment", "networked"}:
        raise RequestProblem("invalid_parameter", "mode must be experiment or networked")
    strategy = payload.get("strategy", "fedavg")
    if not isinstance(strategy, str) or strategy not in EXPERIMENT_CONFIGS:
        raise RequestProblem("invalid_parameter", "Unsupported strategy")
    if mode == "networked" and strategy != "fedavg":
        code = "live_dp_not_supported" if strategy == "dp_fedavg" else "network_strategy_not_supported"
        raise RequestProblem(code, "Networked mode currently supports FedAvg without DP")

    rounds = bounded_int(payload, "rounds", 3, 1, 30)
    local_epochs = bounded_int(payload, "local_epochs", 1, 1, 3)
    seed = bounded_int(payload, "seed", 42, 0, 100000)
    device = payload.get("device", "cpu")
    if not isinstance(device, str) or device not in {"cpu", "cuda"}:
        raise RequestProblem("invalid_parameter", "device must be cpu or cuda")
    freeze_edge = payload.get("freeze_edge", True)
    warm_start = payload.get("warm_start", False)
    if not isinstance(freeze_edge, bool) or not isinstance(warm_start, bool):
        raise RequestProblem("invalid_parameter", "freeze_edge and warm_start must be boolean")

    spec = {
        "mode": mode,
        "strategy": strategy,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "seed": seed,
        "device": device,
        "freeze_edge": freeze_edge,
        "warm_start": warm_start,
        "privacy": {"enabled": False},
    }
    if strategy == "dp_fedavg":
        if payload.get("max_batches") is not None:
            raise RequestProblem(
                "dp_requires_full_pass",
                "DP mode uses every logical batch in the Poisson-sampled schedule so accounting matches training",
            )
        target = bounded_float(payload, "target_epsilon", 8.0, 1.0, 50.0)
        delta = bounded_float(payload, "delta", 1e-5, 1e-8, 1e-3)
        clip = bounded_float(payload, "clip_norm", 1.0, 0.1, 5.0)
        spec["privacy"] = {
            "enabled": True,
            "max_grad_norm": clip,
            **privacy_plan(
                target_epsilon=target,
                delta=delta,
                rounds=rounds,
                local_epochs=local_epochs,
                batch_size=128,
                client_sizes=client_sizes,
            ),
        }
        spec["max_batches"] = None
    else:
        spec["max_batches"] = bounded_int(payload, "max_batches", 4, 1, 60)
    return spec


class RunManager:
    def __init__(self, data_root: Path, dataset_kind: str, client_sizes: list[int]):
        self.data_root = data_root
        self.dataset_kind = dataset_kind
        self.client_sizes = client_sizes
        self.lock = threading.RLock()
        self.process: subprocess.Popen | None = None
        self.run: dict | None = None
        self.closing = False

    def _active(self) -> bool:
        return self.process is not None and self.process.poll() is None

    def start(self, payload: dict) -> dict:
        with self.lock:
            if self.closing:
                raise RequestProblem("controller_stopping", "The local controller is stopping", 503)
            if self._active():
                raise RequestProblem("run_in_progress", "A training run is already active", 409)
            spec = validate_run(payload, self.client_sizes)
            if not dependency_usable("torch") or not dependency_usable("torchvision"):
                raise RequestProblem("runtime_unavailable", "The Torch runtime is unavailable", 412)
            if spec["device"] == "cuda" and not cuda_available():
                raise RequestProblem("runtime_unavailable", "CUDA is not available", 412)
            if spec["mode"] == "networked" and not flower_runtime_usable():
                raise RequestProblem("runtime_unavailable", "Flower is not installed", 412)
            if spec["strategy"] == "dp_fedavg" and not dependency_usable("opacus"):
                raise RequestProblem("runtime_unavailable", "Opacus is not installed", 412)
            if self.dataset_kind == "fixture":
                spec["warm_start"] = False
            if spec["mode"] == "networked" and spec["warm_start"]:
                checkpoint = REPO / "experiments" / "live" / "pretrained.pt"
                if not checkpoint.is_file():
                    raise RequestProblem(
                        "warm_start_unavailable", "The fixed warm-start checkpoint is unavailable", 412
                    )

            run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
            out = (RUNS_ROOT / run_id).resolve()
            out.mkdir(parents=True, exist_ok=False)
            status_file = out / ("live_status.json" if spec["mode"] == "networked" else "status.json")
            log_handle = (out / "runner.log").open("w", encoding="utf-8")
            command, env = self._command(spec, out, status_file, run_id)
            initial = {
                "schema_version": 1,
                "run_id": run_id,
                "mode": spec["mode"],
                "dataset_kind": self.dataset_kind,
                "status": "preparing",
                "phase": "preflight",
                "round": 0,
                "total_rounds": spec["rounds"],
                "strategy": spec["strategy"],
                "history": [],
                "clients": [],
                "privacy": spec["privacy"],
                "created_at": utc_now(),
                "updated_at": utc_now(),
            }
            atomic_json(status_file, initial)
            try:
                self.process = subprocess.Popen(
                    command,
                    cwd=REPO,
                    env=env,
                    stdout=log_handle,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
            except OSError as exc:
                log_handle.close()
                initial.update(
                    status="failed",
                    phase="failed",
                    error_code="training_process_unavailable",
                    updated_at=utc_now(),
                )
                atomic_json(status_file, initial)
                raise RequestProblem(
                    "runtime_unavailable", "The training process could not be started", 412
                ) from exc
            self.run = {
                "run_id": run_id,
                "out": out,
                "status_file": status_file,
                "spec": spec,
                "started_monotonic": time.monotonic(),
                "log_handle": log_handle,
                "cancel_requested": False,
            }
            threading.Thread(target=self._watch, args=(self.process, self.run), daemon=True).start()
            return self.status()

    def _command(self, spec: dict, out: Path, status_file: Path, run_id: str):
        env = subprocess_environment()
        env.update({
            "DATA_ROOT": str(self.data_root),
            "PYTHONUNBUFFERED": "1",
            "FL_DEMO_RUN_ID": run_id,
            "FL_DEMO_DATASET_KIND": self.dataset_kind,
        })
        if spec["mode"] == "networked":
            port = available_loopback_port()
            command = [
                sys.executable, "-u", str(REPO / "scripts" / "demo" / "network_runner.py"),
                "--data-root", str(self.data_root),
                "--out", str(out),
                "--rounds", str(spec["rounds"]),
                "--max-batches", str(spec["max_batches"]),
                "--device", spec["device"],
                "--clients", "0,5",
                "--port", str(port),
                "--tier", "smoke" if self.dataset_kind == "fixture" else "dev",
                "--local-epochs", str(spec["local_epochs"]),
                "--seed", str(spec["seed"]),
            ]
            if spec["freeze_edge"]:
                command.append("--freeze-edge")
            if spec["warm_start"]:
                command.append("--warm-start")
            return command, env

        command = [
            sys.executable, "-u", str(REPO / "scripts" / "run_experiment.py"),
            "--config", str(REPO / "configs" / EXPERIMENT_CONFIGS[spec["strategy"]]),
            "--tier", "smoke" if self.dataset_kind == "fixture" else "dev",
            "--seed", str(spec["seed"]),
            "--output", str(out),
            "--device", spec["device"],
            "--status-file", str(status_file),
            f"data.root={self.data_root}",
            f"federated.rounds={spec['rounds']}",
            f"federated.local_epochs={spec['local_epochs']}",
            "data.num_workers=0",
        ]
        if spec["strategy"] == "dp_fedavg":
            privacy = spec["privacy"]
            command.extend((
                "federated.max_batches=null",
                "data.batch_size=128",
                "privacy.enabled=true",
                f"privacy.noise_multiplier={privacy['noise_multiplier']}",
                f"privacy.max_grad_norm={privacy['max_grad_norm']}",
                f"privacy.target_delta={privacy['delta']}",
                f"privacy.target_epsilon={privacy['target_epsilon']}",
                "privacy.max_physical_batch_size=8",
            ))
        else:
            command.append(f"federated.max_batches={spec['max_batches']}")
        return command, env

    def _watch(self, process: subprocess.Popen, run: dict) -> None:
        code = process.wait()
        with self.lock:
            run["log_handle"].close()
            current = read_json(run["status_file"])
            if run.get("cancel_requested"):
                current.update(status="cancelled", phase="cancelled")
            elif code != 0:
                current.update(status="failed", phase="failed", error_code="training_process_failed")
            elif current.get("status") not in {"completed", "done"}:
                current.update(status="completed", phase="completed")
            current["updated_at"] = utc_now()
            run["finished_elapsed"] = round(time.monotonic() - run["started_monotonic"], 1)
            atomic_json(run["status_file"], current)

    def status(self) -> dict:
        with self.lock:
            if self.run is None:
                return {"schema_version": 1, "status": "idle", "dataset_kind": self.dataset_kind}
            raw = read_json(self.run["status_file"])
            spec = self.run["spec"]
            history = []
            for point in raw.get("history", []):
                if not isinstance(point, dict):
                    continue
                history.append({
                    "round": point.get("round"),
                    "balanced_accuracy": point.get("test_balanced_accuracy", point.get("bal_acc")),
                    "macro_f1": point.get("test_macro_f1", point.get("macro_f1")),
                    "accuracy": point.get("test_accuracy", point.get("accuracy")),
                    "loss": point.get("test_loss", point.get("loss")),
                    "client_drift": point.get("client_drift"),
                    "epsilon_max": point.get("epsilon_max"),
                    "epsilon_mean": point.get("epsilon_mean"),
                    "epsilon_by_client": point.get("epsilon_by_client"),
                    "delta": point.get("delta"),
                })
            private_run = bool(spec["privacy"].get("enabled"))
            clients = []
            for client in raw.get("clients", []):
                if not isinstance(client, dict):
                    continue
                client_keys = [
                    "client_id", "tag", "status", "device", "fit_seconds", "n", "num_samples",
                    "examples_seen", "train_loss", "train_bal_acc", "train_balanced_accuracy", "freeze_backbone",
                    "dp_steps", "dp_sample_rate",
                ]
                if private_run:
                    client_keys = [
                        "client_id", "tag", "status", "device", "fit_seconds",
                        "freeze_backbone", "dp_steps", "dp_sample_rate", "epsilon", "delta",
                    ]
                clients.append({key: client.get(key) for key in client_keys if key in client})
            event_keys = {
                "sequence", "time", "event", "round", "total_rounds", "client_id",
                "participating_clients", "dp_steps", "dp_sample_rate", "fit_seconds",
                "epsilon", "delta",
            }
            if not private_run:
                event_keys.update({
                    "num_samples", "examples_seen", "train_loss", "train_balanced_accuracy",
                })
            events = []
            for event in raw.get("events", []):
                if isinstance(event, dict):
                    events.append({key: event[key] for key in event_keys if key in event})
            return {
                "schema_version": 1,
                "run_id": self.run["run_id"],
                "mode": spec["mode"],
                "dataset_kind": self.dataset_kind,
                "status": raw.get("status", "preparing"),
                "phase": raw.get("phase", raw.get("status", "preparing")),
                "round": raw.get("round", 0),
                "active_round": raw.get("active_round", raw.get("round", 0)),
                "completed_rounds": raw.get(
                    "completed_rounds",
                    history[-1].get("round", 0) if history else 0,
                ),
                "total_rounds": spec["rounds"],
                "strategy": spec["strategy"],
                "privacy": spec["privacy"],
                "history": history,
                "clients": clients,
                "events": events[-40:],
                "elapsed_seconds": self.run.get(
                    "finished_elapsed",
                    round(time.monotonic() - self.run["started_monotonic"], 1),
                ),
                "updated_at": raw.get("updated_at"),
                "error_code": raw.get("error_code"),
            }

    def cancel(self) -> dict:
        with self.lock:
            if not self._active():
                if self.run is not None:
                    return self.status()
                raise RequestProblem("no_active_run", "There is no active run to cancel", 409)
            self.run["cancel_requested"] = True
            current = read_json(self.run["status_file"])
            current.update(status="stopping", phase="stopping", updated_at=utc_now())
            atomic_json(self.run["status_file"], current)
            process = self.process
        try:
            os.killpg(process.pid, signal.SIGINT)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=8.0)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=2.0)
                except subprocess.TimeoutExpired:
                    pass
        return self.status()

    def shutdown(self) -> None:
        """Close the start gate, then stop any process created before it closed."""
        with self.lock:
            self.closing = True
            active = self._active()
        if active:
            self.cancel()


def valid_host(raw: str, port: int) -> bool:
    return raw in {f"127.0.0.1:{port}", f"localhost:{port}", f"[::1]:{port}"}


class DemoHandler(SimpleHTTPRequestHandler):
    server_version = "FederatedDemo/1.0"

    def end_headers(self):
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Referrer-Policy", "no-referrer")
        self.send_header("X-Frame-Options", "DENY")
        self.send_header(
            "Content-Security-Policy",
            "default-src 'self'; img-src 'self' data:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; object-src 'none'; frame-ancestors 'none'",
        )
        super().end_headers()

    def log_message(self, format, *args):
        return

    def _json(self, payload: dict, status: int = HTTPStatus.OK):
        body = json.dumps(payload, allow_nan=False).encode("utf-8")
        self.send_response(int(status))
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _problem(self, problem: RequestProblem):
        self._json({"error": {"code": problem.code, "message": str(problem)}}, problem.status)

    def _authorized(self) -> bool:
        if not valid_host(self.headers.get("Host", ""), self.server.server_port):
            return False
        token = self.headers.get("X-Demo-Token", "")
        if not secrets.compare_digest(token, self.server.demo_token):
            return False
        origin = self.headers.get("Origin")
        if origin:
            parsed = urlparse(origin)
            if parsed.scheme != "http" or not valid_host(parsed.netloc, self.server.server_port):
                return False
        return True

    def _body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0"))
        except ValueError as exc:
            raise RequestProblem("invalid_request", "Invalid content length") from exc
        if length < 0 or length > 32768:
            raise RequestProblem("request_too_large", "Request body is too large", 413)
        try:
            payload = json.loads(self.rfile.read(length) or b"{}")
        except json.JSONDecodeError as exc:
            raise RequestProblem("invalid_json", "Request body must be valid JSON") from exc
        if not isinstance(payload, dict):
            raise RequestProblem("invalid_json", "Request body must be a JSON object")
        return payload

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/api/v1/health":
            if not valid_host(self.headers.get("Host", ""), self.server.server_port):
                self._json({"error": {"code": "forbidden", "message": "Request denied"}}, 403)
                return
            self._json({
                "schema_version": 1,
                "runtime": "local",
                "connected": True,
                "token": self.server.demo_token,
                "dataset": {
                    "kind": self.server.manager.dataset_kind,
                    "ready": True,
                    "centers": len(self.server.manager.client_sizes),
                    "training_images": sum(self.server.manager.client_sizes),
                    "center_sizes": self.server.manager.client_sizes,
                },
                "training_profile": {
                    "tier": "smoke" if self.server.manager.dataset_kind == "fixture" else "dev",
                    "model": (
                        "SmallCNN + GroupNorm"
                        if self.server.manager.dataset_kind == "fixture"
                        else "ResNet-18 + GroupNorm"
                    ),
                },
                "capabilities": {
                    "experiment": dependency_usable("torch") and dependency_usable("torchvision"),
                    "networked": (
                        dependency_usable("torch")
                        and dependency_usable("torchvision")
                        and flower_runtime_usable()
                    ),
                    "dp": (
                        dependency_usable("torch")
                        and dependency_usable("torchvision")
                        and dependency_usable("opacus")
                    ),
                    "cuda": cuda_available(),
                },
                "active_run": self.server.manager._active(),
            })
            return
        if path == "/api/v1/runs/current":
            if not self._authorized():
                self._json({"error": {"code": "forbidden", "message": "Request denied"}}, 403)
                return
            try:
                self._json(self.server.manager.status())
            except Exception:  # noqa: BLE001
                self._json({"error": {"code": "internal_error", "message": "Status unavailable"}}, 500)
            return
        if path.startswith("/api/"):
            self._json({"error": {"code": "not_found", "message": "Unknown API route"}}, 404)
            return
        if path == "/":
            self.path = "/index.html"
        super().do_GET()

    def do_POST(self):
        if not self._authorized():
            self._json({"error": {"code": "forbidden", "message": "Request denied"}}, 403)
            return
        path = urlparse(self.path).path
        try:
            if path == "/api/v1/runs":
                self._json(self.server.manager.start(self._body()), HTTPStatus.ACCEPTED)
                return
            if path == "/api/v1/runs/current/cancel":
                self._json(self.server.manager.cancel(), HTTPStatus.ACCEPTED)
                return
            raise RequestProblem("not_found", "Unknown API route", 404)
        except RequestProblem as problem:
            self._problem(problem)
        except Exception:  # noqa: BLE001
            self._json({
                "error": {"code": "internal_error", "message": "The local run could not be handled"}
            }, 500)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Serve the federated-learning demo locally")
    parser.add_argument("--data-root", default=os.environ.get("DATA_ROOT"))
    parser.add_argument("--allow-fixture", action="store_true")
    parser.add_argument("--host", default="127.0.0.1", choices=("127.0.0.1", "localhost"))
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-open", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 1024 <= args.port <= 65535:
        raise SystemExit("--port must be between 1024 and 65535")
    if args.data_root:
        data_root = Path(args.data_root).expanduser().resolve()
        dataset_kind = "fixture" if data_root == FIXTURE_ROOT.resolve() else "folder"
    elif args.allow_fixture:
        data_root = FIXTURE_ROOT.resolve()
        dataset_kind = "fixture"
    else:
        raise SystemExit("Set DATA_ROOT or pass --data-root. Use --allow-fixture only for testing.")
    client_sizes = validate_data_root(data_root)
    if dataset_kind == "folder" and client_sizes == FED_ISIC2019_TRAIN_SIZES:
        dataset_kind = "fed_isic_sized"
    RUNS_ROOT.mkdir(parents=True, exist_ok=True)
    manager = RunManager(data_root, dataset_kind, client_sizes)
    handler = partial(DemoHandler, directory=str(DIST))
    server = ThreadingHTTPServer((args.host, args.port), handler)
    server.manager = manager
    server.demo_token = secrets.token_urlsafe(32)
    url = f"http://{args.host}:{args.port}/"
    print(f"Federated demo ready at {url}")
    if not args.no_open:
        threading.Timer(0.6, lambda: webbrowser.open(url)).start()
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    previous_sighup = signal.getsignal(signal.SIGHUP) if hasattr(signal, "SIGHUP") else None

    def terminate(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    if hasattr(signal, "SIGHUP"):
        signal.signal(signal.SIGHUP, terminate)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        manager.shutdown()
        server.server_close()
        signal.signal(signal.SIGTERM, previous_sigterm)
        if hasattr(signal, "SIGHUP"):
            signal.signal(signal.SIGHUP, previous_sighup)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
