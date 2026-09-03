#!/usr/bin/env python3
"""Supervise a real localhost Flower federation for the interactive control room."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from fl_med.data.heterogeneity import counts_from_dataset  # noqa: E402


def subprocess_environment() -> dict[str, str]:
    allowed = {
        "PATH", "PYTHONPATH", "LD_LIBRARY_PATH", "DYLD_LIBRARY_PATH", "HOME",
        "XDG_CACHE_HOME", "TORCH_HOME", "CUDA_VISIBLE_DEVICES", "ROCR_VISIBLE_DEVICES",
        "OMP_NUM_THREADS", "MKL_NUM_THREADS", "TMPDIR", "TMP", "TEMP", "VIRTUAL_ENV",
        "CONDA_PREFIX", "SSL_CERT_FILE", "REQUESTS_CA_BUNDLE", "SYSTEMROOT", "WINDIR",
    }
    return {key: value for key, value in os.environ.items() if key in allowed}


def _wait_for_port(port: int, process: subprocess.Popen, timeout: float = 180.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError("Flower coordinator exited before accepting clients")
        with socket.socket() as sock:
            sock.settimeout(0.25)
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        time.sleep(0.25)
    raise TimeoutError("Flower coordinator did not become ready in time")


def _stop(processes: list[subprocess.Popen]) -> None:
    for process in reversed(processes):
        if process.poll() is None:
            process.send_signal(signal.SIGINT)
    deadline = time.monotonic() + 5.0
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=max(0.1, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                process.terminate()
    for process in reversed(processes):
        if process.poll() is None:
            try:
                process.wait(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
    for process in reversed(processes):
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            pass


def _validate_completed_run(out: Path, client_ids: list[int], rounds: int) -> None:
    history_path = out / "history.json"
    status_path = out / "live_status.json"
    try:
        artifact = json.loads(history_path.read_text(encoding="utf-8"))
        status = json.loads(status_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("Flower completed without valid result artifacts") from exc
    observed_rounds = {int(item["round"]) for item in artifact.get("history", [])}
    if rounds not in observed_rounds or status.get("status") != "validating":
        raise RuntimeError("Flower did not publish the requested final round")
    expected = [(round_id, f"center-{client_id}") for round_id in range(1, rounds + 1)
                for client_id in client_ids]
    observed = [
        (int(item.get("round", -1)), str(item.get("tag", "")))
        for item in artifact.get("timings", [])
    ]
    if sorted(observed) != sorted(expected):
        raise RuntimeError("The observed client updates do not exactly match the requested run")
    status.update(status="done", phase="completed")
    events = list(status.get("events") or [])
    events.append({
        "sequence": int(status.get("sequence", 0)) + 1,
        "time": datetime.now(timezone.utc).isoformat(),
        "event": "run_validated",
        "round": rounds,
    })
    status["events"] = events[-80:]
    status["sequence"] = int(status.get("sequence", 0)) + 1
    status["updated_at"] = status["events"][-1]["time"]
    tmp = status_path.with_name(f".{status_path.name}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(status, allow_nan=False), encoding="utf-8")
    os.replace(tmp, status_path)


def _client_ids(raw: str) -> list[int]:
    try:
        values = [int(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("clients must be comma-separated integers") from exc
    if not values or len(values) > 6 or len(values) != len(set(values)):
        raise argparse.ArgumentTypeError("clients must contain one to six unique IDs")
    if any(value < 0 or value > 5 for value in values):
        raise argparse.ArgumentTypeError("client IDs must be between 0 and 5")
    return values


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Run a supervised localhost Flower demo")
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--rounds", type=int, choices=range(1, 31), default=3)
    parser.add_argument("--max-batches", type=int, choices=range(1, 61), default=4)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--tier", choices=("smoke", "dev"), default="dev")
    parser.add_argument("--local-epochs", type=int, choices=range(1, 4), default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clients", type=_client_ids, default=[0, 5])
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--round-timeout", type=float, default=180.0)
    parser.add_argument("--total-timeout", type=float, default=3600.0)
    parser.add_argument("--freeze-edge", action="store_true")
    parser.add_argument("--warm-start", action="store_true")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if not 30.0 <= args.total_timeout <= 21600.0:
        raise SystemExit("--total-timeout must be between 30 and 21600 seconds")
    if not 1024 <= args.port <= 65535:
        raise SystemExit("--port must be between 1024 and 65535")
    if not 0 <= args.seed <= 100000:
        raise SystemExit("--seed must be between 0 and 100000")
    data_root = Path(args.data_root).resolve()
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    count_matrix = counts_from_dataset(data_root / "train")
    global_counts = count_matrix.sum(axis=0)
    if len(global_counts) != 8 or int(global_counts.sum()) <= 0:
        raise ValueError("Training data must contain eight-class count metadata")
    class_counts = ",".join(str(int(value)) for value in global_counts)
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO,
            env=subprocess_environment(), text=True, stderr=subprocess.DEVNULL,
        ).strip()
        dirty = bool(subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=REPO,
            env=subprocess_environment(), text=True, stderr=subprocess.DEVNULL,
        ).strip())
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = None, None
    manifest = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "mode": "networked_flower",
        "dataset_kind": os.environ.get("FL_DEMO_DATASET_KIND", "unknown"),
        "seed": args.seed,
        "tier": args.tier,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "max_batches": args.max_batches,
        "client_ids": args.clients,
        "client_partition_sizes": {
            str(client_id): int(count_matrix[client_id].sum()) for client_id in args.clients
        },
        "global_class_counts": [int(value) for value in global_counts],
        "device": args.device,
        "freeze_edge": args.freeze_edge,
        "warm_start": args.warm_start,
        "git_commit": commit,
        "git_dirty": dirty,
        "dependencies": {
            "flower": importlib.metadata.version("flwr"),
        },
    }
    manifest_path = out / "run_manifest.json"
    manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{os.getpid()}.tmp")
    manifest_tmp.write_text(json.dumps(manifest, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(manifest_tmp, manifest_path)

    for split in ("train", "test"):
        for client_id in args.clients:
            expected = data_root / split / f"client_{client_id}"
            if not expected.is_dir():
                raise FileNotFoundError(f"Missing required data partition: {split}/client_{client_id}")

    processes: list[subprocess.Popen] = []
    handles = []
    stopping = False
    child_env = subprocess_environment()
    child_env["FLWR_TELEMETRY_ENABLED"] = "0"

    def request_stop(_signum=None, _frame=None):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    server_log = (out / "coordinator.log").open("w", encoding="utf-8")
    handles.append(server_log)
    command = [
        sys.executable, "-u", str(REPO / "scripts" / "live" / "server.py"),
        "--data-root", str(data_root),
        "--out", str(out),
        "--rounds", str(args.rounds),
        "--min-clients", str(len(args.clients)),
        "--host", f"127.0.0.1:{args.port}",
        "--device", args.device,
        "--tier", args.tier,
        "--image-size", "64",
        "--batch-size", "16",
        "--num-workers", "0",
        "--round-timeout", str(args.round_timeout),
        "--seed", str(args.seed),
        "--offline-init",
    ]
    checkpoint = REPO / "experiments" / "live" / "pretrained.pt"
    if args.warm_start:
        if not checkpoint.is_file():
            raise FileNotFoundError("The fixed warm-start checkpoint is unavailable")
        command.extend(("--init-model", str(checkpoint)))

    try:
        server = subprocess.Popen(
            command, cwd=REPO, env=child_env,
            stdout=server_log, stderr=subprocess.STDOUT,
        )
        processes.append(server)
        _wait_for_port(args.port, server)

        for client_id in args.clients:
            client_log = (out / f"center-{client_id}.log").open("w", encoding="utf-8")
            handles.append(client_log)
            client_command = [
                sys.executable, "-u", str(REPO / "scripts" / "live" / "client.py"),
                "--data-root", str(data_root),
                "--server", f"127.0.0.1:{args.port}",
                "--client-id", str(client_id),
                "--label", f"center-{client_id}",
                "--device", args.device,
                "--tier", args.tier,
                "--local-epochs", str(args.local_epochs),
                "--max-batches", str(args.max_batches),
                "--image-size", "64",
                "--batch-size", "16",
                "--num-workers", "0",
                "--seed", str(args.seed),
                "--class-counts", class_counts,
            ]
            if args.freeze_edge and client_id == 5:
                client_command.append("--freeze-backbone")
            processes.append(
                subprocess.Popen(
                    client_command,
                    cwd=REPO,
                    env=child_env,
                    stdout=client_log,
                    stderr=subprocess.STDOUT,
                )
            )
            time.sleep(0.35)

        deadline = time.monotonic() + args.total_timeout
        while server.poll() is None and not stopping:
            failed_clients = [p for p in processes[1:] if p.poll() not in (None, 0)]
            if failed_clients:
                raise RuntimeError("A Flower client exited before the federation completed")
            if time.monotonic() >= deadline:
                raise TimeoutError("The Flower run exceeded its total time limit")
            time.sleep(0.25)
        if stopping:
            return 130
        code = int(server.returncode or 0)
        if code == 0:
            _validate_completed_run(out, args.clients, args.rounds)
        return code
    finally:
        _stop(processes)
        for handle in handles:
            handle.close()


if __name__ == "__main__":
    raise SystemExit(main())
