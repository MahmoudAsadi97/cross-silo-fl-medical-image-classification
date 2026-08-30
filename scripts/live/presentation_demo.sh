#!/usr/bin/env bash
# One-command presentation supervisor for the real laptop + Raspberry Pi demo.
#
# The Pi reaches Flower through an SSH reverse tunnel:
#   Pi 127.0.0.1:18080 -> SSH -> WSL 127.0.0.1:8080
# This avoids Windows portproxy, firewall, and changing laptop/WSL addresses.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd -- "$SCRIPT_DIR/../.." && pwd)"
SCRIPT_PATH="$SCRIPT_DIR/$(basename -- "${BASH_SOURCE[0]}")"
DEFAULT_CONFIG="$REPO/configs/presentation_demo.env"

MODE="run"
NO_BROWSER=0
CONFIG_FILE="$DEFAULT_CONFIG"

usage() {
    cat <<'EOF'
Usage:
  bash scripts/live/presentation_demo.sh
  bash scripts/live/presentation_demo.sh --check-only
  bash scripts/live/presentation_demo.sh --configure

Options:
  --configure       Save the one-time Raspberry Pi settings, then exit.
  --check-only      Run every local/remote preflight check without training.
  --no-browser      Do not open the dashboard automatically.
  --config PATH     Use a different local configuration file.
  -h, --help        Show this help.

On presentation day, double-click START_PRESENTATION_DEMO.cmd or run the first
command above. The first run prompts for the Pi SSH address and saves it locally.
EOF
}

while (($#)); do
    case "$1" in
        --configure) MODE="configure"; shift ;;
        --check-only) MODE="check"; shift ;;
        --no-browser) NO_BROWSER=1; shift ;;
        --config)
            [[ $# -ge 2 ]] || { echo "ERROR: --config requires a path" >&2; exit 2; }
            CONFIG_FILE="$2"; shift 2 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "ERROR: unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ "$CONFIG_FILE" != /* ]]; then
    CONFIG_FILE="$REPO/$CONFIG_FILE"
fi

if [[ -t 1 ]]; then
    C_BLUE=$'\033[1;34m'; C_GREEN=$'\033[1;32m'; C_YELLOW=$'\033[1;33m'
    C_RED=$'\033[1;31m'; C_RESET=$'\033[0m'
else
    C_BLUE=""; C_GREEN=""; C_YELLOW=""; C_RED=""; C_RESET=""
fi

step() { printf '\n%s==>%s %s\n' "$C_BLUE" "$C_RESET" "$*"; }
ok() { printf '%sOK%s  %s\n' "$C_GREEN" "$C_RESET" "$*"; }
warn() { printf '%sWARN%s  %s\n' "$C_YELLOW" "$C_RESET" "$*" >&2; }
die() {
    local code="$1"; shift
    printf '\n%sERROR%s  %s\n' "$C_RED" "$C_RESET" "$*" >&2
    exit "$code"
}

configure() {
    local pi_target pi_repo pi_data laptop_data identity answer default_identity=""
    local PI_SSH_TARGET="" PI_REPO="" PI_DATA_ROOT="" LAPTOP_DATA_ROOT=""
    local PI_IDENTITY_FILE="" PI_BOOT_WAIT=""

    if [[ -f "$CONFIG_FILE" ]]; then
        # shellcheck disable=SC1090
        source "$CONFIG_FILE"
    fi
    if [[ -n "$PI_IDENTITY_FILE" && -f "$PI_IDENTITY_FILE" ]]; then
        default_identity="$PI_IDENTITY_FILE"
    fi
    for identity in "$HOME/.ssh/fl_demo_pi_ed25519" "$HOME/.ssh/id_ed25519"; do
        if [[ -z "$default_identity" && -f "$identity" ]]; then
            default_identity="$identity"
            break
        fi
    done

    PI_SSH_TARGET="${PI_SSH_TARGET:-<pi-user>@raspberrypi.local}"
    PI_REPO="${PI_REPO:-/home/<pi-user>/cross-silo-fl-medical-image-classification}"
    PI_DATA_ROOT="${PI_DATA_ROOT:-/home/<pi-user>/fl_data/fed_isic2019/raw}"
    LAPTOP_DATA_ROOT="${LAPTOP_DATA_ROOT:-$HOME/fl_data/fed_isic2019/raw}"
    PI_BOOT_WAIT="${PI_BOOT_WAIT:-30}"

    printf '\nOne-time presentation configuration\n'
    printf 'Use the stable Pi name when available; its IPv4 address is resolved at every start.\n\n'
    read -r -p "Pi SSH target [$PI_SSH_TARGET]: " pi_target
    pi_target="${pi_target:-$PI_SSH_TARGET}"
    [[ -n "$pi_target" ]] || die 20 "The Pi SSH target cannot be empty."

    read -r -p "Pi repository path [$PI_REPO]: " pi_repo
    pi_repo="${pi_repo:-$PI_REPO}"
    read -r -p "Pi data path [$PI_DATA_ROOT]: " pi_data
    pi_data="${pi_data:-$PI_DATA_ROOT}"
    read -r -p "Laptop data path [$LAPTOP_DATA_ROOT]: " laptop_data
    laptop_data="${laptop_data:-$LAPTOP_DATA_ROOT}"
    read -r -p "SSH private key [${default_identity:-use SSH default}]: " identity
    identity="${identity:-$default_identity}"

    mkdir -p -- "$(dirname -- "$CONFIG_FILE")"
    {
        printf '# Local presentation settings. This file is intentionally ignored by Git.\n'
        printf 'PI_SSH_TARGET=%q\n' "$pi_target"
        printf 'PI_REPO=%q\n' "$pi_repo"
        printf 'PI_DATA_ROOT=%q\n' "$pi_data"
        printf 'LAPTOP_DATA_ROOT=%q\n' "$laptop_data"
        printf 'PI_IDENTITY_FILE=%q\n' "$identity"
        printf 'CONDA_ENV=%q\n' "flamby_isic"
        printf 'ROUNDS=%q\n' "3"
        printf 'MAX_BATCHES=%q\n' "4"
        printf 'PI_BOOT_WAIT=%q\n' "$PI_BOOT_WAIT"
    } >"$CONFIG_FILE"
    chmod 600 "$CONFIG_FILE" 2>/dev/null || true
    ok "Saved $CONFIG_FILE"

    read -r -p "Test the configuration now? [Y/n]: " answer
    if [[ "${answer:-Y}" =~ ^[Nn]$ ]]; then
        MODE="configured"
        return 0
    fi
    MODE="check"
}

if [[ "$MODE" == "configure" ]]; then
    configure
    [[ "$MODE" == "check" ]] || exit 0
elif [[ ! -f "$CONFIG_FILE" ]]; then
    if [[ -t 0 ]]; then
        warn "No saved presentation configuration was found."
        configure
    else
        die 21 "Missing $CONFIG_FILE. Run with --configure once."
    fi
fi

if [[ "$MODE" == "configured" ]]; then
    printf '\nConfiguration saved. Start the launcher again when you want to run the demo.\n'
    exit 0
fi

# shellcheck disable=SC1090
source "$CONFIG_FILE"

CONDA_ENV="${CONDA_ENV:-flamby_isic}"
PI_SSH_TARGET="${PI_SSH_TARGET:-}"
PI_REPO="${PI_REPO:-/home/<pi-user>/cross-silo-fl-medical-image-classification}"
PI_DATA_ROOT="${PI_DATA_ROOT:-/home/<pi-user>/fl_data/fed_isic2019/raw}"
PI_IDENTITY_FILE="${PI_IDENTITY_FILE:-}"
LAPTOP_DATA_ROOT="${LAPTOP_DATA_ROOT:-$HOME/fl_data/fed_isic2019/raw}"
CHECKPOINT="${CHECKPOINT:-$REPO/experiments/live/pretrained.pt}"
ROUNDS="${ROUNDS:-3}"
MAX_BATCHES="${MAX_BATCHES:-4}"
FLOWER_PORT="${FLOWER_PORT:-8080}"
DASHBOARD_PORT="${DASHBOARD_PORT:-8000}"
PI_TUNNEL_PORT="${PI_TUNNEL_PORT:-18080}"
ROUND_TIMEOUT="${ROUND_TIMEOUT:-120}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-45}"
DEMO_TIMEOUT="${DEMO_TIMEOUT:-240}"
REMOTE_CLIENT_TIMEOUT="${REMOTE_CLIENT_TIMEOUT:-300}"
PI_BOOT_WAIT="${PI_BOOT_WAIT:-30}"
EXPECTED_FLOWER="${EXPECTED_FLOWER:-1.11.1}"
PYTHON_BIN="${PYTHON_BIN:-python}"
LAPTOP_LABEL="${LAPTOP_LABEL:-laptop-GPU}"
PI_LABEL="${PI_LABEL:-pi5-CPU}"

[[ -n "$PI_SSH_TARGET" ]] || die 22 "PI_SSH_TARGET is empty. Run with --configure."
[[ "$ROUNDS" =~ ^[1-9][0-9]*$ ]] || die 22 "ROUNDS must be a positive integer."
[[ "$MAX_BATCHES" =~ ^[1-9][0-9]*$ ]] || die 22 "MAX_BATCHES must be a positive integer."
[[ "$REMOTE_CLIENT_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || \
    die 22 "REMOTE_CLIENT_TIMEOUT must be a positive integer."
[[ "$PI_BOOT_WAIT" =~ ^[1-9][0-9]*$ ]] || die 22 "PI_BOOT_WAIT must be a positive integer."
[[ "$DEMO_TIMEOUT" =~ ^[1-9][0-9]*$ ]] || die 22 "DEMO_TIMEOUT must be a positive integer."
[[ "$PI_REPO" == /* && "$PI_DATA_ROOT" == /* ]] || \
    die 22 "PI_REPO and PI_DATA_ROOT must be absolute Linux paths."

# A double-clicked .cmd starts a non-interactive shell, where conda may not be
# initialized. Re-enter the configured environment automatically.
if [[ "${CONDA_DEFAULT_ENV:-}" != "$CONDA_ENV" && "${FL_DEMO_CONDA_REEXEC:-0}" != "1" ]]; then
    conda_exe=""
    if command -v conda >/dev/null 2>&1; then
        conda_exe="$(command -v conda)"
    else
        for candidate in "$HOME/miniconda3/bin/conda" "$HOME/anaconda3/bin/conda" "/opt/conda/bin/conda"; do
            if [[ -x "$candidate" ]]; then conda_exe="$candidate"; break; fi
        done
    fi
    if [[ -n "$conda_exe" ]]; then
        reexec_args=(--config "$CONFIG_FILE")
        if [[ "$MODE" == "check" ]]; then
            reexec_args=(--check-only "${reexec_args[@]}")
        fi
        if ((NO_BROWSER)); then
            reexec_args=(--no-browser "${reexec_args[@]}")
        fi
        step "Entering conda environment '$CONDA_ENV'"
        exec env FL_DEMO_CONDA_REEXEC=1 "$conda_exe" run --no-capture-output \
            -n "$CONDA_ENV" bash "$SCRIPT_PATH" "${reexec_args[@]}"
    fi
    warn "Conda was not found automatically; checking the current Python environment."
fi

cd -- "$REPO"

port_is_free() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket()
try:
    s.bind((host, port))
finally:
    s.close()
PY
}

port_is_open() {
    "$PYTHON_BIN" - "$1" "$2" <<'PY' >/dev/null 2>&1
import socket, sys
try:
    with socket.create_connection((sys.argv[1], int(sys.argv[2])), timeout=0.4):
        pass
except OSError:
    raise SystemExit(1)
PY
}

show_port_owner() {
    local port="$1"
    command -v ss >/dev/null 2>&1 && ss -ltnp "sport = :$port" 2>/dev/null || true
}

PI_SSH_TARGET_CONFIGURED="$PI_SSH_TARGET"
PI_SSH_TARGET_EFFECTIVE="$PI_SSH_TARGET"
PI_SSH_HOST="$PI_SSH_TARGET"
SSH_TARGET_OPTS=()

resolve_pi_ssh_target() {
    local target="$PI_SSH_TARGET_CONFIGURED" user_prefix="" user="" host resolved="" deadline

    if [[ "$target" == *@* ]]; then
        user="${target%@*}"
        [[ "$user" =~ ^[A-Za-z0-9._-]+$ ]] || \
            die 26 "Invalid Pi SSH user in target: $target"
        user_prefix="${user}@"
        host="${target##*@}"
    else
        host="$target"
    fi
    [[ -n "$host" && "$host" =~ ^[A-Za-z0-9._-]+$ ]] || \
        die 26 "Invalid Pi SSH target: $target (expected user@hostname or user@IPv4)."
    PI_SSH_HOST="$host"

    # A literal IPv4 address needs no name lookup. Hostnames are resolved here
    # because WSL's `getent ahostsv4` can resolve mDNS names even when OpenSSH's
    # own getaddrinfo lookup cannot.
    if is_valid_ipv4 "$host"; then
        return 0
    fi

    deadline=$((SECONDS + PI_BOOT_WAIT))
    while ((SECONDS < deadline)); do
        if command -v getent >/dev/null 2>&1; then
            resolved="$(timeout 4 getent ahostsv4 "$host" 2>/dev/null | awk 'NR == 1 { print $1; exit }' || true)"
        fi
        if [[ -z "$resolved" ]] && command -v powershell.exe >/dev/null 2>&1; then
            resolved="$(timeout 6 powershell.exe -NoProfile -NonInteractive -Command \
                "[System.Net.Dns]::GetHostAddresses('$host') | Where-Object { \$_.AddressFamily -eq [System.Net.Sockets.AddressFamily]::InterNetwork } | ForEach-Object { \$_.IPAddressToString } | Select-Object -First 1" \
                2>/dev/null | tr -d '\r' | head -n 1 || true)"
        fi
        if [[ -n "$resolved" ]] && is_valid_ipv4 "$resolved"; then break; fi
        resolved=""
        sleep 2
    done

    if [[ -z "$resolved" ]]; then
        cat >&2 <<EOF

Could not find the Raspberry Pi's IPv4 address for: $host
Check that the Pi and laptop are on the same router, then try in WSL:
  getent ahostsv4 $host
Or in Windows Command Prompt:
  ping -4 $host

As a last fallback, run --configure and enter <pi-user>@<CURRENT_PI_IP>.
EOF
        die 26 "Raspberry Pi name resolution failed after ${PI_BOOT_WAIT}s."
    fi

    PI_SSH_TARGET_EFFECTIVE="${user_prefix}${resolved}"
    # Store/check the Pi key under its stable hostname, not its changing DHCP
    # address. accept-new enrolls this alias once but still rejects a changed key.
    SSH_TARGET_OPTS+=( -o "HostKeyAlias=$host" -o CheckHostIP=no )
    ok "Raspberry Pi address: $host -> $resolved"
}

is_valid_ipv4() {
    local ip="$1" part
    local -a octets
    [[ "$ip" =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]] || return 1
    IFS=. read -r -a octets <<<"$ip"
    ((${#octets[@]} == 4)) || return 1
    for part in "${octets[@]}"; do
        ((10#$part <= 255)) || return 1
    done
}

SSH_BIN="${SSH_BIN:-}"
if [[ -z "$SSH_BIN" ]]; then
    if [[ -x /usr/bin/ssh ]]; then
        SSH_BIN=/usr/bin/ssh
    else
        SSH_BIN="$(command -v ssh || true)"
    fi
fi

SSH_OPTS=(
    -4
    -T
    -o BatchMode=yes
    -o StrictHostKeyChecking=accept-new
    -o ConnectTimeout=6
    -o ConnectionAttempts=1
    -o ServerAliveInterval=5
    -o ServerAliveCountMax=3
    -o Compression=no
)
if [[ -n "$PI_IDENTITY_FILE" ]]; then
    [[ -f "$PI_IDENTITY_FILE" ]] || die 23 "SSH key not found: $PI_IDENTITY_FILE"
    SSH_OPTS+=( -i "$PI_IDENTITY_FILE" -o IdentitiesOnly=yes )
fi

step "Fast local preflight"
for required in \
    scripts/live/server.py scripts/live/client.py reports/live_dashboard.html \
    configs/live_fedavg.yaml; do
    [[ -f "$required" ]] || die 24 "Required project file is missing: $required"
done
command -v "$PYTHON_BIN" >/dev/null 2>&1 || die 24 "Python is not available."
[[ -x "$SSH_BIN" ]] || die 24 "The Linux ssh command is not installed."
command -v git >/dev/null 2>&1 || die 24 "Git is not installed."
command -v timeout >/dev/null 2>&1 || die 24 "The timeout command is not installed."
resolve_pi_ssh_target
[[ -s "$CHECKPOINT" ]] || die 24 "Warm-start checkpoint is missing: $CHECKPOINT"
checkpoint_bytes="$(wc -c <"$CHECKPOINT")"
((checkpoint_bytes > 1000000)) || die 24 "Checkpoint appears incomplete: $CHECKPOINT"

for split in train test; do
    for cid in 0 1 2 3 4 5; do
        [[ -d "$LAPTOP_DATA_ROOT/$split/client_$cid" ]] || \
            die 24 "Laptop data missing: $LAPTOP_DATA_ROOT/$split/client_$cid"
    done
done

"$PYTHON_BIN" - "$EXPECTED_FLOWER" <<'PY'
import sys
import flwr, numpy, torch
expected = sys.argv[1]
if flwr.__version__ != expected:
    raise SystemExit(f"Flower {flwr.__version__} found; presentation mode requires {expected}")
if int(numpy.__version__.split('.')[0]) >= 2:
    raise SystemExit(f"NumPy {numpy.__version__} found; presentation mode requires NumPy < 2")
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable; refusing to silently run the laptop client on CPU")
print(f"Python ready: Flower {flwr.__version__}, NumPy {numpy.__version__}, "
      f"GPU {torch.cuda.get_device_name(0)}")
PY

for port in "$DASHBOARD_PORT" "$FLOWER_PORT"; do
    if ! port_is_free 127.0.0.1 "$port"; then
        show_port_owner "$port"
        die 25 "Port $port is already in use. Close the old demo process and try again."
    fi
done
ok "Checkpoint, all laptop shards, CUDA, dependencies, and ports are ready."

step "Raspberry Pi preflight"
pi_ssh_error=""
pi_ssh_ok=0
pi_ssh_deadline=$((SECONDS + PI_BOOT_WAIT))
while :; do
    set +e
    pi_ssh_error="$("$SSH_BIN" "${SSH_OPTS[@]}" "${SSH_TARGET_OPTS[@]}" \
        "$PI_SSH_TARGET_EFFECTIVE" true 2>&1)"
    pi_ssh_rc=$?
    set -e
    if ((pi_ssh_rc == 0)); then
        pi_ssh_ok=1
        break
    fi
    if [[ "$pi_ssh_error" == *"REMOTE HOST IDENTIFICATION HAS CHANGED"* ||
          "$pi_ssh_error" == *"Host key verification failed"* ||
          "$pi_ssh_error" == *"Permission denied"* ||
          "$pi_ssh_error" == *"no such identity"* ||
          "$pi_ssh_error" == *"Bad owner or permissions"* ]]; then
        break
    fi
    ((SECONDS < pi_ssh_deadline)) || break
    sleep 2
done
if ((pi_ssh_ok == 0)); then
    printf '%s\n' "$pi_ssh_error" >&2
    cat >&2 <<EOF

The Pi did not accept non-interactive SSH. Before presentation day, run once:
  ssh -4 -o HostKeyAlias=$PI_SSH_HOST $PI_SSH_TARGET_EFFECTIVE
  ssh-keygen -t ed25519 -f ~/.ssh/fl_demo_pi_ed25519
  ssh-copy-id -i ~/.ssh/fl_demo_pi_ed25519.pub $PI_SSH_TARGET_EFFECTIVE
Then run --configure again and select ~/.ssh/fl_demo_pi_ed25519.

The launcher resolves .local names itself; use the current IPv4 only as a fallback.
EOF
    die 26 "Raspberry Pi SSH preflight failed."
fi

if ! git diff --quiet -- \
    scripts/live/client.py src/fl_med configs/live_fedavg.yaml configs/_base.yaml; then
    die 27 "Laptop live-training code has uncommitted changes; use the reviewed repository revision."
fi
local_commit="$(git rev-parse HEAD)"
printf -v remote_preflight_cmd 'bash -s -- %q %q %q %q %q' \
    "$PI_REPO" "$PI_DATA_ROOT" "$EXPECTED_FLOWER" "$PI_TUNNEL_PORT" "$local_commit"
if ! "$SSH_BIN" "${SSH_OPTS[@]}" "${SSH_TARGET_OPTS[@]}" \
    "$PI_SSH_TARGET_EFFECTIVE" "$remote_preflight_cmd" <<'REMOTE'
set -Eeuo pipefail
repo="$1"; data_root="$2"; expected_flower="$3"; tunnel_port="$4"; expected_commit="$5"
py="$repo/.venv/bin/python"
[[ -d "$repo" ]] || { echo "Pi repository missing: $repo" >&2; exit 31; }
[[ -x "$py" ]] || { echo "Pi virtualenv Python missing: $py" >&2; exit 32; }
[[ -f "$repo/scripts/live/client.py" ]] || { echo "Pi live client missing" >&2; exit 33; }
command -v timeout >/dev/null 2>&1 || { echo "Pi 'timeout' command missing" >&2; exit 33; }
for path in "$data_root/train/client_5" "$data_root/test/client_5"; do
    [[ -d "$path" ]] || { echo "Pi data missing: $path" >&2; exit 34; }
done
actual_commit="$(git -C "$repo" rev-parse HEAD)"
[[ "$actual_commit" == "$expected_commit" ]] || {
    echo "Laptop and Pi repository revisions differ. Update the Pi repository first." >&2
    exit 35
}
git -C "$repo" diff --quiet -- \
    scripts/live/client.py src/fl_med configs/live_fedavg.yaml configs/_base.yaml || {
    echo "Pi live-training code has uncommitted changes." >&2
    exit 36
}
"$py" - "$expected_flower" <<'PY'
import sys
import flwr, numpy, torch
expected = sys.argv[1]
if flwr.__version__ != expected:
    raise SystemExit(f"Pi Flower {flwr.__version__} found; expected {expected}")
if int(numpy.__version__.split('.')[0]) >= 2:
    raise SystemExit(f"Pi NumPy {numpy.__version__} found; expected NumPy < 2")
print(f"Pi ready: Flower {flwr.__version__}, NumPy {numpy.__version__}, PyTorch {torch.__version__}")
PY
"$py" - "$tunnel_port" <<'PY'
import socket, sys
s = socket.socket()
try:
    s.bind(("127.0.0.1", int(sys.argv[1])))
finally:
    s.close()
PY
REMOTE
then
    die 27 "Raspberry Pi software/data preflight failed. See the message above."
fi

set +e
tunnel_probe_output="$(timeout 4 "$SSH_BIN" "${SSH_OPTS[@]}" \
    "${SSH_TARGET_OPTS[@]}" \
    -N -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:$PI_TUNNEL_PORT:127.0.0.1:$FLOWER_PORT" \
    "$PI_SSH_TARGET_EFFECTIVE" 2>&1)"
tunnel_probe_rc=$?
set -e
if ((tunnel_probe_rc != 124)); then
    [[ -n "$tunnel_probe_output" ]] && printf '%s\n' "$tunnel_probe_output" >&2
    die 28 "The Pi SSH server rejected the reverse tunnel (exit $tunnel_probe_rc)."
fi
sleep 1
ok "The physical Pi, its client-5 data, environment, code, and tunnel port are ready."

if [[ "$MODE" == "check" ]]; then
    printf '\n%sALL CHECKS PASSED%s — presentation startup is ready.\n' "$C_GREEN" "$C_RESET"
    exit 0
fi

RUN_ID="$(date -u +%Y%m%dT%H%M%SZ)-$$"
RUN_DIR="$REPO/experiments/live/presentation_runs/$RUN_ID"
LIVE_DIR="$REPO/experiments/live"
mkdir -p -- "$RUN_DIR" "$LIVE_DIR"

if [[ -f "$LIVE_DIR/history.json" ]]; then
    mv -- "$LIVE_DIR/history.json" "$RUN_DIR/history.before.json"
fi
if [[ -f "$LIVE_DIR/live_status.json" ]]; then
    mv -- "$LIVE_DIR/live_status.json" "$RUN_DIR/live_status.before.json"
fi

"$PYTHON_BIN" - "$LIVE_DIR/live_status.json" "$ROUNDS" <<'PY'
import json, pathlib, sys
pathlib.Path(sys.argv[1]).write_text(json.dumps({
    "status": "preparing", "round": 0, "total_rounds": int(sys.argv[2]),
    "history": [], "clients": []
}))
PY

PIDS=()
add_process() { PIDS+=("$2"); }
DEMO_DEADLINE_AT=$((SECONDS + DEMO_TIMEOUT))

cleanup() {
    local i pid
    trap - EXIT INT TERM
    for ((i=${#PIDS[@]}-1; i>=0; i--)); do
        pid="${PIDS[$i]}"
        if kill -0 "$pid" 2>/dev/null; then kill "$pid" 2>/dev/null || true; fi
    done
    for pid in "${PIDS[@]}"; do wait "$pid" 2>/dev/null || true; done
}
trap cleanup EXIT
trap 'warn "Stopping the presentation demo..."; exit 130' INT TERM

wait_for_process_port() {
    local pid="$1" host="$2" port="$3" timeout="$4" log_file="$5" label="$6" elapsed=0
    while ((elapsed < timeout)); do
        if ((SECONDS >= DEMO_DEADLINE_AT)); then
            die 41 "The complete demo exceeded its ${DEMO_TIMEOUT}s safety deadline."
        fi
        if ! kill -0 "$pid" 2>/dev/null; then
            tail -n 30 "$log_file" >&2 || true
            die 40 "$label exited during startup."
        fi
        if port_is_open "$host" "$port"; then return 0; fi
        sleep 1; ((elapsed+=1))
    done
    tail -n 30 "$log_file" >&2 || true
    die 41 "$label did not become ready within ${timeout}s."
}

step "Starting the live dashboard"
"$PYTHON_BIN" -m http.server "$DASHBOARD_PORT" --bind 127.0.0.1 \
    >"$RUN_DIR/dashboard.log" 2>&1 &
dashboard_pid=$!; add_process dashboard "$dashboard_pid"
wait_for_process_port "$dashboard_pid" 127.0.0.1 "$DASHBOARD_PORT" 10 \
    "$RUN_DIR/dashboard.log" "Dashboard server"
"$PYTHON_BIN" - "$DASHBOARD_PORT" <<'PY'
import sys, urllib.request
url = f"http://127.0.0.1:{int(sys.argv[1])}/reports/live_dashboard.html"
with urllib.request.urlopen(url, timeout=3) as response:
    body = response.read(4096)
    if response.status != 200 or b"Federated Learning" not in body:
        raise SystemExit(f"Dashboard validation failed: HTTP {response.status}")
PY
ok "Dashboard is serving on http://localhost:$DASHBOARD_PORT/reports/live_dashboard.html"

step "Starting the Flower coordinator"
env DATA_ROOT="$LAPTOP_DATA_ROOT" "$PYTHON_BIN" scripts/live/server.py \
    --rounds "$ROUNDS" \
    --min-clients 2 \
    --host "127.0.0.1:$FLOWER_PORT" \
    --device cuda \
    --init-model "$CHECKPOINT" \
    --round-timeout "$ROUND_TIMEOUT" \
    --num-workers 0 \
    --out "$LIVE_DIR" \
    >"$RUN_DIR/server.log" 2>&1 &
server_pid=$!; add_process server "$server_pid"
wait_for_process_port "$server_pid" 127.0.0.1 "$FLOWER_PORT" "$STARTUP_TIMEOUT" \
    "$RUN_DIR/server.log" "Flower coordinator"
ok "Coordinator is ready."

step "Starting the laptop GPU client"
env DATA_ROOT="$LAPTOP_DATA_ROOT" "$PYTHON_BIN" scripts/live/client.py \
    --server "127.0.0.1:$FLOWER_PORT" \
    --client-id 0 \
    --label "$LAPTOP_LABEL" \
    --device cuda \
    --max-batches "$MAX_BATCHES" \
    --num-workers 0 \
    >"$RUN_DIR/laptop-client.log" 2>&1 &
laptop_pid=$!; add_process laptop-client "$laptop_pid"

step "Starting the Raspberry Pi client through the encrypted SSH tunnel"
printf -v remote_client_cmd \
    'cd %q && export DATA_ROOT=%q && exec timeout --foreground --signal=INT --kill-after=5s %qs %q scripts/live/client.py --server %q --client-id 5 --label %q --device cpu --max-batches %q --freeze-backbone --num-workers 0' \
    "$PI_REPO" "$PI_DATA_ROOT" "$REMOTE_CLIENT_TIMEOUT" "$PI_REPO/.venv/bin/python" \
    "127.0.0.1:$PI_TUNNEL_PORT" "$PI_LABEL" "$MAX_BATCHES"
"$SSH_BIN" "${SSH_OPTS[@]}" "${SSH_TARGET_OPTS[@]}" \
    -o ExitOnForwardFailure=yes \
    -R "127.0.0.1:$PI_TUNNEL_PORT:127.0.0.1:$FLOWER_PORT" \
    "$PI_SSH_TARGET_EFFECTIVE" "$remote_client_cmd" \
    >"$RUN_DIR/pi-client.log" 2>&1 &
pi_pid=$!; add_process pi-client "$pi_pid"

sleep 2
if ! kill -0 "$laptop_pid" 2>/dev/null; then
    tail -n 30 "$RUN_DIR/laptop-client.log" >&2 || true
    die 42 "Laptop client failed during startup."
fi
if ! kill -0 "$pi_pid" 2>/dev/null; then
    tail -n 30 "$RUN_DIR/pi-client.log" >&2 || true
    die 43 "Pi client or SSH reverse tunnel failed during startup."
fi
ok "Both physical clients are running."

DASHBOARD_URL="http://localhost:$DASHBOARD_PORT/reports/live_dashboard.html"
if ((NO_BROWSER == 0)); then
    if command -v powershell.exe >/dev/null 2>&1; then
        powershell.exe -NoProfile -Command "Start-Process '$DASHBOARD_URL'" >/dev/null 2>&1 || true
    elif command -v cmd.exe >/dev/null 2>&1; then
        cmd.exe /c start "" "$DASHBOARD_URL" >/dev/null 2>&1 || true
    elif command -v xdg-open >/dev/null 2>&1; then
        xdg-open "$DASHBOARD_URL" >/dev/null 2>&1 || true
    fi
fi

printf '\n%sLIVE DEMO RUNNING%s\n' "$C_GREEN" "$C_RESET"
printf 'Dashboard: %s\n' "$DASHBOARD_URL"
printf 'Logs:      %s\n' "$RUN_DIR"
printf 'Expected:  %s real rounds with laptop-GPU + Raspberry-Pi CPU\n\n' "$ROUNDS"

last_state=""
current_round=0
while kill -0 "$server_pid" 2>/dev/null; do
    if ((SECONDS >= DEMO_DEADLINE_AT)); then
        die 47 "The complete demo exceeded its ${DEMO_TIMEOUT}s safety deadline."
    fi
    state="$({ "$PYTHON_BIN" - "$LIVE_DIR/live_status.json" <<'PY'
import json, sys
try:
    d = json.load(open(sys.argv[1], encoding="utf-8"))
    h = d.get("history") or []
    acc = ""
    if h:
        acc = f", balanced accuracy {float(h[-1]['bal_acc']):.3f}"
    print(f"{d.get('status', 'starting')}|{int(d.get('round', 0))}|{acc}")
except Exception:
    print("starting|0|")
PY
    } 2>/dev/null)"
    status_name="${state%%|*}"
    rest="${state#*|}"; current_round="${rest%%|*}"; suffix="${rest#*|}"
    if [[ "$state" != "$last_state" ]]; then
        printf '[demo] %s — round %s/%s%s\n' "$status_name" "$current_round" "$ROUNDS" "$suffix"
        last_state="$state"
    fi
    if ! kill -0 "$laptop_pid" 2>/dev/null && ((current_round < ROUNDS)); then
        tail -n 30 "$RUN_DIR/laptop-client.log" >&2 || true
        die 44 "Laptop client disconnected before the final round."
    fi
    if ! kill -0 "$pi_pid" 2>/dev/null && ((current_round < ROUNDS)); then
        tail -n 30 "$RUN_DIR/pi-client.log" >&2 || true
        die 45 "Pi client disconnected before the final round."
    fi
    sleep 1
done

set +e
wait "$server_pid"; server_rc=$?
set -e
if ((server_rc != 0)); then
    tail -n 40 "$RUN_DIR/server.log" >&2 || true
    die 46 "Flower coordinator failed with exit code $server_rc."
fi

for _ in $(seq 1 10); do
    if ! kill -0 "$laptop_pid" 2>/dev/null && ! kill -0 "$pi_pid" 2>/dev/null; then break; fi
    sleep 1
done

step "Validating the completed physical run"
"$PYTHON_BIN" - "$LIVE_DIR/history.json" "$ROUNDS" "$LAPTOP_LABEL" "$PI_LABEL" <<'PY'
import json, sys
path, rounds, laptop, pi = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
with open(path, encoding="utf-8") as f:
    data = json.load(f)
history = data.get("history") or []
timings = data.get("timings") or []
if not history or int(history[-1].get("round", -1)) != rounds:
    raise SystemExit(f"Expected a final evaluation for round {rounds}")
for rnd in range(1, rounds + 1):
    tags = {str(t.get("tag")) for t in timings if int(t.get("round", -1)) == rnd}
    missing = {laptop, pi} - tags
    if missing:
        raise SystemExit(f"Round {rnd} is missing physical client(s): {sorted(missing)}")
if len(timings) != rounds * 2:
    raise SystemExit(f"Expected {rounds * 2} client timing records, found {len(timings)}")
latest = history[-1]
print(f"Validated {rounds} rounds and both clients; final balanced accuracy="
      f"{float(latest['bal_acc']):.3f}")
PY

cp -- "$LIVE_DIR/history.json" "$RUN_DIR/history.json"
cp -- "$LIVE_DIR/live_status.json" "$RUN_DIR/live_status.json"
ok "The live two-machine federated run completed and was validated."

printf '\nThe dashboard will remain available for your explanation.\n'
if [[ -t 0 ]]; then
    read -r -p "Press Enter when you are ready to close the demo..." _
fi
