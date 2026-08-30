# Raspberry Pi 5 as a real federated client

This turns the project from a *simulated* federation (all clients in one Python
loop) into a **genuinely distributed** one: the Raspberry Pi acts as an "edge
hospital" that trains on its own local data and sends **only model updates** over
the network to the coordinating server on your laptop. Raw images never leave the
Pi — the privacy claim made physical.

We use [Flower](https://flower.ai) (`flwr`) for the transport. The model, data
loaders, training loop and metrics are the *same* code as the simulation
(`src/fl_med/federated_live/` just wraps them), so the run is directly comparable.

> **Design for speed.** The Pi trains the **smallest silo** (client 5, ~281
> images) for **1 local epoch**, capped with `--max-batches`, for a **short** run
> (~8 rounds). The rigorous 3-seed numbers stay in the simulation; this run is the
> *proof it works on real distributed hardware* — plus a **straggler** measurement.

## Presentation day: one command (recommended)

The presentation launcher replaces the manual server/dashboard/client terminals.
It checks the checkpoint, every laptop data shard, CUDA, Flower/NumPy versions,
ports, passwordless Pi SSH, the Pi virtual environment, client-5 data, matching
client code, and a free tunnel port **before** starting anything. A failed check
therefore stops in seconds instead of failing halfway through the demonstration.

It uses an SSH reverse tunnel:

```text
Pi client -> Pi 127.0.0.1:18080 -> encrypted SSH tunnel
          -> laptop WSL 127.0.0.1:8080 -> Flower coordinator
```

This avoids Windows 10 `portproxy`, Administrator access, firewall changes, and
changing laptop/WSL IP addresses. The Pi still performs genuine local training
on a second physical machine; SSH only carries the Flower connection.

**Configure and test once before the presentation:**

```bash
bash scripts/live/presentation_demo.sh --configure
bash scripts/live/presentation_demo.sh --check-only
```

Use the stable Pi name in the SSH target (normally `<pi-user>@raspberrypi.local`). At
every start, the launcher resolves that name through WSL's explicit IPv4 lookup
and gives SSH the resulting numeric address. A router/hotspot DHCP change after a
restart therefore needs no reconfiguration. The Pi host key remains pinned to
the stable name after the first successful preflight enrolls it, so an unexpected
changed device is then rejected. Run the check once on a trusted network before
presentation day. Use a literal target such as `<pi-user>@192.168.1.50` only if
`.local` lookup is unavailable.

The defaults use three rounds, four local batches, and a frozen Pi backbone. The
real run normally takes about 2–3 minutes; the startup checks take seconds. It
also waits briefly if the Pi is still finishing its boot.

If the SSH check reports that a password or unknown host key is required, set it
up once from WSL:

```bash
PI_IP="$(getent ahostsv4 raspberrypi.local | awk '$2 == "STREAM" {print $1; exit}')"
ssh -4 -o HostKeyAlias=raspberrypi.local -o CheckHostIP=no "<pi-user>@$PI_IP"
ssh-keygen -t ed25519 -f ~/.ssh/fl_demo_pi_ed25519
ssh-copy-id -i ~/.ssh/fl_demo_pi_ed25519.pub \
    -o HostKeyAlias=raspberrypi.local -o CheckHostIP=no "<pi-user>@$PI_IP"
```

Update the Pi repository to the same revision as the laptop before running the
check. The launcher deliberately refuses to mix different `client.py` versions.

**On presentation day:** double-click `START_PRESENTATION_DEMO.cmd` in the
repository root. It enters the `flamby_isic` conda environment, runs preflight,
starts the dashboard/coordinator/laptop client/Pi client, opens the browser,
validates that both physical clients completed every configured round, and
archives logs in `experiments/live/presentation_runs/`.

If name lookup ever fails, verify `getent ahostsv4 raspberrypi.local` in WSL or
`ping -4 raspberrypi.local` in Windows, then run `--configure` with the displayed
IPv4 as the last-resort fallback. Changes to the laptop LAN or WSL address do not
matter because the Flower connection uses the outbound SSH reverse tunnel.

The launcher also disables unnecessary ImageNet-weight downloads: the server
loads `pretrained.pt`, and each client immediately receives the global Flower
parameters. The live run therefore has no internet dependency.

---

## 0. Prerequisites

- Raspberry Pi 5 with **64-bit Raspberry Pi OS** (Bookworm). 64-bit is required
  for the PyTorch aarch64 wheels.
- Pi and laptop on the **same LAN** (same Wi-Fi/router or Ethernet).
- The repo cloned on **both** machines.

---

## 1. Install the software on the Pi

```bash
# on the Pi
sudo apt update && sudo apt install -y python3-venv libopenblas0 git
git clone https://github.com/MahmoudAsadi97/cross-silo-fl-medical-image-classification.git
cd cross-silo-fl-medical-image-classification

python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip

# CPU PyTorch for aarch64 (no CUDA on the Pi) + the pinned Flower version
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install 'flwr==1.11.1' numpy pyyaml pillow

pip install -e . --no-deps        # register the fl_med package
python -c "import torch, flwr, fl_med; print('pi ready:', torch.__version__, flwr.__version__)"
```

## 2. Copy the Pi's local data shard

The Pi only needs **client 5's** images (small). From the laptop:

```bash
# from the laptop (adjust paths); creates the same layout on the Pi
PI_IP="$(getent ahostsv4 raspberrypi.local | awk '$2 == "STREAM" {print $1; exit}')"
PI="<pi-user>@$PI_IP"
rsync -av data/fed_isic2019/raw/train/client_5  $PI:~/fl_data/fed_isic2019/raw/train/
rsync -av data/fed_isic2019/raw/test/client_5   $PI:~/fl_data/fed_isic2019/raw/test/
```

On the Pi, `DATA_ROOT` is then `~/fl_data/fed_isic2019/raw`.

---

## 3. Networking: let the Pi reach the server in WSL

Your server runs in **WSL2**, which by default has its own NAT'd network the Pi
can't reach. Pick **one** fix:

**Option A — mirrored networking (Windows 11, easiest).** Edit
`C:\Users\<you>\.wslconfig`:

```ini
[wsl2]
networkingMode=mirrored
```

Then in PowerShell: `wsl --shutdown`, reopen WSL. WSL now shares the laptop's LAN
IP, so the Pi connects straight to it.

**Option B — port forward (fallback).** In WSL: `hostname -I` (note the WSL IP).
Then in an **admin** PowerShell:

```powershell
netsh interface portproxy add v4tov4 listenport=8080 listenaddress=0.0.0.0 connectport=8080 connectaddress=<WSL_IP>
netsh advfirewall firewall add rule name="flower8080" dir=in action=allow protocol=TCP localport=8080
```

**Find the laptop's LAN IP** (the address the Pi dials): `ipconfig` on Windows →
the IPv4 of your Wi-Fi/Ethernet adapter, e.g. `192.168.1.50`.

---

## 4. Run it

**First, on the laptop, pre-train a global model once** (single process — no GPU
contention) so the live run shows real accuracy from round 0:

```bash
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
    python scripts/live/pretrain_and_save.py --rounds 15 --device cuda
# -> experiments/live/pretrained.pt  (~0.20 balanced accuracy)
```

**Laptop (server + one local client for comparison)** — two terminals in WSL:

```bash
# terminal 1 — the coordinator, warm-started, waits for 2 clients (laptop + Pi)
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
    python scripts/live/server.py --rounds 8 --min-clients 2 --host 0.0.0.0:8080 \
        --device cuda --init-model experiments/live/pretrained.pt

# terminal 2 — a laptop hospital (silo 0), fast (GPU): the straggler baseline
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
    python scripts/live/client.py --server 127.0.0.1:8080 --client-id 0 --label laptop-c0 --device cuda
```

**Raspberry Pi (the edge hospital, silo 5):**

```bash
source .venv/bin/activate
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw \
    python scripts/live/client.py --server 192.168.1.50:8080 \
        --client-id 5 --label pi5 --device cpu --max-batches 8
```

(Replace `192.168.1.50` with your laptop's LAN IP.) When all 8 rounds finish the
server writes `experiments/live/history.json`. Make the figures on the laptop:

```bash
python scripts/live/plot_live.py
# -> reports/figures/live_accuracy.png  (accuracy vs round)
# -> reports/figures/live_straggler.png (per-client time/round: Pi vs laptop)
```

> **Verify locally first (no Pi needed):**
> `DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/live/run_local_demo.sh`
> runs the server + 2 clients on the laptop to confirm the whole path works.

---

## 4b. Faster Pi rounds — freeze-backbone (partial-model FL)

Quantization/pruning speed up *inference*, not training (training needs float
gradients, and structured pruning would change the architecture and break
FedAvg averaging). The correct **training** speedup for a weak device is
**freeze-backbone**: the Pi updates only the small classifier head, so the
expensive backward pass through the deep backbone disappears — while the
architecture (and therefore FedAvg aggregation) is unchanged. The Pi simply
"abstains" on the backbone and "votes" on the head.

**First, verify + measure on the Pi** (self-validating — prints PASS/FAIL for
correctness, then the measured speedup):

```bash
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/live/bench_pi.py \
    --client-id 5 --max-batches 8 --repeats 3
```

**MEASURED RESULT (Raspberry Pi 5, 2026-08-29, 3+3 rounds, warm-started model):**
all 4 correctness checks PASS (backbone bit-identical; head learns; state-dict
FedAvg-aggregatable). Full model **8.28 s/round** → frozen backbone **0.92 s/round**
= **8.99× speedup** (head = 4,104 of 11.18 M params, 0.04%). This shrinks the
measured straggler gap (Pi vs laptop GPU) from **~17× to <2×**. Laptop control:
2.48 s → 0.57 s (4.35×), same 4× PASS. Raw data: `experiments/live/bench_pi.json`.

**Then run the live client with the speedup:**

```bash
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/live/client.py \
    --server <LAN_IP>:8080 --client-id 5 --label pi5 --device cpu \
    --max-batches 8 --freeze-backbone
```

For the snappiest possible demo also lower the per-round work: `--max-batches 4`.

**Bonus (deployment story): quantized INFERENCE benchmark** — where int8 *does*
belong. Measures ms/image, model size, and accuracy for fp32 / TorchScript /
dynamic-int8 / static-int8 on the Pi:

```bash
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/edge_infer_bench.py --client-id 5
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` / client hangs | Wrong laptop IP, firewall, or WSL networking not mirrored/forwarded (§3). Confirm `ping <laptop-ip>` from the Pi. |
| Flower API errors | Server and Pi must use the **same** `flwr` version (`pip install 'flwr==1.11.1'` on both). |
| Shape/parameter mismatch | Server and all clients must use the **same** `--config`/`--tier`/`--image-size` so the model architecture matches. |
| Pi round is slow | Lower `--max-batches` (e.g. 4), keep `--image-size 64`. The slowness is expected — it's the finding (straggler). |
| `torch` install fails on Pi | Ensure **64-bit** OS; install inside a venv; `sudo apt install libopenblas0`; use the CPU `--index-url` above. |
| Pi runs out of memory | Smaller `--batch-size` (e.g. 8) and `--image-size 64`. |

## 6. What to report

- **It's real FL:** two machines, only weight tensors on the wire, raw images
  stayed on each device.
- **Straggler finding:** the Pi's fit time per round vs the laptop's
  (`live_straggler.png`) — FedAvg is synchronous, so the round is gated by the
  slowest device. This motivates client sampling / asynchronous FL and is a
  genuine cross-device systems result, not a limitation to hide.
- **Edge feasibility:** a $80 device meaningfully participates in training a
  clinical model without its data ever leaving — the core FL value proposition,
  demonstrated on hardware.
