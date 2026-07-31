# Installation Manual

Cross-Silo Federated Learning for Medical Image Classification (`fl_med`).

This guide installs the project and reproduces the experiments. Two paths: a
**minimal** install (pure-python core: config, metrics, the correctness checks,
heterogeneity analysis — no GPU needed) and a **full** install (GPU training, DP,
attacks). The project was developed and validated on **WSL2 Ubuntu** with an NVIDIA
GPU; the same steps work on native Linux.

## 1. Requirements

- Python 3.10+
- Git
- For training: an NVIDIA GPU + recent driver, and CUDA-enabled PyTorch. ~8 GB VRAM
  is enough for the `dev` tier and (with the batch-memory manager) for DP.
- Disk: ~2 GB for the environment; the real dataset is several GB more.

## 2. Get the code

```bash
git clone https://github.com/MahmoudAsadi97/cross-silo-fl-medical-image-classification.git
cd cross-silo-fl-medical-image-classification
```

## 3a. Minimal install (no GPU) — verify correctness immediately

```bash
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
python -m pip install -U pip
pip install -e .                                       # core deps only (numpy, sklearn, pyyaml, matplotlib...)
python scripts/verify_core_math.py                     # expect: 15/15 checks passed
```

This proves FedAvg aggregation, the SCAFFOLD equations, the metrics, the DP
accountant, and secure aggregation are correct — with no torch and no data.

## 3b. Full install (GPU) — the environment of record

Training was done in a **conda** environment (`flamby_isic`) that already contained a
CUDA build of PyTorch, torchvision, **Opacus** and **FLamby**. To avoid disturbing such
a curated environment, install the package WITHOUT touching its dependencies:

```bash
conda activate flamby_isic          # your env with torch+cuda, opacus, sklearn, etc.
pip install -e . --no-deps          # registers `fl_med` only; nothing else changes
pip install pytest                  # test runner (if not present)
python -c "import torch; print('CUDA:', torch.cuda.is_available())"   # expect: CUDA: True
```

Fresh machine instead of a pre-built env? Install the extras explicitly:

```bash
pip install -e ".[torch,privacy,dev]"     # torch/torchvision + opacus + pytest/ruff
```

## 4. Get the dataset (Fed-ISIC2019)

The real data is **git-ignored**; only a tiny synthetic fixture is committed (so tests
and the `smoke` tier run without it). Expected on-disk layout:

```
data/fed_isic2019/raw/{train,test}/client_<0..5>/class_<0..7>/*.jpg
```

Obtain it via FLamby (authoritative) or the Hugging Face mirror `flwrlabs/fed-isic2019`;
full steps are in `data/README.md`. Keep the official split so numbers stay comparable.

**Performance tip (important on WSL):** reading images from a Windows drive (`/mnt/c`)
is slow. Copy the dataset onto the Linux-native filesystem once and point runs at it:

```bash
mkdir -p ~/fl_data && cp -r data/fed_isic2019 ~/fl_data/
export DATA_ROOT=$HOME/fl_data/fed_isic2019/raw     # scripts read this; ~7x faster
```

## 5. Smoke-test the full pipeline

```bash
python -m pytest -q                                              # unit + integration tests
python scripts/run_experiment.py --config configs/fedavg.yaml --tier smoke   # trains on the fixture
```

## 6. Common pitfalls

- **`Python was not found` on Windows** — that's the Microsoft Store stub. Use WSL, not
  the Windows shell.
- **`ModuleNotFoundError: fl_med`** — run from the repo root, or `pip install -e .`
  (add `export PYTHONPATH=src` as a fallback).
- **CUDA `Segmentation fault` / OOM in WSL** — restart the WSL VM from Windows PowerShell
  (`wsl --shutdown`) to reset the GPU passthrough; for DP OOM lower
  `privacy.max_physical_batch_size`.
- **`git push: Could not resolve host`** — a WSL DNS drop; set `/etc/resolv.conf` to a
  public resolver (e.g. `nameserver 8.8.8.8`).
- **DP-SGD error about in-place ops / BatchNorm** — DP requires the GroupNorm model
  (configs already set `model.norm: group`); handled automatically by the DP engine.
