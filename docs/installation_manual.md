# Installation manual

## Requirements
- Python 3.10+
- ~2 GB disk for the environment; the real dataset is several GB more.
- (Optional) NVIDIA GPU + CUDA for `full`-tier training.

## 1. Environment
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
```

## 2. Install the package
```bash
pip install -e .                      # pure-python core (config, metrics, math, heterogeneity)
pip install -e ".[torch,dev]"         # + torch/torchvision + pytest/ruff (training & tests)
pip install -e ".[torch,privacy,dashboard,edge,dev]"   # everything
```
The core installs without torch on purpose, so the correctness checks and the
heterogeneity analysis run on minimal machines.

## 3. Verify
```bash
python scripts/verify_core_math.py    # expect "12/12 checks passed"
python scripts/generate_fixture.py    # writes the tiny fixture
make smoke                            # end-to-end FedAvg on the fixture (needs torch)
make test                             # pytest (torch tests auto-skip without torch)
```

## 4. Get the data
See `data/README.md`. Not needed for smoke/tests.

## Troubleshooting
- **`make` unavailable (Windows):** run the command under each Makefile target
  directly, or set `PYTHONPATH=src` and call the `scripts/*.py` files.
- **torch install slow:** use CPU wheels — `pip install torch torchvision
  --index-url https://download.pytorch.org/whl/cpu`.
