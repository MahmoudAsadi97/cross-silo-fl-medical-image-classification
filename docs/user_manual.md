# User Manual

How to run experiments and read the results of the `fl_med` cross-silo federated
learning system. (Install first — see `docs/installation_manual.md`.)

## 1. Key concepts

- **Tier** (`--tier smoke|dev|full`) selects scale, as a flag, never a code edit:
  `smoke` = tiny committed fixture (seconds, no GPU/data); `dev` = small real-data
  subset on a single GPU (minutes); `full` = all data, full schedule (hours, for the
  report numbers).
- **Config**: one YAML per experiment in `configs/`, layered on `configs/_base.yaml`.
  Override anything on the command line, e.g. `training.epochs=5 optimizer.lr=0.01`.
- **Outputs**: every run writes to `experiments/<name>_<tier>_seed<k>/`:
  `run_config.yaml` (git commit + tier + hardware for provenance), `metrics.csv`
  (per-round/epoch), `summary.json`, and curve PNGs.

Set `DATA_ROOT` once so runs read the fast native-disk copy:
`export DATA_ROOT=$HOME/fl_data/fed_isic2019/raw`.

## 2. Running single experiments

```bash
# baselines
python scripts/run_experiment.py --config configs/centralized.yaml --tier dev --device cuda
python scripts/run_experiment.py --config configs/local_only.yaml  --tier dev --device cuda
# federated strategies
python scripts/run_experiment.py --config configs/fedavg.yaml   --tier dev --seed 0 --device cuda
python scripts/run_experiment.py --config configs/fedprox.yaml  --tier dev --seed 0 --device cuda
python scripts/run_experiment.py --config configs/scaffold.yaml --tier dev --seed 0 --device cuda
```

## 3. Full comparison and analyses

```bash
# non-IID heterogeneity analysis (figures + CSV)   -> experiments/heterogeneity/
python scripts/analyze_heterogeneity.py

# 3-seed method comparison at dev tier, then aggregate + figures
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_comparison.sh dev cuda 0 1 2

# differential privacy: privacy-utility sweep + curve
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_dp_sweep.sh dev cuda 0
python scripts/plot_dp_curve.py dev

# empirical privacy: membership-inference attack (non-private vs DP)
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw python scripts/run_mia.py --device cuda --rounds 15

# secure aggregation demonstration (no GPU needed)
python scripts/run_secure_agg.py

# aggregate any set of runs into a mean+/-std table
python scripts/aggregate_results.py --tier dev --compare fedavg fedprox
```

## 4. Full-tier runs (report numbers)

`scripts/run_full.sh` runs the whole comparison + DP sweep at full tier. It is
**resumable**: any run whose `summary.json` exists is skipped, so after an interruption
just re-run the same command. Use `tmux` so it survives disconnects.

```bash
tmux new -s full
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw IMG=128 ROUNDS=30 bash scripts/run_full.sh cuda "0 1 2"
# detach: Ctrl-b then d   |   reattach: tmux attach -t full
```

`IMG`/`ROUNDS` are optional overrides (image 200 / 50 rounds is the maximal FLamby
setting but very slow on 8 GB; 128 / 30 is a strong, faster compromise).

## 5. Reading the results

- **Balanced accuracy** (mean per-class recall) is the primary metric (FLamby standard);
  macro-F1 is secondary. Accuracy alone is misleading on this imbalanced data.
- **client_drift** in a federated `metrics.csv` measures how far clients pull apart;
  FedProx and especially SCAFFOLD reduce it.
- **eps_max** (DP runs) is the per-client privacy budget; it grows each round.
- Regenerate report figures anytime: `python scripts/make_figures.py`,
  `python scripts/plot_comparison.py dev`.

## 6. Rebuilding the technical report

```bash
node reports/build_report.js        # writes reports/technical_report.docx from experiments/
```

## 7. Reproducing a past run

Open its `experiments/.../run_config.yaml` — it records the exact config, git commit,
tier and hardware. Re-run the same `--config`/`--tier`/overrides to reproduce it.
