# User manual

## Concepts
- **Tier** (`--tier smoke|dev|full`): scale of a run. `smoke` = synthetic fixture,
  seconds; `dev` = small real subset on CPU; `full` = all data/seeds on GPU.
- **Config**: one YAML per experiment in `configs/`, layered on `configs/_base.yaml`.
  Override anything on the CLI: `training.epochs=5 optimizer.lr=0.01`.

## Running experiments
```bash
# baselines
python scripts/run_experiment.py --config configs/centralized.yaml --tier dev
python scripts/run_experiment.py --config configs/local_only.yaml  --tier dev

# federated strategies
python scripts/run_experiment.py --config configs/fedavg.yaml   --tier dev --seed 0
python scripts/run_experiment.py --config configs/fedprox.yaml  --tier dev --seed 0
python scripts/run_experiment.py --config configs/scaffold.yaml --tier dev --seed 0
```
Outputs go to `experiments/<name>_<tier>_seed<k>/`: `run_config.yaml` (provenance),
`metrics.csv`, `summary.json`, and curve/drift PNGs.

## Analysis
```bash
python scripts/analyze_heterogeneity.py                 # non-IID metrics + figures
python scripts/aggregate_results.py --compare fedavg fedprox   # mean±std + Wilcoxon
python scripts/make_figures.py                          # regenerate report figures
```

## Multi-seed (report numbers)
Run each config for seeds 0,1,2 (`--seed k`), then `aggregate_results.py` reports
mean±std and a paired test. Never report a single-seed number.

## Reproducing a past run
Open its `experiments/.../run_config.yaml`; it records the exact config, git commit,
tier, and hardware. Re-run the same `--config`/`--tier`/overrides to reproduce.
