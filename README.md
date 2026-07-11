# Cross-Silo Federated Learning for Medical Image Classification

Engineering and rigorously evaluating a **cross-silo federated learning (FL)**
system for skin-lesion classification, where several simulated hospitals train a
shared model **without exchanging raw patient images**. The project quantifies the
trade-offs between model performance, data heterogeneity (non-IID), privacy
(differential privacy), and edge-deployment feasibility on a Raspberry Pi 5.

> **Research question.** How can cross-silo Federated Learning be engineered to
> train clinically-inspired image-classification models without sharing raw data,
> while quantifying the trade-offs between model performance, data heterogeneity,
> privacy guarantees, and edge-deployment feasibility?

**Dataset:** [Fed-ISIC2019](https://github.com/owkin/FLamby) — ~23,247 dermoscopy
images, 8 diagnostic classes, split across **6 natural cross-silo clients** (4
hospitals; one contributes 3 clients via 3 imaging technologies). Severe class
imbalance and non-IID label distributions. Official metric: **balanced accuracy**.

## Status
Phase 0 complete: a clean, tested, reproducible pipeline. Baselines and federated
strategies are code-complete and unit-tested; full training numbers are produced on
GPU. See `CHANGELOG.md` for the live state.

## What's implemented
- **Strategies:** FedAvg, FedProx, SCAFFOLD (corrected option-II control variates).
- **Baselines:** centralized (upper bound) and local-only (lower reference).
- **Privacy:** GroupNorm models wired for Opacus DP-SGD; per-client (ε, δ) accounting *(in progress)*.
- **Heterogeneity:** per-client entropy, KL / Jensen-Shannon / Hellinger to the global
  pool, 1-D EMD, missing-class counts — with figures.
- **Reproducibility:** tiered configs (`smoke`/`dev`/`full`), global seeding, per-run
  manifests (git hash + hardware), multi-seed aggregation, paired Wilcoxon tests.
- **Rigor:** unit tests for aggregation, SCAFFOLD equations, and metrics; a torch-free
  correctness runner; GitHub Actions CI.

## Quickstart
```bash
git clone <repo> && cd cross-silo-fl-medical-image-classification
python -m pip install -e ".[torch,dev]"        # core-only: pip install -e .

python scripts/verify_core_math.py             # torch-free correctness checks
python scripts/analyze_heterogeneity.py        # non-IID analysis (needs the real data)
make smoke                                      # end-to-end FedAvg on the tiny fixture
make test                                       # unit + integration tests
```
See `docs/installation_manual.md` and `docs/user_manual.md` for details, and
`data/README.md` to obtain Fed-ISIC2019.

## Tiers
Everything is config-driven; a `--tier` flag selects scale:
`smoke` (synthetic fixture, seconds, CI) · `dev` (small real subset, CPU) ·
`full` (all data, all seeds, GPU — the report numbers).

## Repository layout
```
src/fl_med/     package: data, models, strategies, engine, privacy, security, edge
configs/        one tiered YAML per experiment
scripts/        run_experiment, analyze_heterogeneity, aggregate_results, make_figures
experiments/    per-run outputs (metrics.csv, run_config.yaml, figures)
tests/          unit + integration
docs/           installation & user manuals
dashboard/      Streamlit app (reads experiments/)
reports/        technical report + figures
```

## License
MIT — see `LICENSE`.
