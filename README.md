# Cross-Silo Federated Learning for Privacy-Preserving Medical Image Classification

Six naturally partitioned hospital data silos (the real centres of Fed-ISIC2019) train one skin-lesion classifier **without ever exchanging raw
patient images**, and every trade-off is measured: accuracy vs centralized training,
data heterogeneity (non-IID), differential privacy, attack resistance (confidentiality
*and* integrity), communication cost, and real edge deployment on a **Raspberry Pi 5**.

> **Research question.** How can cross-silo Federated Learning be engineered to train
> clinically relevant image-classification models without sharing raw data, while
> quantifying the trade-offs between model performance, data heterogeneity, and
> privacy guarantees?

**Dataset:** [Fed-ISIC2019](https://github.com/owkin/FLamby) — ~23,247 dermoscopy images,
8 diagnostic classes, **6 natural cross-silo clients** (sizes 281–7,947; severe class
imbalance and non-IID label mixes). Official metric: **balanced accuracy** (random = 0.125).

## Status: COMPLETE
All 8 research sub-questions answered with reproducible, artifact-backed evidence.
New here? **Read [`docs/design.md`](docs/design.md) first** — the map of the whole project
(architecture, design decisions, lessons learned). Release history: [`CHANGELOG.md`](CHANGELOG.md).

## Headline results (full tier: ResNet-18+GroupNorm, 128 px, 30 rounds, 3 seeds)

| Method | Balanced accuracy | Client drift |
|---|---|---|
| Centralized (upper bound) | **0.456 ± 0.004** | – |
| **FedAvg (federated)** | **0.320 ± 0.030** | 8.56 |
| FedProx | 0.224 ± 0.024 | 1.55 |
| SCAFFOLD | 0.217 ± 0.003 | 0.92 |
| Local-only (lower ref.) | 0.209 ± 0.008 | – |

- **Federation works:** FedAvg recovers ~45% of the isolated→pooled accuracy gap with no
  image leaving a silo. **Honest finding:** FedProx/SCAFFOLD (and FedAdam) cut drift 5–9×
  but do **not** beat well-tuned FedAvg here — drift control buys stability, not accuracy.
- **Privacy, layered & verified:** DP-SGD with an independently-built Rényi-DP accountant
  (clean monotonic privacy–utility curve, 0.336 → 0.151); a **membership-inference attack**
  drops from AUC 0.555 → 0.503 (chance) under DP; **secure aggregation** recovers the exact
  average (err ≈ 3×10⁻¹³) while hiding every individual update.
- **Integrity:** a 2-of-6 **model-poisoning attack** collapses plain FedAvg to random
  (0.125); robust aggregators (coordinate-median / trimmed-mean / Krum) defend (~0.19).
- **Efficiency:** layer-wise top-k sparsification → **50× smaller uploads** (44.7 → 0.89 MB)
  at essentially no accuracy cost.
- **Real hardware:** genuinely distributed run (Flower/gRPC) — laptop server + **Raspberry
  Pi 5** as a real edge hospital. Measured **17× straggler**, then **mitigated it**:
  freeze-backbone partial-model FL gives a **verified 8.99× speedup** on the Pi
  (8.28 → 0.92 s/round; 4/4 on-device correctness checks; gap shrinks to <2×).

## Deliverables
- `reports/technical_report.docx` — 11-page technical report (all figures, 13 IEEE refs)
- `reports/dashboard.html` — self-explanatory results dashboard (single file, open in a browser)
- `reports/fl_process_visualizer.html` — animated replay of the real 30-round federated run
- `reports/live_dashboard.html` — real-time view of a running federation (polls the live server)
- `reports/presentation.pptx` — defense slide deck (speaker notes included)
- `docs/installation_manual.md` · `docs/user_manual.md` · `docs/raspberry_pi_setup.md`

## Quickstart
```bash
git clone <repo> && cd cross-silo-fl-medical-image-classification
python -m pip install -e ".[torch,dev]"        # core-only: pip install -e .

python scripts/verify_core_math.py             # 21 torch-free correctness checks
make smoke                                     # end-to-end FedAvg on the tiny fixture
make test                                      # unit + integration tests
```
Real-data runs (see manuals for details):
```bash
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_comparison.sh dev cuda 0 1 2
DATA_ROOT=$HOME/fl_data/fed_isic2019/raw IMG=128 ROUNDS=30 bash scripts/run_full.sh cuda "0 1 2"
```

## Rigor & reproducibility
One identical training loop for every strategy (fair-comparison protocol); tiered configs
(`smoke`/`dev`/`full`) switched by a flag; global seeding + per-run manifests (git commit,
tier, hardware); ≥3 seeds for headline numbers; **21/21 pure-numpy correctness checks**
(aggregation, SCAFFOLD equations, metrics, DP accountant, secure aggregation, robust
aggregators) runnable without a GPU; pytest + GitHub Actions CI. Negative results
(FedAdam, Grad-CAM shortcut) are reported openly.

## Repository layout
```
src/fl_med/       package: data, models, strategies, engine, privacy, security, federated_live
configs/          one tiered YAML per experiment (fedavg/fedprox/scaffold/fedadam/dp/live...)
scripts/          run_experiment, run_full, robustness/comms/gradcam/MIA benches, build_* tools
scripts/live/     REAL networked FL (Flower): server, client, Pi benchmark, demo launchers
experiments/      per-run outputs (metrics.csv, run_config.yaml, summary.json, figures)
tests/            unit + integration
docs/             installation, user & Raspberry-Pi manuals
reports/          technical report, dashboards, presentation, figures
```

## License
**Permission-required** — the repository is public for viewing and academic evaluation,
but any use, copying, modification, or redistribution requires the author's prior
written permission (asadiherism@gmail.com). See `LICENSE` for the full terms.
