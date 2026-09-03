# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Changed
- Replaced the internal build log and context notes with `docs/design.md` (architecture,
  design decisions, lessons learned) and this changelog.
- Removed temporary files, legacy result trees, machine-specific paths and duplicate
  configuration directories from the repository.
- Demo scripts and manuals now use placeholders for host-specific values; real values live
  in an uncommitted local `.env` file.

## 2026-08-30

### Added
- One-command Raspberry Pi presentation demo (`scripts/live/presentation_demo.sh`,
  `START_PRESENTATION_DEMO.cmd`), DHCP-safe and tolerant of transient hotspots.
- Redesigned live dashboard with a labelled training chart.

## 2026-08-29

### Added
- Freeze-backbone partial-model FL for weak devices; verified on the Raspberry Pi 5
  (4/4 correctness checks, 8.28 s → 0.92 s per round, 8.99× speedup; straggler gap
  from ≈ 17× to < 2×). Self-validating benchmark `scripts/live/bench_pi.py`.
- Edge inference benchmark (`scripts/edge_infer_bench.py`): fp32, TorchScript, dynamic
  and static INT8.
- Live dashboards: `reports/fl_process_visualizer.html` (replay of the real 30-round run)
  and `reports/live_dashboard.html` (polls the running server).

### Changed
- Technical report regenerated (11 pages) with the straggler-mitigation result.
- CI lint pinned to an explicit ruff rule set; remaining lint issues fixed.

## 2026-08-12

### Added
- Byzantine robustness: sign-flip model-poisoning attack and coordinate-median /
  trimmed-mean / Krum aggregators (`src/fl_med/security/robust_agg.py`,
  `scripts/run_robustness.py`).
- FedAdam server optimiser (`strategies/fedadam.py`); did not beat FedAvg here.
- Grad-CAM explainability bench (`scripts/run_gradcam.py`); attention is diffuse and
  border-heavy on the compute-limited model.
- Communication efficiency: layer-wise top-k sparsification, 50× smaller uploads at
  ~no accuracy cost (`scripts/run_comms.py`).
- Self-explanatory results dashboard (`reports/dashboard.html`) and defence slide deck.
- Report sections 3.7–3.8; torch-free correctness checks extended to 20.

## 2026-08-11

### Added
- Real distributed FL over Flower/gRPC (`src/fl_med/federated_live/`, `scripts/live/`):
  laptop server + GPU client and a Raspberry Pi 5 CPU client. Warm-started 8-round run at
  ≈ 0.20 balanced accuracy; measured ≈ 17× straggler (Pi 8.2 s vs laptop 0.5 s per round).
- Raspberry Pi set-up guide; report section 3.6.

## 2026-07-31

### Added
- Full-tier results (ResNet-18 + GroupNorm, 128 px, 30 rounds, 3 seeds) for centralized,
  FedAvg, FedProx, SCAFFOLD and local-only.
- Technical report (Word), user manual, installation manual, report generator.

## 2026-07-28

### Added
- Secure aggregation with antisymmetric pairwise masks (`security/secure_agg.py`);
  exact aggregate recovered (error ≈ 3e-13).
- Resumable full-tier launcher `scripts/run_full.sh`.
- Membership-inference attack (`security/attacks/mia.py`, `scripts/run_mia.py`):
  AUC 0.555 non-private → 0.503 under DP-SGD.

## 2026-07-26

### Added
- DP privacy–utility curve (σ sweep) and figure.

## 2026-07-24

### Added
- DP-SGD via Opacus with an independent pure-numpy Rényi-DP accountant and per-client
  (ε, δ) reporting every round; Opacus-compatible ResNet (non-in-place ReLU, patched
  residual); `BatchMemoryManager` for large logical batches.
- Dev-tier 3-seed comparison of the five methods.

## 2026-07-20

### Changed
- Class-weighted loss, GroupNorm and larger dev volumes so federated training escapes
  majority-class collapse.
- SCAFFOLD stabilised with plain SGD (momentum 0) and a gradient-norm safety clip.

## 2026-07-11

### Added
- Clean `fl_med` package: FedAvg, FedProx and corrected SCAFFOLD on one shared training
  loop; tiered configs (`smoke`/`dev`/`full`); pytest suite; GitHub Actions CI; committed
  synthetic fixture; non-IID heterogeneity analysis of the real metadata.

### Changed
- Legacy week-1/2 scripts moved to `archive/` (since removed).

## 2026-03-26 – 2026-03-27

### Added
- Initial project structure, environment freeze, dataset utilities, Fed-ISIC2019 download
  helper, centralized baseline with scheduler and early stopping, local-only baselines.
