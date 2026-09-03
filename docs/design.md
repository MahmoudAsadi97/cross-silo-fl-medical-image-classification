# Design notes — cross-silo federated learning for skin-lesion classification

This document is the map of the project: what it is for, how the code is organised,
the design decisions that hold it together, and the lessons that came out of running it
on real hardware. Results live in the [README](../README.md) and the technical report;
the release history is in [CHANGELOG.md](../CHANGELOG.md).

## 1. Purpose and research question

Hospitals hold exactly the data needed to train good diagnostic models and are the least
able to share it. Cross-silo federated learning (FL) lets a small number of institutions
train one model while every raw image stays where it was collected.

**Research question.** How can cross-silo FL be engineered to train clinically relevant
image-classification models without sharing raw data, while quantifying the trade-offs
between model performance, data heterogeneity and privacy guarantees?

Sub-questions, each answered with a generated artefact (metric, figure, test or log):

1. What non-IID heterogeneity exists across institutions, and how can it be quantified?
2. How does FedAvg compare with centralized training and with isolated (local-only) training?
3. Does FedProx improve stability or performance under heterogeneity?
4. Does SCAFFOLD reduce client drift compared with FedAvg and FedProx?
5. What is the performance cost of DP-SGD?
6. How does the privacy budget (ε, δ) evolve over rounds?
7. Which security and privacy risks remain, and how do secure aggregation and robust
   aggregation mitigate server-side and client-side threats?
8. When is FL preferable to centralized learning or data-sharing agreements?

## 2. Dataset

**Fed-ISIC2019** from the FLamby benchmark: ~23,247 dermoscopy images, 8 diagnostic
classes, six *naturally partitioned* centres (four hospitals; one contributes three
centres through three imaging devices). Class imbalance is severe (melanocytic nevus
dominates) and the label mix differs per centre. The official metric is balanced accuracy
(mean per-class recall; chance = 0.125); macro-F1 is reported alongside.

Training-set sizes and missing classes per centre: c0 7,947 (0 missing), c1 2,531 (2),
c2 2,156 (0), c3 1,448 (0), c4 525 (5), c5 281 (2). Centre 5 is the natural "edge
hospital" and is the one that runs on the Raspberry Pi.

The images are not redistributed with this repository; `data/README.md` describes how to
obtain them and the expected folder layout. A tiny synthetic fixture is committed so the
whole pipeline, tests and CI run without the real data.

## 3. Repository map

```
src/fl_med/
  config.py            tiered YAML resolution (smoke/dev/full) + dotted CLI overrides
  seeding.py           global seeding (Python, numpy, torch, CUBLAS workspace)
  logging.py           run manifest (git commit, tier, hardware) + YAML sanitiser
  metrics.py           pure-numpy balanced accuracy, macro-F1, confusion matrix, AUC
  eval.py              multi-seed summaries, paired tests, plots
  losses.py            inverse-frequency weighted cross-entropy
  data/                folder-backed dataset, tier/seed-aware loaders, heterogeneity
                       metrics (entropy, KL/JS/Hellinger, EMD), fixture generator
  models/              small_cnn, resnet18 (+ BatchNorm→GroupNorm), mobilenet_v2,
                       efficientnet_b0
  strategies/          aggregation, base (hook interface), fedavg, fedprox, scaffold,
                       fedadam
  engine/              train_eval (one shared loop), client (local training + DP path),
                       server (round loop, drift diagnostic, per-client ε accounting),
                       baselines (centralized, local-only)
  privacy/             accounting.py (independent RDP accountant), dp_engine.py (Opacus)
  security/            secure_agg.py (pairwise masks), robust_agg.py (median, trimmed
                       mean, Krum), attacks/ (membership inference, sign-flip poisoning)
  federated_live/      networked FL over Flower/gRPC, reusing the same engine

configs/     one tiered YAML per experiment (fedavg, fedprox, scaffold, fedadam, dp, live…)
scripts/     run_experiment (single entry point), run_full (resumable launcher),
             robustness / communication / Grad-CAM / MIA benches, figure and report builders
scripts/live/ server, client, Pi benchmark, demo launchers for the real distributed run
tests/       unit + integration tests (torch-dependent tests skip when torch is absent)
experiments/ per-run outputs: metrics.csv, run_config.yaml, summary.json, figures
docs/        installation, user and Raspberry Pi manuals; this document
reports/     technical report, dashboards, presentation, figures
```

## 4. Design decisions and rationale

**One training loop for every strategy.** FedAvg, FedProx, SCAFFOLD and FedAdam share the
same client loop, model initialisation, data splits, compute budget (rounds × local
epochs), evaluation set and metric. A strategy differs only through hooks
(`extra_loss`, `after_backward`, `aggregate`). A difference in results is therefore
attributable to the algorithm, not to the harness.

**Tiers are a flag, never a code edit.** `--tier smoke|dev|full` selects the committed
fixture, a real-data development setting, or the full schedule. Every run writes a
manifest with the git commit, tier and hardware so a number can always be traced back.

**GroupNorm instead of BatchNorm for the DP-comparable track.** Opacus cannot take
per-sample gradients through BatchNorm, and BatchNorm running statistics are ill-defined
across non-IID centres. The non-private baseline uses the same GroupNorm model so that
DP noise is the *only* difference in the privacy–utility comparison. This costs absolute
accuracy relative to the published BatchNorm/EfficientNet-B0 benchmark, which is why the
README reports the setting explicitly.

**Inverse-frequency weighted cross-entropy.** Plain cross-entropy collapses to the
majority class (balanced accuracy = 1/8). Class weights are computed from label counts
only, which is shareable meta-information, never from images.

**SCAFFOLD with option-II control variates and plain SGD.** `c_i⁺ = c_i − c + (x − y_i)/(K·η)`
with momentum 0, so the control variate is a bounded mean gradient (momentum and Adam
produced NaNs). A gradient-norm clip of 1e3 is a numerical safety net, disabled under DP
because Opacus clips per sample.

**An independent privacy accountant.** `privacy/accounting.py` is a pure-numpy Rényi-DP
accountant used to report per-client (ε, δ) every round; δ = 1e-5 < 1/N for every centre.

**Secure aggregation as additive antisymmetric pairwise masks.** Masks cancel to the exact
weighted average (verified error ≈ 3e-13) while each masked update on its own is
uninformative. Key agreement and dropout handling are out of scope and stated as such.

**Reporting.** Balanced accuracy is reported as best-round and final-round, mean ± std
over ≥ 3 seeds. Best-round is selected on the test set because no validation split is
materialised; this is a mild optimistic bias and is stated wherever the numbers appear.

**The networked run warm-starts from a pre-trained global model.** A short real
distributed run cannot converge from scratch, and a single laptop GPU cannot host many
client processes. Continued FL from a checkpoint is legitimate and makes the run
representative; the rigorous accuracy comparison stays with the simulation, the live run
proves distribution and timing.

## 5. Experimental protocol

* `smoke`: committed synthetic fixture, tiny; used by CI.
* `dev`: real data, ResNet-18 + GroupNorm, 64 px, 15 rounds × 60 batches.
* `full`: real data, 128 px, 30 rounds, seeds 0/1/2 — the numbers in the README and report.
  (The base config lists 200 px / 50 rounds; the full run used 128 px / 30 rounds via
  environment overrides for laptop-GPU feasibility.)

Every run writes `experiments/<name>_<tier>_seed<k>/` with `run_config.yaml`,
`metrics.csv`, `summary.json`, `curves.png` and `drift.png`. `scripts/run_full.sh` is
resumable: a run whose `summary.json` exists is skipped.

## 6. Real distributed FL on a Raspberry Pi 5

The strategy comparison is a simulation (clients trained sequentially on one GPU). To
show the system is genuinely distributed, `federated_live/` wraps the same engine behind
Flower/gRPC: the laptop runs the server, central evaluation and a fast GPU client
(centre 0); a Raspberry Pi 5 runs centre 5 as a CPU client and exchanges only model
updates over the network.

Results: sustained balanced accuracy ≈ 0.20 (peak 0.213) over 8 warm-started rounds, and
a measured **straggler** effect — the Pi needed ≈ 8.2 s per round against ≈ 0.5 s for the
laptop GPU (≈ 17×), so every synchronous round was gated by the Pi. Freezing the backbone
and training only the classifier head on the Pi (partial-model FL; the architecture is
unchanged, so FedAvg still aggregates) cut the Pi's round time from 8.28 s to 0.92 s
(8.99×, verified by the self-checking `scripts/live/bench_pi.py`, 4/4 correctness checks),
shrinking the gap to under 2×.

Set-up, networking and troubleshooting are in `docs/raspberry_pi_setup.md`. The demo
scripts read every host-specific value (SSH target, paths, key file) from a local `.env`
file that is not committed; `configs/presentation_demo.env.example` shows the shape.

## 7. Lessons learned

* Do not run many GPU client processes on one laptop: six concurrent trainers contend for
  the GPU (~345 s per round) and exhaust WSL memory. Use the sequential simulation for
  accuracy and a few light processes across machines for the distributed demo.
* WSL2 networking is NAT'd; a server bound inside WSL is not reachable from the LAN by
  default. Mirrored networking needs a recent Windows build; `netsh interface portproxy`
  plus a firewall rule is the reliable fallback. WSL cannot resolve `.local` names.
* Phone hotspots are a poor choice for device-to-device traffic (tiny subnet, possible
  client isolation, DHCP addresses that drift). Prefer a router or direct Ethernet.
* Opacus: disable in-place ReLU and patch the ResNet residual to `out = out + identity`;
  use `BatchMemoryManager` for large logical batches; keep `poisson_sampling=True`.
* A balanced accuracy frozen at 0.125 is majority-class collapse, not an optimiser bug:
  weighted loss plus enough training volume fixes it.
* rsync `--exclude=data` matches *any* directory called `data` and silently dropped
  `src/fl_med/data/` when copying to the Pi. Anchor top-level excludes with a leading slash.
* On the Pi, `pip install -e .` without `--no-deps` upgraded numpy to 2.x and broke
  `flwr`; pin numpy 1.26 and a compatible scipy, or install with `--no-deps`.

## 8. Known limitations and next steps

* Absolute accuracy is below the published Fed-ISIC2019 benchmark because of the GroupNorm
  / 128 px setting; a benchmark-parity track (EfficientNet-B0, BatchNorm, 200 px) is planned.
* No materialised validation split; model selection uses the test set (stated above).
* Secure aggregation and robust aggregation run in the simulation; the networked path uses
  plain FedAvg. Moving them into the Flower path, with TLS and node authentication, is the
  next engineering milestone.
* FedProx, SCAFFOLD and FedAdam reduce drift but did not beat FedAvg on accuracy in this
  setting; Grad-CAM shows diffuse, border-heavy attention on the compute-limited model.
  Both are reported as findings, not hidden.

The full upgrade plan is tracked in the repository's issues.
