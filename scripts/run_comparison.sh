#!/usr/bin/env bash
# Run the full method comparison (Phase 2 + 3) across seeds, then aggregate.
#
#   bash scripts/run_comparison.sh [TIER] [DEVICE] [SEEDS...]
#   bash scripts/run_comparison.sh dev  cuda 0 1 2      # default
#   bash scripts/run_comparison.sh smoke cpu 0          # quick pipeline check
#
# Fair-comparison protocol: all federated strategies use the SAME optimizer (SGD),
# same model/data/tier/budget; only the strategy differs. Baselines (centralized,
# local-only) give the upper/lower references. Expected ordering on balanced
# accuracy: majority < local-only < FL < centralized.
set -euo pipefail

TIER="${1:-dev}"
DEVICE="${2:-cuda}"
shift || true; shift || true
SEEDS=("${@:-0 1 2}")
# shellcheck disable=SC2206
SEEDS=(${SEEDS[@]})

# Shared optimizer for the fair federated head-to-head (override _base/config defaults).
OPT_OVERRIDES=("optimizer.name=sgd" "optimizer.lr=0.01" "optimizer.momentum=0.9")

run () {  # run <config> <extra-overrides...>
  local cfg="$1"; shift
  echo ">>> [$TIER] $cfg  seed=$SEED  device=$DEVICE"
  python scripts/run_experiment.py --config "configs/${cfg}.yaml" \
      --tier "$TIER" --seed "$SEED" --device "$DEVICE" "$@"
}

echo "=== Comparison | tier=$TIER device=$DEVICE seeds=${SEEDS[*]} ==="
for SEED in "${SEEDS[@]}"; do
  # Baselines (their own optimizer defaults are fine for bounds).
  run centralized
  run local_only
  # Federated strategies with a shared optimizer for fairness.
  run fedavg   "${OPT_OVERRIDES[@]}"
  run fedprox  "${OPT_OVERRIDES[@]}"
  run scaffold "${OPT_OVERRIDES[@]}"
done

echo "=== Aggregating across seeds ==="
python scripts/aggregate_results.py --compare fedavg fedprox || true
python scripts/make_figures.py || true

echo "Done. Summary: experiments/comparison.json ; figures: reports/figures/"
