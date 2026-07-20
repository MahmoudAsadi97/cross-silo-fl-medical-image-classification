#!/usr/bin/env bash
# Run the full method comparison (Phase 2 + 3) across seeds, then aggregate.
#
#   bash scripts/run_comparison.sh [TIER] [DEVICE] [SEEDS...]
#   bash scripts/run_comparison.sh dev  cuda 0 1 2      # default
#   bash scripts/run_comparison.sh smoke cpu 0          # quick pipeline check
#
# All methods share the SAME optimizer/model/data/tier/budget (fair-comparison
# protocol); only the strategy differs. Baselines (centralized, local-only) give
# the upper/lower references. Expected ordering on balanced accuracy:
# majority < local-only < FL < centralized.
#
# Speed: set DATA_ROOT to a WSL-native copy to avoid slow /mnt/c image reads, e.g.
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_comparison.sh dev cuda 0 1 2
set -euo pipefail

TIER="${1:-dev}"
DEVICE="${2:-cuda}"
shift || true; shift || true
SEEDS=("${@:-0 1 2}")
# shellcheck disable=SC2206
SEEDS=(${SEEDS[@]})

EXTRA=()
if [[ -n "${DATA_ROOT:-}" ]]; then EXTRA+=("data.root=${DATA_ROOT}"); fi

run () {  # run <config>
  local cfg="$1"; shift
  echo ">>> [$TIER] $cfg  seed=$SEED  device=$DEVICE"
  python scripts/run_experiment.py --config "configs/${cfg}.yaml" \
      --tier "$TIER" --seed "$SEED" --device "$DEVICE" "${EXTRA[@]}" "$@"
}

echo "=== Comparison | tier=$TIER device=$DEVICE seeds=${SEEDS[*]} data_root=${DATA_ROOT:-<default>} ==="
for SEED in "${SEEDS[@]}"; do
  run centralized
  run local_only
  run fedavg
  run fedprox
  run scaffold
done

echo "=== Aggregating across seeds ==="
python scripts/aggregate_results.py --compare fedavg fedprox || true
python scripts/make_figures.py || true
echo "Done. Summary: experiments/comparison.json ; figures: reports/figures/"
