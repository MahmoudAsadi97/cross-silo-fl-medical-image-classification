#!/usr/bin/env bash
# Privacy-utility sweep (Phase 4): matched non-private FedAvg baseline + DP-SGD at
# several noise multipliers. Produces per-run summaries; plot with plot_dp_curve.py.
#
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_dp_sweep.sh dev cuda 0
set -euo pipefail
TIER="${1:-dev}"; DEVICE="${2:-cuda}"; SEED="${3:-0}"
EXTRA=(); [[ -n "${DATA_ROOT:-}" ]] && EXTRA+=("data.root=${DATA_ROOT}")

echo "=== DP privacy-utility sweep | tier=$TIER seed=$SEED ==="
# matched non-private baseline (GroupNorm + Adam, DP disabled)
python scripts/run_experiment.py --config configs/dp_fedavg.yaml --tier "$TIER" \
    --seed "$SEED" --device "$DEVICE" --output "experiments/dp_none_${TIER}_seed${SEED}" \
    "${EXTRA[@]}" privacy.enabled=false

for SIGMA in 0.5 1.0 2.0 4.0; do
  python scripts/run_experiment.py --config configs/dp_fedavg.yaml --tier "$TIER" \
      --seed "$SEED" --device "$DEVICE" --output "experiments/dp_s${SIGMA}_${TIER}_seed${SEED}" \
      "${EXTRA[@]}" privacy.enabled=true privacy.noise_multiplier=${SIGMA}
done
echo "Done. Plot: python scripts/plot_dp_curve.py $TIER"
