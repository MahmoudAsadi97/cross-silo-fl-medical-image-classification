#!/usr/bin/env bash
# Privacy-utility sweep (Phase 4): matched non-private FedAvg baseline + DP-SGD at
# several noise multipliers. DP-SGD needs a LARGE batch for usable signal-to-noise
# (noise ~ sigma/batch after averaging), so this uses batch=128 + all data/round +
# more rounds than the standard dev tier. Plot with plot_dp_curve.py.
#
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/run_dp_sweep.sh dev cuda 0
set -euo pipefail
TIER="${1:-dev}"; DEVICE="${2:-cuda}"; SEED="${3:-0}"
EXTRA=(); [[ -n "${DATA_ROOT:-}" ]] && EXTRA+=("data.root=${DATA_ROOT}")

# DP-friendly training budget (bigger batch + more rounds + all data per round).
DP_TRAIN=(data.batch_size=128 federated.rounds=20 federated.max_batches=null data.num_workers=2)

echo "=== DP privacy-utility sweep | tier=$TIER seed=$SEED (batch=128, 20 rounds) ==="
# matched non-private baseline (GroupNorm + Adam, same budget, DP disabled)
python scripts/run_experiment.py --config configs/dp_fedavg.yaml --tier "$TIER" \
    --seed "$SEED" --device "$DEVICE" --output "experiments/dp_none_${TIER}_seed${SEED}" \
    "${EXTRA[@]}" "${DP_TRAIN[@]}" privacy.enabled=false

for SIGMA in 0.5 1.0 2.0 4.0; do
  python scripts/run_experiment.py --config configs/dp_fedavg.yaml --tier "$TIER" \
      --seed "$SEED" --device "$DEVICE" --output "experiments/dp_s${SIGMA}_${TIER}_seed${SEED}" \
      "${EXTRA[@]}" "${DP_TRAIN[@]}" privacy.enabled=true privacy.noise_multiplier=${SIGMA}
done
echo "Done. Plot: python scripts/plot_dp_curve.py $TIER"
