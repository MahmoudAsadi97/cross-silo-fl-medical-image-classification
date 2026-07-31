#!/usr/bin/env bash
# Full-tier launcher for the FINAL report numbers. Resilient + RESUMABLE: each run
# is independent, failures don't abort the batch, and any run whose summary.json
# already exists is SKIPPED -- so after a crash/shutdown just re-run this and it
# continues where it left off.
#
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw IMG=128 ROUNDS=30 bash scripts/run_full.sh cuda "0 1 2"
set -uo pipefail

DEVICE="${1:-cuda}"
SEEDS="${2:-0 1 2}"
IMG="${IMG:-}"; ROUNDS="${ROUNDS:-}"
EXTRA=(); [[ -n "${DATA_ROOT:-}" ]] && EXTRA+=("data.root=${DATA_ROOT}")
[[ -n "$IMG" ]]    && EXTRA+=("data.image_size=${IMG}")
[[ -n "$ROUNDS" ]] && EXTRA+=("federated.rounds=${ROUNDS}" "training.epochs=${ROUNDS}")

run () {  # run <name> <config> <extra...>
  local name="$1" cfg="$2"; shift 2
  local out="experiments/${name}_full_seed${SEED}"
  if [[ -f "$out/summary.json" ]]; then
    echo "=== skip ${name} seed=${SEED} (already done)"; return
  fi
  echo ">>> [full] ${name} seed=${SEED}  $(date +%H:%M:%S)"
  python scripts/run_experiment.py --config "configs/${cfg}.yaml" --tier full \
      --seed "$SEED" --device "$DEVICE" --output "$out" "${EXTRA[@]}" "$@" \
      || echo "!!! ${name} seed=${SEED} FAILED (continuing)"
}

echo "=== FULL-TIER BATCH (resumable) | device=$DEVICE seeds=[$SEEDS] img=${IMG:-200} $(date) ==="
for SEED in $SEEDS; do
  run centralized centralized
  run local_only  local_only
  run fedavg      fedavg
  run fedprox     fedprox
  run scaffold    scaffold
done

# DP privacy-utility sweep at full tier (1 seed).
SEED=0
run dp_none dp_fedavg data.batch_size=128 federated.max_batches=null \
    privacy.enabled=false privacy.max_physical_batch_size=8
for SIGMA in 0.5 1.0 2.0 4.0; do
  run "dp_s${SIGMA}" dp_fedavg data.batch_size=128 federated.max_batches=null \
      privacy.enabled=true privacy.noise_multiplier=${SIGMA} privacy.max_physical_batch_size=8
done

echo "=== aggregating + figures ==="
python scripts/aggregate_results.py --tier full --compare fedavg fedprox || true
python scripts/plot_comparison.py full || true
python scripts/plot_dp_curve.py full || true
python scripts/make_figures.py || true
echo "=== FULL-TIER BATCH DONE $(date) ==="
