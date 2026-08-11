#!/usr/bin/env bash
# LOCAL end-to-end test of the real-FL path: a server + N clients as SEPARATE
# processes on this one machine (localhost). Proves the networked pipeline works
# AND that the federation actually learns, before the Raspberry Pi is involved.
# Same code paths as the cross-machine run; only the addresses are 127.0.0.1.
#
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/live/run_local_demo.sh
#
# Knobs: ROUNDS (10), MAXB batches/round/client (30), DEVICE (cuda; set cpu if
# the GPU can't hold 6 processes), CLIENTS ("0 1 2 3 4 5").
set -uo pipefail
cd "$(dirname "$0")/../.."

# A single laptop GPU cannot train many client processes at once -- they contend
# for the one GPU and for RAM (6 procs => ~345s/round + OOM). This LOCAL check
# therefore uses just 2 clients: enough to prove the networked pipeline end to end.
# The multi-hospital ACCURACY comes from the simulation; the real DISTRIBUTED demo
# is laptop + Raspberry Pi (docs/raspberry_pi_setup.md).
#
# Set INIT=<path/to/pretrained.pt> to warm-start the global model so the short run
# shows real accuracy (make one with scripts/live/pretrain_and_save.py).
ROUNDS="${ROUNDS:-8}"
MAXB="${MAXB:-25}"
DEVICE="${DEVICE:-cpu}"
CLIENTS="${CLIENTS:-0 1}"
INIT="${INIT:-}"
export DATA_ROOT="${DATA_ROOT:-$HOME/fl_data/fed_isic2019/raw}"
NCLIENTS=$(echo "$CLIENTS" | wc -w)

echo "=== LOCAL real-FL demo: server + ${NCLIENTS} clients | rounds=$ROUNDS maxb=$MAXB device=$DEVICE ==="
echo "    clients=[$CLIENTS]  data_root=$DATA_ROOT"

INIT_ARG=""; [[ -n "$INIT" ]] && INIT_ARG="--init-model $INIT"
python scripts/live/server.py --rounds "$ROUNDS" --min-clients "$NCLIENTS" \
    --host 127.0.0.1:8080 --device "$DEVICE" $INIT_ARG &
SERVER=$!
trap 'kill $SERVER 2>/dev/null' EXIT
sleep 5   # let the server bind before clients dial in

for cid in $CLIENTS; do
  python scripts/live/client.py --server 127.0.0.1:8080 --client-id "$cid" \
      --label "c${cid}" --device "$DEVICE" --max-batches "$MAXB" &
  sleep 0.5
done

wait $SERVER
echo "=== local demo done -> experiments/live/history.json ==="
echo "    make figures:  python scripts/live/plot_live.py"
