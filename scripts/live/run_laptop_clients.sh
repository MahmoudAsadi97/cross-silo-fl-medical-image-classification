#!/usr/bin/env bash
# For the REAL Raspberry Pi run: launch the laptop's hospitals (clients 0-4) that
# connect to the server; the Pi runs client 5 separately (docs/raspberry_pi_setup.md).
# Order: (1) start the server, (2) run this, (3) start the Pi client. The server
# should use --min-clients 6 so it waits for all five laptop clients + the Pi.
#
#   DATA_ROOT=$HOME/fl_data/fed_isic2019/raw bash scripts/live/run_laptop_clients.sh
#
# Knobs: SERVER (127.0.0.1:8080), MAXB (30), DEVICE (cuda), CLIENTS ("0 1 2 3 4").
set -uo pipefail
cd "$(dirname "$0")/../.."

SERVER="${SERVER:-127.0.0.1:8080}"
MAXB="${MAXB:-30}"
DEVICE="${DEVICE:-cuda}"
CLIENTS="${CLIENTS:-0 1 2 3 4}"
export DATA_ROOT="${DATA_ROOT:-$HOME/fl_data/fed_isic2019/raw}"

echo "laptop hospitals [$CLIENTS] -> $SERVER  (device=$DEVICE, maxb=$MAXB)"
for cid in $CLIENTS; do
  python scripts/live/client.py --server "$SERVER" --client-id "$cid" \
      --label "laptop-c${cid}" --device "$DEVICE" --max-batches "$MAXB" &
  sleep 0.5
done
wait
echo "laptop clients finished"
