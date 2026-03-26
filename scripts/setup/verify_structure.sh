#!/usr/bin/env bash
set -e
for d in src configs scripts notebooks results docs planning tests; do
  [ -d "$d" ] && echo "[OK] $d" || echo "[MISSING] $d"
done
