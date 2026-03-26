#!/usr/bin/env bash
set -e

ROOT="data/fed_isic2019/raw"

if [ ! -d "$ROOT" ]; then
  echo "[ERROR] Dataset raw directory not found: $ROOT"
  exit 1
fi

if command -v tree >/dev/null 2>&1; then
  tree -L 3 "$ROOT"
else
  echo "[INFO] tree not installed. Showing find output instead."
  find "$ROOT" -maxdepth 3
fi
