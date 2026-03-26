#!/usr/bin/env bash
set -e
echo "Python: $(python --version)"
echo "Pip: $(pip --version)"
python scripts/setup/check_gpu.py
