#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="/data/home/lzy/cbd/code/ARiADNE_MAMBA"
PYTHON_BIN="/data/home/lzy/miniconda3/envs/ariadne_mamba/bin/python"

cd "$PROJECT_DIR"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

exec "$PYTHON_BIN" -u driver.py
