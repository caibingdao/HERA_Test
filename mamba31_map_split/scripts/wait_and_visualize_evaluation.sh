#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 SYSTEMD_SERVICE RESULTS_DIR VISUAL_DIR" >&2
  exit 2
fi

SERVICE_NAME="$1"
RESULTS_DIR="$2"
VISUAL_DIR="$3"
PROJECT_DIR="/users/cbd/code/ARiADNE_MAMBA_Test"
PYTHON_BIN="/users/cbd/miniconda3/envs/ariadne_mamba/bin/python"

cd "$PROJECT_DIR"

while systemctl --user is-active --quiet "$SERVICE_NAME"; do
  echo "$(date --iso-8601=seconds) waiting for $SERVICE_NAME"
  sleep 30
done

for required_file in episodes.csv coverage_curve.csv evaluation_metadata.json; do
  if [[ ! -f "$RESULTS_DIR/$required_file" ]]; then
    echo "ERROR: evaluation ended without $RESULTS_DIR/$required_file" >&2
    exit 1
  fi
done

if [[ -e "$VISUAL_DIR" ]]; then
  echo "ERROR: visual output already exists: $VISUAL_DIR" >&2
  exit 1
fi

exec "$PYTHON_BIN" scripts/create_visual_report.py \
  --results-dir "$RESULTS_DIR" \
  --output-dir "$VISUAL_DIR" \
  --dpi 180
