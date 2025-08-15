#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="${1:-./bts_on_time_data}"
SAMPLE_REL="eda/sample_100k_week2_features.parquet"
echo "[RUN] week3_baselines.py --root $ROOT_DIR --sample $SAMPLE_REL"
python week3_baselines.py --root "$ROOT_DIR" --sample "$SAMPLE_REL"
echo "[DONE] outputs under $ROOT_DIR/eda"
