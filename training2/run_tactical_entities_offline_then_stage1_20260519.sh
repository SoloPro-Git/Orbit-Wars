#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

unset RAY_ADDRESS

CONFIG="training2/config/stage1_tactical_entities_20260519.yaml"
GEN_LOG="training2/stage1_tactical_entities_20260519_generate.log"
TRAIN_LOG="training2/stage1_tactical_entities_20260519_train.log"

.venv/bin/python -m training2.generate_stage1_dataset_ray \
  --config "$CONFIG" 2>&1 | tee "$GEN_LOG"

.venv/bin/python -m training2.train_stage1_ray \
  --config "$CONFIG" \
  --allow-no-swanlab 2>&1 | tee "$TRAIN_LOG"
