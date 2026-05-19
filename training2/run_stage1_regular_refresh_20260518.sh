#!/usr/bin/env bash
set -euo pipefail

cd /data2/solo/Orbit-Wars

export PYTHONUNBUFFERED=1
export RAY_ADDRESS=10.0.104.198:6380
export SWANLAB_NO_INTERACTIVE=1
export SWANLAB_DISABLE_INTERACTIVE=1

CFG=training2/config/stage1_regular_refresh_20260518.yaml
DATA=data/training2/stage1_regular_current_20260518
GENLOG=training2/stage1_regular_current_20260518_generate.log
TRAINLOG=training2/stage1_regular_current_20260518_train.log

mkdir -p training2/checkpoints/stage1_regular_current_20260518

echo "[stage1-refresh] config: ${CFG}"
echo "[stage1-refresh] data:   ${DATA}"
echo "[stage1-refresh] ray:    ${RAY_ADDRESS}"
echo "[stage1-refresh] generate log: ${GENLOG}"
echo "[stage1-refresh] train log:    ${TRAINLOG}"

.venv/bin/python -m training2.generate_stage1_dataset_ray \
  --config "${CFG}" \
  --out "${DATA}" \
  2>&1 | tee "${GENLOG}"

echo "[stage1-refresh] dataset generation done, starting stage1 training"

.venv/bin/python -m training2.train_stage1_ray \
  --config "${CFG}" \
  --allow-no-swanlab \
  2>&1 | tee "${TRAINLOG}"
