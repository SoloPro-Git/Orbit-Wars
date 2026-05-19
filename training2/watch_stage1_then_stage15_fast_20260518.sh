#!/usr/bin/env bash
set -euo pipefail

cd /data2/solo/Orbit-Wars

export PYTHONUNBUFFERED=1
export RAY_ADDRESS=10.0.104.198:6380
export SWANLAB_NO_INTERACTIVE=1
export SWANLAB_DISABLE_INTERACTIVE=1

STAGE1_CFG=training2/config/stage1_regular_refresh_20260518.yaml
STAGE1_TRAIN_LOG=training2/stage1_regular_current_20260518_train.log
STAGE1_CKPT=training2/checkpoints/stage1_regular_current_20260518/latest.pt

STAGE15_CFG=training2/config/stage15_fast_after_stage1_20260518.yaml
STAGE15_LOG=training2/stage15_fast_after_stage1_20260518.log

echo "[watch-stage15] waiting for stage1 training to start: ${STAGE1_TRAIN_LOG}"
while [[ ! -f "${STAGE1_TRAIN_LOG}" ]]; do
  date +"[watch-stage15] %F %T still waiting for stage1 train log"
  sleep 60
done

echo "[watch-stage15] stage1 training log exists; waiting for train_stage1_ray to finish"
while pgrep -af "[t]raining2.train_stage1_ray.*stage1_regular_refresh_20260518.yaml" >/dev/null; do
  date +"[watch-stage15] %F %T stage1 still running"
  sleep 300
done

if [[ ! -s "${STAGE1_CKPT}" ]]; then
  echo "[watch-stage15] missing checkpoint: ${STAGE1_CKPT}" >&2
  exit 1
fi

echo "[watch-stage15] launching stage15 fast env from ${STAGE1_CKPT}"
echo "[watch-stage15] config: ${STAGE15_CFG}"
echo "[watch-stage15] log:    ${STAGE15_LOG}"

.venv/bin/python -m training2.train_stage15_ray \
  --config "${STAGE15_CFG}" \
  --resume "${STAGE1_CKPT}" \
  --no-swanlab \
  2>&1 | tee "${STAGE15_LOG}"
