#!/usr/bin/env bash
set -euo pipefail

cd /data2/solo/Orbit-Wars

DELAY_SECONDS="${1:-3000}"
TARGET_TMUX="${2:-\$0:stage1-regular-refresh}"

STAGE1_CFG=training2/config/stage1_regular_refresh_20260518.yaml
STAGE1_CKPT=training2/checkpoints/stage1_regular_current_20260518/latest.pt
STAGE15_CFG=training2/config/stage15_fast_after_stage1_20260518.yaml
STAGE15_LOG=training2/stage15_fast_after_stage1_20260518.log

echo "[defer-stage15] sleeping ${DELAY_SECONDS}s before checking stage1"
sleep "${DELAY_SECONDS}"

echo "[defer-stage15] waiting for stage1 generation/training to finish"
while pgrep -af "[t]raining2.generate_stage1_dataset_ray.*stage1_regular_refresh_20260518.yaml|[t]raining2.train_stage1_ray.*stage1_regular_refresh_20260518.yaml" >/dev/null; do
  date +"[defer-stage15] %F %T stage1 pipeline still running"
  sleep 120
done

if [[ ! -s "${STAGE1_CKPT}" ]]; then
  echo "[defer-stage15] missing checkpoint after stage1: ${STAGE1_CKPT}" >&2
  exit 1
fi

if pgrep -af "[t]raining2.train_stage15_ray.*stage15_fast_after_stage1_20260518.yaml" >/dev/null; then
  echo "[defer-stage15] stage15 already running, skip"
  exit 0
fi

CMD="cd /data2/solo/Orbit-Wars; export PYTHONUNBUFFERED=1 RAY_ADDRESS=10.0.104.198:6380 SWANLAB_NO_INTERACTIVE=1 SWANLAB_DISABLE_INTERACTIVE=1; .venv/bin/python -m training2.train_stage15_ray --config ${STAGE15_CFG} --resume ${STAGE1_CKPT} --no-swanlab 2>&1 | tee ${STAGE15_LOG}"

echo "[defer-stage15] sending stage15 command to tmux target ${TARGET_TMUX}"
tmux send-keys -t "${TARGET_TMUX}" "${CMD}" C-m
