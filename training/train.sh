#!/bin/bash
# Orbit Wars Ray 分布式训练启动脚本
#
# 环境变量配置（可选）:
#   export GPUS="1,2,3,4,5,6,7"           # 使用的 GPU
#   export ROLLOUT_WORKERS=14            # Rollout worker 数量
#   export GAMES_PER_ROLLOUT=4           # 每个 rollout 的游戏数
#   export MAX_ITERATIONS=10000          # 最大迭代次数
#
# 使用方式:
#   bash train.sh                        # 使用默认配置
#   GPUS="1,2" bash train.sh             # 使用 GPU 1,2
#   ROLLOUT_WORKERS=8 bash train.sh      # 使用 8 个 workers

set -e

# 默认配置
GPUS=${GPUS:-"1,2,3,4,5,6,7"}
ROLLOUT_WORKERS=${ROLLOUT_WORKERS:-24}
GPUS_PER_WORKER=${GPUS_PER_WORKER:-0.25}
TRAINER_GPUS=${TRAINER_GPUS:-0.5}
GAMES_PER_ROLLOUT=${GAMES_PER_ROLLOUT:-8}
MAX_ITERATIONS=${MAX_ITERATIONS:-10000}

echo "============================================================"
echo "  Orbit Wars - Ray 分布式训练"
echo "============================================================"
echo "可见 GPU: $GPUS"
echo "Rollout Workers: $ROLLOUT_WORKERS"
echo "GPU per Worker: $GPUS_PER_WORKER"
echo "Trainer GPU: $TRAINER_GPUS"
echo "Games per Rollout: $GAMES_PER_ROLLOUT"
echo "Max Iterations: $MAX_ITERATIONS"
echo "总并行游戏: $((ROLLOUT_WORKERS * GAMES_PER_ROLLOUT)) 局"
echo "============================================================"
echo ""

# 清理旧进程
echo "[清理] 停止旧进程..."
pkill -9 -f "train_ray.py" 2>/dev/null || true
rm -rf /data2/solo/Orbit-Wars/training/.ray_temp 2>/dev/null || true
sleep 2

# 启动训练
echo "[启动] 开始训练..."
CUDA_VISIBLE_DEVICES=$GPUS python train_ray.py \
  --rollout-workers $ROLLOUT_WORKERS \
  --gpus-per-worker $GPUS_PER_WORKER \
  --trainer-gpus $TRAINER_GPUS \
  --games-per-rollout $GAMES_PER_ROLLOUT \
  --max-iterations $MAX_ITERATIONS \
  --no-ray-redis

echo ""
echo "✅ 训练完成"
