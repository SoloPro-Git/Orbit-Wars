#!/bin/bash
# Orbit Wars 完整训练流程（预训练 + RL）
#
# 环境变量配置（可选）:
#   export GPUS="1,2,3,4,5,6,7"           # 使用的 GPU
#   export SKIP_PRETRAIN=1               # 跳过预训练
#
# 使用方式:
#   bash train.sh                        # 完整流程（预训练 + RL）
#   SKIP_PRETRAIN=1 bash train.sh        # 跳过预训练，只运行 RL

set -e

# 默认配置
GPUS=${GPUS:-"0,1,2,3,4,5,6,7"}
CONFIG_FILE=${CONFIG_FILE:-"config/default.yaml"}
SKIP_PRETRAIN=${SKIP_PRETRAIN:-0}

echo "============================================================"
echo "  Orbit Wars 训练启动器（自动化）"
echo "============================================================"
echo "可见 GPU: $GPUS"
echo "配置文件: $CONFIG_FILE"
echo "跳过预训练: $SKIP_PRETRAIN"
echo "============================================================"
echo ""

# 激活虚拟环境
cd "$(dirname "$0")/../"
source .venv/bin/activate
cd training

echo "自动化训练流程："
echo "  1. 检查专家数据"
if [ "$SKIP_PRETRAIN" = "0" ]; then
    echo "  2. 专家数据预训练（Ray 多GPU 分布式）"
    echo "  3. 自动评估质量"
fi
echo "  $((3 + SKIP_PRETRAIN)). Ray 分布式强化学习（多GPU PPO）"
echo ""

# 检查专家数据
EXPERT_DATA_DIR="../data/expert_demonstrations"
if [ ! -d "$EXPERT_DATA_DIR" ] || [ -z "$(ls -A $EXPERT_DATA_DIR 2>/dev/null)" ]; then
    echo "⚠️  未找到专家数据: $EXPERT_DATA_DIR"
    echo "跳过预训练，直接运行强化学习..."
    SKIP_PRETRAIN=1
fi

# 清理旧进程
echo "[清理] 停止旧进程..."
pkill -9 -f "train_with_expert.py" 2>/dev/null || true
pkill -9 -f "train_ray.py" 2>/dev/null || true
rm -rf .ray_temp 2>/dev/null || true
sleep 2

echo "✅ 准备就绪，开始训练..."
echo ""

# 启动完整训练流程
if [ "$SKIP_PRETRAIN" = "1" ]; then
    echo "跳过预训练，直接运行强化学习..."
    CUDA_VISIBLE_DEVICES=$GPUS python train_ray.py \
        --config "$CONFIG_FILE" \
        --skip-pretrain
else
    echo "运行完整流程（预训练 + RL）..."
    CUDA_VISIBLE_DEVICES=$GPUS python train_with_expert.py \
        --config "$CONFIG_FILE"
fi

echo ""
echo "✅ 训练完成"
