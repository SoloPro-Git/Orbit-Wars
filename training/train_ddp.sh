#!/bin/bash
# 多卡分布式训练启动脚本（可指定GPU数量）

set -e

# 设置SwanLab API key
export SWANLAB_API_KEY="2tljFkPrQC4udTXKKfuwC"

# 默认参数
NUM_GPUS=${NUM_GPUS:-8}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_PATH=${1:-"config/default.yaml"}

# 激活虚拟环境
if [ -d "../.venv" ]; then
    source ../.venv/bin/activate
    echo "✓ 虚拟环境已激活"
else
    echo "✗ 错误：未找到 .venv 目录"
    exit 1
fi

echo "=========================================="
echo "  Orbit Wars 多卡分布式训练"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "配置文件: $CONFIG_PATH"
echo "Master端口: $MASTER_PORT"
echo "=========================================="

# 检查GPU可用性
if ! command -v nvidia-smi &> /dev/null; then
    echo "✗ 错误：未检测到 nvidia-smi，请确认已安装 NVIDIA 驱动"
    exit 1
fi

# 检查可用的GPU数量
AVAILABLE_GPUS=$(nvidia-smi --list-gpus | wc -l)
if [ $AVAILABLE_GPUS -lt $NUM_GPUS ]; then
    echo "✗ 警告：请求 $NUM_GPUS 张GPU，但只检测到 $AVAILABLE_GPUS 张"
    echo "  将使用 $AVAILABLE_GPUS 张GPU进行训练"
    NUM_GPUS=$AVAILABLE_GPUS
fi

echo "✓ 检测到 $AVAILABLE_GPUS 张GPU，使用前 $NUM_GPUS 张"
echo ""

# 使用 torchrun 启动分布式训练
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_ddp.py \
    "$CONFIG_PATH"

echo ""
echo "=========================================="
echo "训练完成！"
echo "=========================================="
