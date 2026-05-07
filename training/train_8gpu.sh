#!/bin/bash
# 8卡分布式训练启动脚本

set -e

# 激活虚拟环境
source ../.venv/bin/activate

# 设置SwanLab API key
export SWANLAB_API_KEY="2tljFkPrQC4udTXKKfuwC"

# 配置参数
NUM_GPUS=8
MASTER_PORT=29500

# 可选：自定义配置文件路径
CONFIG_PATH=${1:-"config/default.yaml"}

echo "=========================================="
echo "  Orbit Wars 8卡分布式训练"
echo "=========================================="
echo "GPU数量: $NUM_GPUS"
echo "配置文件: $CONFIG_PATH"
echo "Master端口: $MASTER_PORT"
echo "=========================================="

# 使用 torchrun 启动分布式训练
# torchrun 会自动设置以下环境变量：
# - RANK: 进程序号
# - WORLD_SIZE: 总进程数
# - LOCAL_RANK: 本地进程序号
torchrun \
    --nproc_per_node=$NUM_GPUS \
    --master_port=$MASTER_PORT \
    train_ddp.py \
    "$CONFIG_PATH"

echo "训练完成！"
