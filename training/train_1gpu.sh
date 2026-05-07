#!/bin/bash
# 单卡训练启动脚本

set -e

# 激活虚拟环境
source ../.venv/bin/activate

# 设置SwanLab API key
export SWANLAB_API_KEY="2tljFkPrQC4udTXKKfuwC"

# 可选：自定义配置文件路径
CONFIG_PATH=${1:-"config/default.yaml"}

echo "=========================================="
echo "  Orbit Wars 单卡训练"
echo "=========================================="
echo "GPU数量: 1"
echo "配置文件: $CONFIG_PATH"
echo "=========================================="

# 单卡训练不需要 torchrun，直接运行即可
# train_ddp.py 会自动检测到没有分布式环境变量，使用单卡模式
python train_ddp.py "$CONFIG_PATH"

echo "训练完成！"
