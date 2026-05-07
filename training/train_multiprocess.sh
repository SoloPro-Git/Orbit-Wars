#!/bin/bash
# 多GPU多进程训练启动脚本
#
# 使用方法:
#   bash train_multiprocess.sh                    # 默认: 4卡，每卡4进程
#   bash train_multiprocess.sh 4 6                # 4卡，每卡6进程
#   bash train_multiprocess.sh 2 4                # 2卡，每卡4进程
#   bash train_multiprocess.sh 1 8                # 1卡，8进程
#
# 或使用自定义配置:
#   bash train_multiprocess.sh "0:2,1:6,2:4"      # GPU0:2进程, GPU1:6进程, GPU2:4进程

set -e

# 确保在正确的目录
cd "$(dirname "$0")"

# 激活虚拟环境
source ../.venv/bin/activate

# 参数解析
if [ $# -eq 0 ]; then
    # 默认配置: 4卡，每卡4进程
    GPU_CONFIG="0:4,1:4,2:4,3:4"
elif [ $# -eq 1 ]; then
    # 单个参数，视为自定义配置
    GPU_CONFIG="$1"
elif [ $# -eq 2 ]; then
    # 两个参数: <gpu数量> <每gpu进程数>
    NUM_GPUS=$1
    PROCESSES_PER_GPU=$2

    # 构建配置字符串
    GPU_CONFIG=""
    for ((i=0; i<NUM_GPUS; i++)); do
        if [ -n "$GPU_CONFIG" ]; then
            GPU_CONFIG="$GPU_CONFIG,$i:$PROCESSES_PER_GPU"
        else
            GPU_CONFIG="$i:$PROCESSES_PER_GPU"
        fi
    done
else
    echo "使用方法:"
    echo "  bash train_multiprocess.sh                    # 默认: 4卡，每卡4进程"
    echo "  bash train_multiprocess.sh 4 6                # 4卡，每卡6进程"
    echo "  bash train_multiprocess.sh 2 4                # 2卡，每卡4进程"
    echo "  bash train_multiprocess.sh 1 8                # 1卡，8进程"
    echo "  bash train_multiprocess.sh \"0:2,1:6,2:4\"    # 自定义配置"
    exit 1
fi

# 显示配置信息
echo "=========================================="
echo "  Orbit Wars 多进程训练"
echo "=========================================="
echo "GPU配置: $GPU_CONFIG"
echo "=========================================="
echo ""

# 启动训练
python train_flexible.py \
    --gpu-config "$GPU_CONFIG" \
    --config config/default.yaml \
    --log-dir logs/multiprocess \
    --kill-existing
