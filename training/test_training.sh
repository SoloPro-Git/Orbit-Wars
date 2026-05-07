#!/bin/bash
# 测试训练系统初始化

set -e

echo "=========================================="
echo "  测试训练系统初始化"
echo "=========================================="

# 激活虚拟环境
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_ROOT/.venv/bin/python"

# 测试1: 检查依赖包
echo ""
echo "[测试1] 检查依赖包..."
$VENV_PYTHON -c "
import torch
import numpy
import kaggle_environments
import swanlab
import yaml
print('✓ 所有依赖包已安装')
print(f'  PyTorch: {torch.__version__}')
print(f'  CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'  GPU数量: {torch.cuda.device_count()}')
"

# 测试2: 检查配置文件
echo ""
echo "[测试2] 检查配置文件..."
cd "$SCRIPT_DIR"
if [ -f "config/default.yaml" ]; then
    echo "✓ 配置文件存在: config/default.yaml"
else
    echo "✗ 配置文件不存在"
    exit 1
fi

# 测试3: 测试模型初始化
echo ""
echo "[测试3] 测试模型初始化..."
$VENV_PYTHON -c "
import torch
from core.config import AppConfig
from core.model import OrbitWarsModel

config = AppConfig.from_yaml('config/default.yaml')
device = 'cuda' if torch.cuda.is_available() else 'cpu'

model = OrbitWarsModel(
    config.model,
    n_planets=40,
    max_players=4,
).to(device)

params = sum(p.numel() for p in model.parameters())
print(f'✓ 模型初始化成功')
print(f'  设备: {device}')
print(f'  参数量: {params:,}')
"

# 测试4: 环境验证
echo ""
echo "[测试4] 环境配置验证..."
$VENV_PYTHON -c "
from core.validation import validate_training_environment
validate_training_environment()
"

# 测试5: 测试DDP环境（可选）
echo ""
echo "[测试5] 测试DDP环境（可选）..."
if [ "$1" == "--ddp" ]; then
    torchrun --nproc_per_node=2 --master_port=29501 test_ddp.py
else
    echo "  跳过DDP测试（使用 --ddp 参数启用）"
fi

echo ""
echo "=========================================="
echo "✓ 所有测试通过！训练系统准备就绪。"
echo "=========================================="
echo ""
echo "开始训练："
echo "  单卡: python core/train.py"
echo "  8卡: ./train_8gpu.sh"
echo ""
