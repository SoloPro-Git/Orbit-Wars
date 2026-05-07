# Orbit Wars 训练系统

## 环境配置

### 安装依赖

```bash
# 1. 创建虚拟环境
python3.12 -m venv .venv
source .venv/bin/activate

# 2. 安装PyTorch (CUDA 12.1版本)
uv pip install "torch>=2.5.0,<2.6" --index-url https://download.pytorch.org/whl/cu121

# 3. 安装其他依赖
uv pip install kaggle-environments numpy pyyaml swanlab
```

### 验证环境

```bash
cd training
torchrun --nproc_per_node=8 --master_port=29500 test_ddp.py
```

应该看到类似输出：
```
✓ DDP环境配置正确！8卡训练准备就绪。
```

## 训练方式

### 单卡训练

```bash
cd training
source ../.venv/bin/activate
python core/train.py
```

### 8卡分布式训练（推荐）

**方式1：使用固定8卡脚本**
```bash
cd training
./train_8gpu.sh
```

**方式2：使用可配置GPU数量脚本**
```bash
cd training
# 使用默认8卡
./train_ddp.sh

# 或指定GPU数量
NUM_GPUS=4 ./train_ddp.sh

# 或指定配置文件
./train_ddp.sh config/default.yaml
```

**方式3：直接使用torchrun**
```bash
cd training
torchrun --nproc_per_node=8 --master_port=29500 train_ddp.py
```

## 训练配置

配置文件位于 `training/config/default.yaml`，主要参数：

- **模型配置**：Transformer架构参数
- **训练配置**：学习率、批次大小、并行游戏数
- **自我博弈配置**：对手池大小、采样策略
- **奖励配置**：各类奖励权重
- **环境配置**：游戏参数（步数、速度、地图大小等）

## 训练输出

- **Checkpoint**: 保存在 `checkpoints/` 目录
- **日志**: 使用SwanLab记录训练指标
- **模型**: 最终模型保存为 `model.pt`（用于Kaggle提交）

## 性能优化建议

1. **GPU利用率**：调整 `num_parallel_games` 参数以充分利用8卡
2. **批次大小**：根据显存大小调整 `batch_size`
3. **保存频率**：调整 `save_interval` 以平衡训练速度和checkpoint频率
4. **对手池**：调整 `pool_size` 以平衡训练稳定性和收敛速度

## 故障排查

### CUDA错误
```bash
# 检查驱动版本
nvidia-smi

# 检查PyTorch CUDA版本
python -c "import torch; print(torch.version.cuda)"
```

### 显存不足
- 减少 `num_parallel_games`
- 减少 `batch_size`
- 减少模型 `d_model` 参数

### 多卡通信问题
- 检查NCCL环境变量
- 确保防火墙不阻止GPU间通信
- 尝试更换 `MASTER_PORT`
