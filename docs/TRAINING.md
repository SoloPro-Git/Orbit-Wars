# Orbit Wars 训练快速启动

## 快速开始

### 1. 测试环境

```bash
cd training
./test_training.sh
```

预期输出：
```
✓ 所有测试通过！训练系统准备就绪。
```

### 2. 开始训练

**单卡训练：**
```bash
cd training
python core/train.py
```

**8卡分布式训练（推荐）：**
```bash
cd training
./train_8gpu.sh
```

**自定义GPU数量：**
```bash
cd training
NUM_GPUS=4 ./train_ddp.sh
```

### 3. 监控训练

训练日志会自动上传到SwanLab，可以在浏览器中查看训练进度。

Checkpoint保存在 `checkpoints/` 目录。

### 4. 提交模型

训练完成后，模型会自动保存为 `model.pt`，可直接用于Kaggle提交。

## 系统配置

- **GPU**: 8x NVIDIA H20-3e (每张143GB显存)
- **PyTorch**: 2.5.1+cu121
- **训练框架**: PPO + 自我博弈
- **模型参数**: 1,604,950

## 性能优化

当前配置针对8卡H20优化：
- 并行游戏数: 128
- 批次大小: 2048
- 每卡处理约16个并行游戏

可根据硬件调整 `training/config/default.yaml` 中的参数。

## 故障排查

**问题**: CUDA错误
```bash
# 检查GPU状态
nvidia-smi

# 重新安装PyTorch
uv pip install "torch>=2.5.0,<2.6" --index-url https://download.pytorch.org/whl/cu121
```

**问题**: 导入错误
```bash
# 确保在正确的目录
cd training
python core/train.py
```

**问题**: 显存不足
- 减少 `num_parallel_games`
- 减少 `batch_size`
- 减少模型 `d_model`
