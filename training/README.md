# Orbit Wars 训练系统

## 快速开始

```bash
cd training

# 默认配置: 4 GPU，每个 GPU 4 个进程（16个总进程）
bash train_multiprocess.sh

# 4 GPU，每个 GPU 6 个进程（24个总进程，高性能）
bash train_multiprocess.sh 4 6

# 2 GPU，每个 GPU 4 个进程（8个总进程）
bash train_multiprocess.sh 2 4

# 1 GPU，8 个进程（8个总进程，单GPU调试）
bash train_multiprocess.sh 1 8

# 自定义配置: GPU 0 运行 2 进程，GPU 1 运行 6 进程
bash train_multiprocess.sh "0:2,1:6"
```

## 使用说明

`train_multiprocess.sh` 参数说明：
- **无参数**: 默认 4 卡，每卡 4 进程
- **1个参数**: 自定义配置（如 `"0:2,1:6,2:4"`）
- **2个参数**: GPU数量 和 每GPU进程数（如 `4 6`）

# 4 GPU DDP
bash train_4gpu.sh
```

## 监控命令

```bash
# GPU 使用情况
watch -n 1 nvidia-smi

# CPU 使用情况
htop

# 查看所有训练日志
tail -f logs/multiprocess/*.log

# 查看特定进程日志
tail -f logs/multiprocess/gpu0_worker0.log
```

## 性能对比

| 配置 | GPU 数量 | 每GPU进程数 | 总进程数 | 预估CPU利用率 |
|------|---------|-----------|---------|-------------|
| `train_multiprocess.sh 1 8` | 1 | 8 | 8 | ~6% |
| `train_multiprocess.sh 2 4` | 2 | 4 | 8 | ~12% |
| `train_multiprocess.sh 4 4` | 4 | 4 | 16 | ~25% |
| `train_multiprocess.sh 4 6` | 4 | 6 | 24 | ~37% |

## 配置文件

所有配置都在 `config/default.yaml` 中：

```yaml
training:
  batch_size: 8192          # 每个进程的批次大小
  num_parallel_games: 256   # 每个进程的并行游戏数
  num_feature_workers: 4    # 每个进程的特征提取进程数
```

**推荐配置**：
- **总进程数 ≤ 16**: `num_feature_workers: 4`
- **总进程数 16-32**: `num_feature_workers: 2`
- **总进程数 > 32**: `num_feature_workers: 1`

## 硬件建议

- **128 核 CPU**: 推荐使用 `4 6` 配置（24 个总进程）
- **64 核 CPU**: 推荐使用 `4 4` 配置（16 个总进程）
- **32 核 CPU**: 推荐使用 `2 4` 配置（8 个总进程）

## 停止训练

按 `Ctrl+C` 会优雅地停止所有进程。

## 故障排除

### GPU 内存不足
**解决方案**: 减少每个 GPU 的进程数
```bash
# 从 6 减少到 4
bash train_multiprocess.sh 4 4
```

### CPU 利用率低
**解决方案**: 增加总进程数
```bash
# 从 4 4 增加到 4 6
bash train_multiprocess.sh 4 6
```

### 进程意外退出
**解决方案**: 查看日志文件
```bash
tail -n 50 logs/multiprocess/gpu0_worker0.log
```
