# Orbit Wars 训练文档

本目录包含 Orbit Wars 强化学习训练脚本，使用 Ray 分布式框架进行 PPO 训练。

## 📁 文件说明

### 核心文件

- **`train_ray.py`** - Ray 分布式训练主脚本
  - 使用 Ray Actors 分布式执行 rollout 和 PPO 更新
  - 支持多 GPU 并行训练
  - 集成 SwanLab 实验追踪

- **`train.sh`** - 训练启动脚本（推荐使用）
  - 环境变量配置
  - 自动清理旧进程
  - 灵活的参数设置

### 配置文件

- **`config/default.yaml`** - 默认训练配置
  - 模型超参数
  - PPO 训练参数
  - Ray 分布式配置

## 🚀 快速开始

### 方式 1：使用启动脚本（推荐）

```bash
# 使用默认配置（7卡，14 workers）
cd training
bash train.sh

# 自定义 GPU
GPUS="1,2,3" bash train.sh

# 自定义 worker 数量
ROLLOUT_WORKERS=8 bash train.sh

# 完全自定义
GPUS="1,2" ROLLOUT_WORKERS=4 GAMES_PER_ROLLOUT=8 bash train.sh
```

### 方式 2：直接运行 Python 脚本

```bash
cd training

# 基础配置
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
  --rollout-workers 14 \
  --gpus-per-worker 0.25 \
  --trainer-gpus 0.5 \
  --games-per-rollout 4 \
  --max-iterations 10000 \
  --no-ray-redis
```

## ⚙️ 配置参数说明

### 环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `GPUS` | `"1,2,3,4,5,6,7"` | 使用的 GPU 列表 |
| `ROLLOUT_WORKERS` | `14` | Rollout actor 数量 |
| `GPUS_PER_WORKER` | `0.25` | 每个 worker 的 GPU 数量 |
| `TRAINER_GPUS` | `0.5` | Trainer 的 GPU 数量 |
| `GAMES_PER_ROLLOUT` | `4` | 每个 rollout 运行的游戏数 |
| `MAX_ITERATIONS` | `10000` | 最大训练迭代次数 |

### 命令行参数

```bash
python train_ray.py \
  --rollout-workers N      # Rollout worker 数量
  --gpus-per-worker X       # 每个 worker 的 GPU 数量（小数表示共享）
  --trainer-gpus Y          # Trainer 的 GPU 数量
  --games-per-rollout N     # 每次 rollout 的游戏数
  --max-iterations N        # 最大迭代次数
  --no-ray-redis            # 单机模式（不启动 Redis）
  --config PATH             # 配置文件路径
```

## 📊 推荐配置

### 快速调试（单卡）

```bash
GPUS="1" ROLLOUT_WORKERS=2 GAMES_PER_ROLLOUT=2 bash train.sh
```

**预期性能**：
- 迭代时间：30-35 秒
- 总并行游戏：4 局
- GPU 利用率：中等

### 平衡配置（3-4 卡）

```bash
GPUS="1,2,3" ROLLOUT_WORKERS=6 GAMES_PER_ROLLOUT=4 bash train.sh
```

**预期性能**：
- 迭代时间：35-45 秒
- 总并行游戏：24 局
- GPU 利用率：高

### 高性能配置（7 卡）

```bash
# 使用默认配置即可
bash train.sh
```

**预期性能**：
- 迭代时间：40-60 秒
- 总并行游戏：56 局
- GPU 利用率：很高

## 🔍 监控训练

### GPU 监控

```bash
# 实时监控 GPU 利用率
watch -n 1 nvidia-smi

# 查看 GPU 内存使用
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### 日志查看

训练过程中会看到以下进度信息：

```
[1/10000] Rollout (4人局): Rollout: 2000条, 29.0s | PPO更新...
```

- **Rollout**: 收集经验数据的时间
- **PPO更新**: 策略更新的时间
- **Reward**: 平均奖励（越高越好）

### SwanLab 监控

训练会自动上传到 SwanLab：
- 项目地址：https://swanlab.cn/@Solo/orbit-wars
- 实时查看训练曲线、损失函数、奖励等指标

## 🛠️ 故障排除

### 问题 1：训练卡住不动

**症状**：进度条停在 "Rollout 启动中..."

**可能原因**：
- 第一次迭代需要初始化 actors，需要 5-10 分钟
- 某个 worker 执行很慢

**解决方案**：
- 减少 worker 数量：`ROLLOUT_WORKERS=8 bash train.sh`
- 耐心等待第一次迭代完成
- 查看日志确认进度

### 问题 2：GPU 利用率 0%

**症状**：`nvidia-smi` 显示 GPU 利用率为 0%

**原因**：Rollout 在 CPU 上运行游戏模拟，GPU 只用于模型推理

**正常现象**：GPU 利用率 0-30% 是正常的

### 问题 3：磁盘空间不足

**症状**：Ray 报错 `/tmp` 空间不足

**解决方案**：已自动配置到 `/data2` 分区，仍有问题请手动清理：

```bash
rm -rf /data2/solo/Orbit-Wars/training/.ray_temp
```

## 📝 训练输出

### Checkpoint

模型会定期保存到 `checkpoints/` 目录：

```
checkpoints/
├── model_iter_100.pt
├── model_iter_200.pt
└── ...
```

### 日志文件

- SwanLab 日志：自动上传到云端
- Ray 日志：`.ray_temp/session_*/logs/`
- 训练日志：标准输出

## 🔗 相关文件

- **`core/model.py`** - 神经网络模型定义
- **`core/ppo.py`** - PPO 算法实现
- **`core/rollout.py`** - Rollout 数据收集
- **`core/config.py`** - 配置管理
