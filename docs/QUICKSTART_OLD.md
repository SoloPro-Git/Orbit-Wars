# Orbit Wars 训练命令说明

## 🚀 快速开始

### 完整训练流程（预训练 + RL）

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./train.sh
```

### 只运行强化学习

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
    --config training/config/default.yaml \
    --skip-pretrain
```

## 📋 之前的命令 vs 现在的命令

### 之前（命令行参数）

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
  --rollout-workers 24 \
  --gpus-per-worker 0.25 \
  --trainer-gpus 0.5 \
  --max-iterations 10000 \
  --games-per-rollout 8
```

### 现在（配置文件）

**推荐方式 - 使用配置文件**：
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
    --config training/config/default.yaml
```

**配置文件** (`training/config/default.yaml`)：
```yaml
ray:
  num_rollout_workers: 24
  num_gpus_per_worker: 0.25
  trainer_num_gpus: 0.5
  games_per_rollout: 64

training:
  max_iterations: 10000
```

**命令行覆盖（可选）**：
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
    --config training/config/default.yaml \
    --rollout-workers 32  # 覆盖配置文件的 24
```

## 💻 GPU 配置

### 8 GPU (1,2,3,4,5,6,7,8)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
    --config training/config/default.yaml
```

资源使用：
- 24 workers × 0.25 GPU = 6 GPU
- Trainer 0.5 GPU
- 总计: 6.5 GPU

### 4 GPU (1,2,3,4)

```bash
CUDA_VISIBLE_DEVICES=1,2,3,4 python train_ray.py \
    --config training/config/default.yaml
```

或调整配置文件中的 `num_rollout_workers: 12`

## 🔧 修改配置

编辑 `training/config/default.yaml`：

```yaml
ray:
  num_rollout_workers: 32        # 从 24 改为 32
  games_per_rollout: 8          # 保持
  num_gpus_per_worker: 0.25    # 保持
  trainer_num_gpus: 0.5         # 保持

training:
  max_iterations: 10000         # 保持
```

## 📊 专家数据预训练

如果启用（`expert_data.enabled: true`），会自动先进行预训练。

**跳过预训练**：
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 python train_ray.py \
    --config training/config/default.yaml \
    --skip-pretrain
```

## ⚠️ 注意事项

1. `train.sh` 在根目录，会自动运行预训练 + RL
2. `train_ray.py` 也在根目录（不是 training 目录）
3. 所有配置都在 `training/config/default.yaml`

---

**简单来说**：
- 之前：命令行指定所有参数
- 现在：配置文件 + 可选命令行覆盖

**最简单的使用**：
```bash
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./train.sh
```
