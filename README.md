# Orbit Wars - 强化学习训练系统

基于 PPO 算法和专家数据预训练的 Orbit Wars 游戏 AI。

## 🚀 快速开始

### 1. 训练模型（完整流程）

```bash
cd training
CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./train.sh
```

这会自动运行：
- 专家数据预训练（如果数据存在）
- 模型质量评估
- 强化学习训练（PPO）

### 2. 只运行强化学习

```bash
cd training
SKIP_PRETRAIN=1 CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7 ./train.sh
```

## 📁 项目结构

```
Orbit-Wars/
├── training/               # 训练模块
│   ├── train.sh           # 主训练脚本
│   ├── train_ray.py       # RL 训练
│   ├── train_with_expert.py  # 完整流程
│   ├── expert/            # 专家数据系统
│   │   ├── ray_pretrainer.py  # 预训练器
│   │   └── ...
│   ├── core/              # 核心模块
│   └── config/            # 配置文件
├── data/
│   └── expert_demonstrations/  # 专家数据
├── docs/                  # 文档
└── scripts/               # 工具脚本
```

## 📖 详细文档

所有文档都在 `docs/` 目录：
- [TRAINING.md](docs/TRAINING.md) - 训练指南
- [项目结构](docs/PROJECT_STRUCTURE.md) - 详细的代码结构

## 🎯 核心功能

- ✅ 专家数据预训练（行为克隆 + value function）
- ✅ PPO 强化学习训练
- ✅ Ray 分布式训练（多 GPU）
- ✅ 自我对战机制
- ✅ SwanLab 实验跟踪

## ⚙️ 配置

所有配置都在 `training/config/default.yaml`：

```yaml
expert_data:
  enabled: true              # 启用预训练
  data_dir: data/expert_demonstrations

ray:
  num_rollout_workers: 24    # Worker 数量
  num_gpus_per_worker: 0.25  # 每个 Worker GPU
  games_per_rollout: 64      # 每次游戏数

training:
  max_iterations: 10000      # 最大迭代
```

## 🛠️ 常用命令

```bash
# 生成专家数据
./generate_expert_data.sh

# 完整训练
cd training && ./train.sh

# 跳过预训练
cd training && SKIP_PRETRAIN=1 ./train.sh
```

## 🤖 Codex / 本地模拟器默认

本地测试、数据生成和训练 smoke check 默认优先使用
`training2.make_fast_orbit_wars(..., use_numba=True)`，不要直接走官方
`kaggle_environments.make("orbit_wars")`。只有做官方一致性校验或排查
simulator drift 时再使用 Kaggle 官方环境。

项目级约定见 [AGENTS.md](AGENTS.md)。

## ⚠️ 注意事项

1. 所有训练脚本都在 `training/` 目录下
2. 从根目录运行时使用 `./training/train.sh`
3. 从 training 目录运行时使用 `./train.sh`

## 📚 更多信息

查看 `docs/` 目录了解详细信息。
