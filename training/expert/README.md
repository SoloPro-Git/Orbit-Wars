# Orbit Wars 专家策略系统

这个模块实现了从 Kaggle Orbit Wars 竞赛中提取的高分策略，用于生成训练数据。

## 功能特性

- **KaggleExpertAgent**: 完整的高分策略实现，包含：
  - 太阳避障算法
  - 星球位置预测（考虑公转）
  - 彗星位置预测
  - 战斗模拟和防御计算
  - 智能 ROI 目标选择

- **ExpertDataGenerator**: 自动生成训练数据
  - 运行专家策略对战
  - 记录 (observation, action) 对
  - 支持大规模数据生成

- **ExpertDataset**: 数据管理和加载
  - 批量加载和缓存
  - 训练/验证集分割
  - 统计信息展示

## 目录结构

```
training/expert/
├── __init__.py              # 模块导出
├── base_agent.py            # ExpertAgent 基类
├── kaggle_expert.py         # Kaggle 高分策略实现
├── data_generator.py        # 数据生成器
└── README.md                # 本文档
```

## 快速开始

### 1. 生成专家演示数据

```bash
# 生成 100 局对局（约 200,000 样本）
python generate_expert_data.py --num_episodes 100

# 生成 500 局对局，每 50 局保存一个文件
python generate_expert_data.py --num_episodes 500 --episodes_per_file 50

# 自定义保存位置
python generate_expert_data.py --num_episodes 200 --save_dir data/my_expert_data
```

### 2. 在 Python 中使用

```python
from training.expert import load_expert_dataset

# 加载数据集
dataset = load_expert_dataset("data/expert_demonstrations")

# 查看统计信息
from training.expert import print_statistics
print_statistics(dataset)

# 分割训练集和验证集
train_ds, val_ds = dataset.split(train_ratio=0.8)

# 获取批次数据
batch = train_ds.get_batch(batch_size=32)

for sample in batch:
    observation = sample["observation"]
    actions = sample["actions"]
    reward = sample["reward"]
    # 训练模型...
```

### 3. 使用专家智能体

```python
from training.expert import KaggleExpertAgent

# 创建专家智能体
expert = KaggleExpertAgent(player_id=0)

# 获取动作
observation = env.get_raw_observation(player_id=0)
actions = expert.get_actions(observation)
# actions 格式: [[from_planet_id, angle, num_ships], ...]
```

## 数据格式

每个样本包含以下字段：

```python
{
    "player_id": int,              # 玩家 ID (0-3)
    "step": int,                   # 当前回合数
    "observation": {               # Kaggle 格式的观测
        "player": int,
        "planets": [[id, owner, x, y, radius, ships, production], ...],
        "fleets": [[id, owner, x, y, angle, from_planet_id, ships], ...],
        "angular_velocity": float,
        "initial_planets": [...],
        "comets": [...],
        "comet_planet_ids": [...],
        # ... 其他字段
    },
    "actions": [                   # 专家采取的动作
        [from_planet_id, angle, num_ships],
        ...
    ],
    "reward": float,               # 获得的奖励
    "done": bool,                  # 游戏是否结束
}
```

## 高级用法

### 自定义数据生成

```python
from training.expert import ExpertDataGenerator

generator = ExpertDataGenerator(
    num_players=4,
    save_dir="data/my_expert_data",
)

# 生成单局对局
trajectory = generator.generate_episode(seed=42)

# 生成完整数据集
file_paths = generator.generate_dataset(
    num_episodes=100,
    episodes_per_file=10,
    prefix="my_data",
)
```

### 分析数据质量

```python
from training.expert import ExpertDataset, print_statistics

dataset = ExpertDataset(
    data_dir="data/expert_demonstrations",
    max_samples=10000,  # 只加载部分数据
)

print_statistics(dataset)

# 分析动作分布
action_counts = [len(s["actions"]) for s in dataset.data]
import numpy as np
print(f"平均动作数: {np.mean(action_counts):.2f}")
print(f"最大动作数: {max(action_counts)}")
```

## 性能指标

- **生成速度**: ~6 秒/局 (4 玩家)
- **数据大小**: ~1 MB/局 (pickle 格式)
- **样本数量**: ~2000 样本/局/玩家
- **内存占用**: ~500 MB / 100 局对局

## 注意事项

1. **数据质量**: 专家策略虽然强大，但不是最优解。建议：
   - 生成足够多的数据（建议 > 500 局）
   - 定期检查数据质量
   - 可以混合多个专家策略的数据

2. **存储空间**: 大规模数据集可能占用大量磁盘空间：
   - 100 局 ≈ 100 MB
   - 1000 局 ≈ 1 GB

3. **训练建议**:
   - 先用少量数据（50-100 局）验证训练流程
   - 确认模型能从专家数据中学习
   - 再逐步增加数据量

## 后续改进

- [ ] 添加更多专家策略变体
- [ ] 实现数据增强（旋转、镜像等）
- [ ] 支持增量更新数据集
- [ ] 添加数据质量评估指标
- [ ] 优化数据存储格式（HDF5、Parquet 等）

## 参考资料

- Kaggle Orbit Wars 竞赛: https://www.kaggle.com/competitions/orbit-wars
- 原始策略代码: training/expert/kaggle_expert.py
