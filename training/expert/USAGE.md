# 专家数据系统使用指南

## 快速开始

### 1. 生成专家数据

```bash
# 生成 100,000 条专家演示数据（推荐）
./generate_100k_expert_data.sh

# 或使用 Python 模块
python -m training.expert.generate_large_dataset --num_samples 100000 --format jsonl
```

### 2. 配置训练

在配置文件中启用专家数据预训练：

```yaml
expert_data:
  enabled: true                        # 启用专家数据预训练
  data_dir: data/expert_demonstrations  # 专家数据目录
  num_pretrain_iterations: 100         # 预训练迭代次数
  pretrain_batch_size: 256             # 预训练批次大小
  pretrain_learning_rate: 1e-4         # 预训练学习率
  mix_expert_data_ratio: 0.3           # 在 RL 训练中混合专家数据的比例
  use_until_iteration: 100             # 前 N 次迭代使用专家数据
```

### 3. 训练模型

```bash
# 使用专家数据预训练
python training/train_ray.py --config training/config/expert_pretraining.yaml
```

## 训练流程

### 阶段 1: 专家数据预训练（Iteration 0-100）

- 使用行为克隆训练模型模仿专家策略
- 学习基本的游戏规则和策略
- 不需要与环境交互，训练快速

### 阶段 2: 混合训练（Iteration 101-500）

- 混合使用 RL 采集的数据和专家数据
- 比例：70% RL 数据 + 30% 专家数据
- 逐步减少对专家数据的依赖

### 阶段 3: 纯 RL 训练（Iteration 500+）

- 完全使用 RL 采集的数据
- 通过自我对战不断改进
- 探索新的策略

## 数据格式

数据以 JSONL 格式存储，每行一个样本：

```json
{
  "player_id": 0,
  "step": 42,
  "observation": {
    "planets": [[id, owner, x, y, radius, ships, production], ...],
    "fleets": [[id, owner, x, y, angle, from_planet_id, ships], ...],
    "angular_velocity": 0.04,
    ...
  },
  "actions": [[from_planet_id, angle, num_ships], ...],
  "reward": 0.0,
  "done": false
}
```

## 代码示例

### 加载和使用专家数据

```python
from training.expert import load_expert_dataset, ExpertPretrainer
from training.core.config import AppConfig

# 加载配置
config = AppConfig.from_yaml("training/config/expert_pretraining.yaml")

# 加载数据集
dataset = load_expert_dataset(config.expert_data.data_dir)
train_ds, val_ds = dataset.split(train_ratio=0.8)

# 创建预训练器
pretrainer = ExpertPretrainer(model, config.expert_data)

# 预训练模型
stats = pretrainer.pretrain(feature_extractor, num_iterations=100)

# 保存预训练模型
torch.save(model.state_dict(), "checkpoints/pretrained_model.pkl")
```

### 在训练中混合专家数据

```python
from training.expert import mix_expert_data_with_rl

# RL 采集数据
rl_batch = collect_rollouts(env, model)

# 混合专家数据
expert_dataset = load_expert_dataset("data/expert_demonstrations")
mixed_batch = mix_expert_data_with_rl(
    rl_batch,
    expert_dataset,
    mix_ratio=0.3,
)

# 训练模型
model.update(mixed_batch)
```

## 文件结构

```
training/expert/
├── __init__.py                  # 模块导出
├── base_agent.py                # ExpertAgent 基类
├── kaggle_expert.py             # Kaggle 高分策略
├── data_generator.py            # 数据生成器（支持 JSONL/PKL）
├── pretraining.py               # 预训练器
├── generate_large_dataset.py    # 大规模数据生成脚本
├── generate_expert_data.py      # 便捷数据生成工具
├── test_expert_data_generation.py  # 测试套件
├── example_use_expert_data.py   # 使用示例
└── README.md                    # 详细文档

根目录：
├── generate_100k_expert_data.sh # 生成 100k 数据的快捷脚本
├── test_jsonl_format.py         # JSONL 格式测试
└── training/config/
    └── expert_pretraining.yaml  # 专家预训练配置示例
```

## 配置选项

### ExpertDataConfig

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用专家数据预训练 |
| `data_dir` | str | `"data/expert_demonstrations"` | 专家数据目录 |
| `num_pretrain_iterations` | int | `100` | 预训练迭代次数 |
| `pretrain_batch_size` | int | `256` | 预训练批次大小 |
| `pretrain_learning_rate` | float | `1e-4` | 预训练学习率 |
| `behavior_clone_loss_coef` | float | `1.0` | 行为克隆损失系数 |
| `mix_expert_data_ratio` | float | `0.3` | 在 RL 训练中混合专家数据的比例 |
| `use_until_iteration` | int | `100` | 前 N 次迭代使用专家数据（-1 表示一直使用） |

## 性能指标

- **生成速度**: ~6 秒/局 (4 玩家)
- **数据大小**: ~10 MB/局 (JSONL 格式)
- **样本数量**: ~2000 样本/局/玩家
- **100,000 样本**: ~13 局，~80 秒，~130 MB
- **1,000,000 样本**: ~125 局，~13 分钟，~1.2 GB

## 故障排除

### Q: 数据生成失败？
A: 检查磁盘空间，确保至少有 2GB 可用空间。使用 `--format jsonl` 生成更小的文件。

### Q: 训练时找不到专家数据？
A: 检查配置文件中的 `data_dir` 路径是否正确，确保数据已生成。

### Q: 预训练后模型表现差？
A:
1. 检查数据质量，使用 `--validate` 验证
2. 增加 `num_pretrain_iterations`
3. 调整 `pretrain_learning_rate`

### Q: 如何判断模型已学会专家策略？
A: 监控以下指标：
- 预训练损失持续下降
- 与专家对战胜率 > 40%
- 产生的动作分布与专家相似

## 最佳实践

1. **数据量**: 建议生成 100,000 - 500,000 条数据
2. **多样性**: 使用不同的随机种子生成数据
3. **验证**: 生成后使用 `--validate` 检查数据质量
4. **渐进训练**: 先预训练，再混合训练，最后纯 RL
5. **监控指标**: 同时监控预训练损失和 RL 奖励

## 下一步

- [ ] 添加更多专家策略变体
- [ ] 实现数据增强
- [ ] 支持在线学习（边生成边训练）
- [ ] 添加更详细的数据质量报告

## 参考资料

- Kaggle Orbit Wars 竞赛: https://www.kaggle.com/competitions/orbit-wars
- 行为克隆论文: "Behavioral Cloning from Observation"
- PPO 算法: "Proximal Policy Optimization Algorithms"
