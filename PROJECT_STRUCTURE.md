# 项目文件结构

## 目录结构

```
Orbit-Wars/
├── data/                              # 数据目录
│   └── expert_demonstrations/         # 专家演示数据（生成后）
│
├── docs/                              # 项目文档
│   ├── agent.md                       # Agent 设计文档
│   ├── rule.md                        # 游戏规则文档
│   ├── TRAINING.md                    # 训练指南
│   ├── EXPERT_DATA_SUMMARY.md         # 专家数据系统总结
│   └── HOW_TO_GENERATE_DATA.md        # 数据生成指南
│
├── scripts/                           # 工具脚本
│   ├── debug_run.py                   # 调试运行脚本
│   ├── example_use_expert_data.py     # 专家数据使用示例
│   ├── test_jsonl_format.py           # JSONL 格式测试
│   └── test_multiprocess.py           # 多进程测试
│
├── training/                          # 训练模块
│   ├── expert/                        # 专家数据系统
│   │   ├── __init__.py
│   │   ├── base_agent.py              # 基类
│   │   ├── kaggle_expert.py           # Kaggle 高分策略
│   │   ├── data_generator.py          # 数据生成器
│   │   ├── pretraining.py             # 预训练器
│   │   ├── multiprocess_generator.py  # 多进程生成器
│   │   ├── generate_large_dataset.py  # 大规模数据生成
│   │   ├── generate_expert_data.py    # 便捷工具
│   │   ├── generate_1000_episodes.sh  # 生成脚本（1000局）
│   │   ├── README.md                  # 技术文档
│   │   └── USAGE.md                   # 使用指南
│   │
│   ├── config/                        # 配置文件
│   │   └── expert_pretraining.yaml    # 专家预训练配置
│   │
│   ├── core/                          # 核心训练模块
│   │   ├── config.py                  # 配置管理
│   │   ├── model.py                   # 模型定义
│   │   ├── ppo.py                     # PPO 算法
│   │   ├── feature_engineering.py     # 特征工程
│   │   ├── env_wrapper.py             # 环境封装
│   │   ├── action.py                  # 动作解码
│   │   └── ...
│   │
│   ├── checkpoints/                   # 模型检查点
│   ├── logs/                          # 训练日志
│   ├── train_ray.py                   # Ray 分布式训练入口
│   └── train.sh                       # 训练启动脚本
│
├── inference/                         # 推理模块
│   ├── predictor.py                   # 预测器
│   └── ...
│
├── generate_expert_data.sh            # 快捷脚本（生成数据）
├── main.py                            # 主入口
├── pyproject.toml                     # 项目配置
├── uv.lock                            # 依赖锁定文件
└── .gitignore                         # Git 忽略规则
```

## 快速开始

### 生成专家数据

```bash
./generate_expert_data.sh
```

这将生成 1000 局专家数据（约 800 万样本，10 GB）。

### 训练模型

```bash
python training/train_ray.py --config training/config/expert_pretraining.yaml
```

### 使用文档

- [HOW_TO_GENERATE_DATA.md](docs/HOW_TO_GENERATE_DATA.md) - 如何生成数据
- [EXPERT_DATA_SUMMARY.md](docs/EXPERT_DATA_SUMMARY.md) - 专家数据系统总结
- [USAGE.md](training/expert/USAGE.md) - 详细使用指南

## 脚本说明

### 根目录脚本

- `generate_expert_data.sh` - 快捷脚本，调用 `training/expert/generate_1000_episodes.sh`

### Scripts 目录

- `test_multiprocess.py` - 测试多进程生成器
- `test_jsonl_format.py` - 测试 JSONL 格式
- `example_use_expert_data.py` - 使用示例
- `debug_run.py` - 调试工具

### Expert 目录脚本

- `generate_1000_episodes.sh` - 生成 1000 局数据（主脚本）
- `generate_expert_data.py` - 便捷数据生成工具
- `generate_large_dataset.py` - 大规模数据生成
- `multiprocess_generator.py` - 多进程生成器（可独立运行）

## 数据格式

### JSONL 格式（推荐）

每行一个 JSON 对象：

```json
{"player_id": 0, "step": 42, "observation": {...}, "actions": [...], "reward": 0.0, "done": false}
```

优点：
- 人类可读
- 支持流式加载
- 易于调试
- 可用 `jq` 等工具处理

## 开发指南

### 添加新功能

1. 专家策略：在 `training/expert/` 添加新文件
2. 训练功能：在 `training/core/` 添加模块
3. 脚本工具：放在 `scripts/` 目录

### 文档规范

- API 文档：放在模块内
- 用户指南：放在 `docs/` 目录
- 技术文档：放在相关模块目录

## 维护建议

1. **定期清理**
   - 清理 `data/` 目录下的测试数据
   - 清理 `training/checkpoints/` 旧检查点
   - 清理 `swanlog/` 旧日志

2. **版本控制**
   - `data/` 目录已在 `.gitignore` 中
   - 不要提交大文件到 Git
   - 使用脚本重新生成数据

3. **性能优化**
   - 使用多进程生成数据
   - 合理设置进程数（CPU 核心数的 50-80%）
   - 定期监控磁盘空间
