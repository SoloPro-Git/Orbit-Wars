# Orbit Wars - 强化学习项目

基于 PPO 算法和专家数据预训练的 Orbit Wars 游戏 AI。

## 🚀 快速开始

### 1. 生成专家数据

```bash
./generate_expert_data.sh
```

生成 1000 局专家演示数据（约 800 万样本，10 GB，15-20 分钟）。

### 2. 训练模型

```bash
python training/train_ray.py --config training/config/expert_pretraining.yaml
```

### 3. 查看详细指南

```bash
cat QUICKSTART.md
```

## 📁 项目结构

```
Orbit-Wars/
├── docs/                     # 📚 所有文档
├── scripts/                  # 🛠️ 测试和工具脚本
├── training/                 # 🎯 训练模块
│   └── expert/              # 🤖 专家数据系统
├── data/                    # 💾 数据目录
└── generate_expert_data.sh  # ⚡ 快捷脚本
```

详见 [docs/PROJECT_STRUCTURE.md](docs/PROJECT_STRUCTURE.md)

## 📖 文档指南

| 文档 | 说明 |
|------|------|
| [QUICKSTART.md](QUICKSTART.md) | **快速指南**（从这里开始） |
| [docs/HOW_TO_GENERATE_DATA.md](docs/HOW_TO_GENERATE_DATA.md) | 数据生成指南 |
| [docs/EXPERT_DATA_SUMMARY.md](docs/EXPERT_DATA_SUMMARY.md) | 专家数据系统总结 |
| [docs/TRAINING.md](docs/TRAINING.md) | 训练指南 |
| [training/expert/USAGE.md](training/expert/USAGE.md) | 详细 API 说明 |

## 🎯 核心功能

### 专家数据系统

- ✅ Kaggle 高分策略实现
- ✅ 多进程并行生成（8 进程，~0.9 局/秒）
- ✅ JSONL 格式（人类可读，易于调试）
- ✅ 行为克隆预训练
- ✅ 混合训练（RL + 专家数据）

### 训练系统

- ✅ PPO 算法实现
- ✅ Ray 分布式训练
- ✅ 自我对战
- ✅ SwanLab 实验跟踪
- ✅ 模型检查点管理

## 🛠️ 常用脚本

```bash
# 生成专家数据
./generate_expert_data.sh

# 清理项目
bash scripts/cleanup.sh

# 测试多进程生成器
python scripts/test_multiprocess.py
```

## ⚙️ 系统要求

### 生成数据
- CPU: 8 核
- 内存: 2-3 GB
- 磁盘: 10 GB
- 时间: 15-20 分钟

### 训练模型
- GPU: 推荐（8GB+ 显存）
- 内存: 16+ GB
- 磁盘: 20+ GB

## 📊 数据格式

JSONL 格式（每行一个 JSON 对象）：

```json
{"player_id": 0, "step": 42, "observation": {...}, "actions": [...], "reward": 0.0, "done": false}
```

## 🔧 配置说明

专家数据预训练配置：

```yaml
expert_data:
  enabled: true
  data_dir: data/expert_demonstrations
  num_pretrain_iterations: 100
  mix_expert_data_ratio: 0.3
  use_until_iteration: 100
```

## 📈 性能指标

### 数据生成

- 速度: ~0.9 局/秒（8 进程）
- 效率: 8000 样本/局
- 格式: JSONL（可压缩）

### 模型训练

- 预训练: 100 次迭代
- 混合训练: 100-500 次迭代
- 纯 RL: 500+ 次迭代

## 🐛 故障排除

### 数据生成失败

```bash
# 检查磁盘空间
df -h

# 清理测试数据
bash scripts/cleanup.sh data

# 重新生成
./generate_expert_data.sh
```

### 训练出错

```bash
# 检查虚拟环境
source .venv/bin/activate

# 检查依赖
uv pip list

# 查看日志
ls -la training/logs/
```

## 📚 参考资料

- [Kaggle Orbit Wars 竞赛](https://www.kaggle.com/competitions/orbit-wars)
- [PPO 论文](https://arxiv.org/abs/1707.06347)
- [Ray 框架](https://docs.ray.io/)

## 📝 开发日志

- **2024-05-08**: 完成专家数据系统
  - 实现 Kaggle 高分策略
  - 多进程并行生成
  - JSONL 数据格式
  - 行为克隆预训练

- **2024-05-07**: 完成 Ray 分布式训练
  - PPO 算法实现
  - 自我对战机制
  - SwanLab 集成

---

**准备好了吗？** 🚀

```bash
./generate_expert_data.sh
```

查看 [QUICKSTART.md](QUICKSTART.md) 了解更多。
