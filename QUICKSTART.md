# Orbit Wars 项目快速指南

## 📁 项目结构已整理

所有文件已整理完毕，目录结构清晰：

```
Orbit-Wars/
├── docs/                  # 所有文档
├── scripts/               # 测试和工具脚本
├── training/              # 训练模块
│   └── expert/            # 专家数据系统
├── data/                  # 数据目录
└── generate_expert_data.sh # 快捷脚本
```

## 🚀 快速开始

### 1. 生成专家数据（1000 局）

```bash
./generate_expert_data.sh
```

这将生成：
- **1000 局对局**
- **约 8,000,000 个样本**
- **约 10 GB 数据**
- **耗时 15-20 分钟**（8 进程并行）

### 2. 训练模型

```bash
# 使用专家数据预训练
python training/train_ray.py --config training/config/expert_pretraining.yaml
```

## 📖 文档指南

| 文档 | 说明 | 位置 |
|------|------|------|
| **快速指南** | 本文档 | `QUICKSTART.md` |
| **数据生成指南** | 如何生成专家数据 | `docs/HOW_TO_GENERATE_DATA.md` |
| **专家系统总结** | 系统架构和完成情况 | `docs/EXPERT_DATA_SUMMARY.md` |
| **训练指南** | 如何训练模型 | `docs/TRAINING.md` |
| **游戏规则** | Orbit Wars 规则 | `docs/rule.md` |
| **Agent 设计** | 智能体设计文档 | `docs/agent.md` |
| **使用指南** | 详细 API 说明 | `training/expert/USAGE.md` |
| **项目结构** | 完整目录结构 | `docs/PROJECT_STRUCTURE.md` |

## 🛠️ 常用脚本

### 根目录快捷脚本

```bash
./generate_expert_data.sh  # 生成 1000 局专家数据
```

### 测试脚本

```bash
# 测试多进程生成器
python scripts/test_multiprocess.py

# 测试 JSONL 格式
python scripts/test_jsonl_format.py

# 查看使用示例
python scripts/example_use_expert_data.py
```

### 手动运行（自定义参数）

```bash
python -m training.expert.multiprocess_generator \
    --num_episodes 1000 \
    --num_processes 8 \
    --save_dir data/expert_demonstrations \
    --format jsonl \
    --validate
```

## 📊 数据位置

生成的数据保存在：

```
data/expert_demonstrations/
├── expert_data_0000.jsonl
├── expert_data_0001.jsonl
├── ...
└── expert_data_0049.jsonl
```

## 🎯 使用数据训练

```python
from training.expert import load_expert_dataset

# 加载数据
dataset = load_expert_dataset("data/expert_demonstrations")
print(f"总样本数: {len(dataset):,}")

# 分割数据集
train_ds, val_ds = dataset.split(train_ratio=0.8)

# 获取批次训练
batch = train_ds.get_batch(batch_size=256)
```

## ⚙️ 配置说明

专家数据预训练配置位于：

```
training/config/expert_pretraining.yaml
```

关键配置项：

```yaml
expert_data:
  enabled: true                       # 启用专家数据预训练
  data_dir: data/expert_demonstrations # 数据目录
  num_pretrain_iterations: 100        # 预训练迭代次数
  mix_expert_data_ratio: 0.3          # 混合数据比例
  use_until_iteration: 100            # 使用到第 N 次迭代
```

## 🔧 系统要求

### 生成数据

- **CPU**: 8 核（可调整）
- **内存**: 2-3 GB
- **磁盘**: 10 GB 可用空间
- **时间**: 15-20 分钟

### 训练模型

- **GPU**: 推荐（至少 8GB 显存）
- **内存**: 16+ GB
- **磁盘**: 20+ GB（用于检查点和日志）

## 📈 预期性能

### 生成数据性能

```
进度显示: 生成进度 | 成功: 456 | 样本: 3,648,000 : 45%|████▌
速度: ~0.9 局/秒（8 进程）
时间: ~18 分钟（1000 局）
```

### 训练性能

```
预训练阶段: 100 次迭代
混合训练阶段: 100-500 次迭代
纯 RL 阶段: 500+ 次迭代
```

## 🐛 故障排除

### Q: 生成数据时出错？

A: 检查以下几点：
1. 磁盘空间是否充足（需要 10GB+）
2. 虚拟环境是否激活
3. 依赖包是否安装完整

### Q: 训练时找不到数据？

A: 确认：
1. 已运行 `./generate_expert_data.sh`
2. 数据位于 `data/expert_demonstrations/`
3. 配置文件路径正确

### Q: 如何调整生成速度？

A: 修改进程数：

```bash
# 慢速（2 进程）
python -m training.expert.multiprocess_generator --num_processes 2

# 快速（16 进程）
python -m training.expert.multiprocess_generator --num_processes 16
```

## 📚 更多资源

- [Kaggle Orbit Wars 竞赛](https://www.kaggle.com/competitions/orbit-wars)
- [训练配置文件](training/config/expert_pretraining.yaml)
- [专家系统文档](training/expert/USAGE.md)

---

**准备好了吗？开始生成数据吧！** 🚀

```bash
./generate_expert_data.sh
```
