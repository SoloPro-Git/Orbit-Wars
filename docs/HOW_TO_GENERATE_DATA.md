# 生成专家数据指南

## 快速开始

### 生成 1000 局专家数据（推荐）

在项目根目录执行：

```bash
./generate_expert_data.sh
```

或直接使用：

```bash
bash training/expert/generate_1000_episodes.sh
```

这将生成：
- **1000 局对局**
- **约 8,000,000 个样本**
- **约 10 GB 数据**（JSONL 格式）
- **耗时约 15-20 分钟**（8 进程并行）

## 参数说明

默认配置（推荐）：
- 对局数：1000
- 并行进程：8
- 每文件对局数：20
- 保存格式：JSONL
- 保存目录：`data/expert_demonstrations`

## 手动运行（自定义参数）

如果需要自定义参数：

```bash
python -m training.expert.multiprocess_generator \
    --num_episodes 1000 \
    --num_processes 8 \
    --episodes_per_file 20 \
    --save_dir data/expert_demonstrations \
    --format jsonl \
    --validate
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--num_episodes` | 生成对局数 | 1000 |
| `--num_processes` | 并行进程数（建议 CPU 核心数） | 8 |
| `--episodes_per_file` | 每个文件保存的对局数 | 20 |
| `--save_dir` | 保存目录 | data/expert_demonstrations |
| `--format` | 文件格式（jsonl/pkl） | jsonl |
| `--validate` | 生成后验证数据 | false |

## 进度显示

生成过程中会显示实时进度：

```
生成进度 | 成功: 123 | 样本: 984,000 : 45%|████▌     | 450/1000 [02:30<03:15,  2.87it/s]
```

显示内容：
- ✓ 成功生成的对局数
- ✓ 累计样本数
- ✓ 进度条和百分比
- ✓ 预计剩余时间

## 性能优化建议

### 进程数选择

根据你的 CPU 核心数选择：

```bash
# 查看 CPU 核心数
nproc

# 设置进程数（建议为核心数的 50-80%）
--num_processes 4   # 8 核 CPU
--num_processes 8   # 16 核 CPU
--num_processes 16  # 32 核 CPU
```

### 内存使用

每进程约占用 200-300 MB 内存，确保有足够内存：

```
内存需求 = 进程数 × 300 MB
```

## 数据验证

生成完成后自动验证（如果启用 `--validate`）：

```
验证数据...
发现 50 个数据文件
✓ 共加载 8,000,000 个样本

============================================================
数据集统计信息
============================================================
总样本数: 8,000,000

玩家样本分布:
  玩家 0: 2,000,000 个样本
  玩家 1: 2,000,000 个样本
  玩家 2: 2,000,000 个样本
  玩家 3: 2,000,000 个样本
...
```

## 使用生成的数据

### Python API

```python
from training.expert import load_expert_dataset

# 加载数据
dataset = load_expert_dataset("data/expert_demonstrations")

print(f"总样本数: {len(dataset):,}")

# 分割数据集
train_ds, val_ds = dataset.split(train_ratio=0.8)

# 获取批次
batch = train_ds.get_batch(batch_size=256)

# 训练模型
for sample in batch:
    observation = sample["observation"]
    actions = sample["actions"]
    # ... 训练代码
```

### 配置训练

在训练配置中启用专家数据：

```yaml
expert_data:
  enabled: true
  data_dir: data/expert_demonstrations
  num_pretrain_iterations: 100
  mix_expert_data_ratio: 0.3
  use_until_iteration: 100
```

## 磁盘空间

确保有足够的磁盘空间：

| 对局数 | 数据量 | 磁盘空间 |
|--------|--------|----------|
| 100    | 800K   | ~1 GB    |
| 500    | 4M     | ~5 GB    |
| 1000   | 8M     | ~10 GB   |
| 2000   | 16M    | ~20 GB   |

## 故障排除

### Q: 生成速度慢？

A: 调整进程数和检查 CPU 使用率：

```bash
# 减少进程数
--num_processes 4

# 或增加进程数（如果 CPU 有空闲）
--num_processes 16
```

### Q: 内存不足？

A: 减少并行进程数：

```bash
--num_processes 2
```

### Q: 磁盘空间不足？

A: 减少生成对局数或清理旧数据：

```bash
# 生成 500 局
--num_episodes 500

# 或清理后重新生成
rm -rf data/expert_demonstrations/*
```

### Q: 生成失败如何恢复？

A: 脚本支持断点续传，重新运行会跳过已生成的数据。

## 文件结构

生成后的文件结构：

```
data/expert_demonstrations/
├── expert_data_0000.jsonl  # 0-19 局
├── expert_data_0001.jsonl  # 20-39 局
├── expert_data_0002.jsonl  # 40-59 局
...
└── expert_data_0049.jsonl  # 980-999 局
```

每个文件约 200 MB，包含 20 局对局（约 160,000 样本）。

## 后续步骤

数据生成完成后：

1. **验证数据质量**（如果未启用 `--validate`）
   ```bash
   python -c "
   from training.expert import load_expert_dataset, print_statistics
   dataset = load_expert_dataset('data/expert_demonstrations')
   print_statistics(dataset)
   "
   ```

2. **开始训练**
   ```bash
   python training/train_ray.py --config training/config/expert_pretraining.yaml
   ```

3. **监控训练**
   - 检查预训练损失
   - 检查与专家对战胜率
   - 检查 RL 奖励

## 更多信息

- [USAGE.md](training/expert/USAGE.md) - 详细使用指南
- [EXPERT_DATA_SUMMARY.md](EXPERT_DATA_SUMMARY.md) - 系统总结
