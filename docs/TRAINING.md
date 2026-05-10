# 训练配置说明

## 🔧 配置项

### 专家数据预训练配置

在 `training/config/expert_pretraining.yaml` 中配置：

```yaml
expert_data:
  # 基础配置
  enabled: true                        # 启用专家数据预训练
  data_dir: data/expert_demonstrations  # 数据目录

  # 训练配置
  num_pretrain_iterations: 100         # ⭐ 最大训练轮数
  pretrain_batch_size: 256             # 批次大小
  pretrain_learning_rate: 1e-4         # 学习率
  behavior_clone_loss_coef: 1.0        # 损失系数

  # Ray 分布式配置
  num_pretrain_workers: 4              # ⭐ Worker 数量
  gpus_per_worker: 0.5                 # ⭐ 每个 Worker GPU 数量

  # 自动化配置
  auto_proceed: true                   # ⭐ 自动继续（不中断）
```

## 📊 资源使用

### 预训练阶段（Ray 分布式）

使用 **Ray 多 GPU 并行**：

```
Worker 0: GPU 0.0
Worker 1: GPU 0.5
Worker 2: GPU 1.0
Worker 3: GPU 1.5
```

- 总 GPU: 约 2 个
- 每个Worker独立训练
- 自动聚合梯度

### 强化学习阶段（Ray 分布式）

```
Rollout Workers: 8 个
每个 Worker: 0.25 GPU
Trainer: 1 个完整 GPU
```

## ⚙️ 调整 GPU 数量

### 方案 1: 调整 Worker 数量

```yaml
expert_data:
  num_pretrain_workers: 8      # 增加到 8 个
  gpus_per_worker: 0.25        # 每个 Worker 0.25 GPU
```
总 GPU: 8 × 0.25 = 2 个

### 方案 2: 调整每 Worker GPU

```yaml
expert_data:
  num_pretrain_workers: 4      # 保持 4 个
  gpus_per_worker: 1.0         # 每个 Worker 1 个完整 GPU
```
总 GPU: 4 × 1.0 = 4 个

## 🚀 训练命令

### 完整训练（推荐）

```bash
./train.sh
```

自动执行：
1. 生成数据（如需要）
2. Ray 分布式预训练
3. 自动评估
4. Ray 分布式强化学习

### 单独预训练

```bash
python training/train_with_expert.py \
    --config training/config/expert_pretraining.yaml
```

## 📈 训练进度

### 预训练

```
--- 迭代 1/100 ---
  训练 4 个 workers...
  聚合模型参数...
  Train Loss: 2.345 | Val Loss: 2.412
  ✓ 保存 checkpoint: training/checkpoints/pretrain_iter_50.pkl
```

### 强化学习

```
Iteration 100 | Reward: 125.3 | Win Rate: 0.52
Iteration 200 | Reward: 156.7 | Win Rate: 0.58
```

## 💾 输出文件

### Checkpoints

```
training/checkpoints/
├── pretrain_iter_50.pkl        # 预训练中间检查点
├── pretrain_iter_100.pkl       # 预训练中间检查点
├── pretrained_model.pkl        # 预训练最终模型
├── iter_100.pkl                # RL 检查点
├── iter_200.pkl                # RL 检查点
└── best_model.pkl              # 最佳模型
```

### SwanLab

- 实时查看训练曲线
- 对比不同实验
- 监控 GPU 使用
