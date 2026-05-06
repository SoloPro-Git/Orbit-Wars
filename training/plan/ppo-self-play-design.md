# Orbit Wars RL 训练方案设计

## 方案概述

**核心方案**: PPO 自我博弈 + 历史对手池进化
**备选方案**: AlphaZero MCTS（备选 A）、RL + Beam Search 混合（备选 B）
**技术栈**: PyTorch + SwanLab 实验管理 + kaggle_environments

---

## 1. 整体架构

```
[自我博弈 Rollout (8卡并行)] → [数据 Buffer] → [PPO Update] → [Checkpoint] → [评估 & 加入对手池]
         ↑                                                                    |
         └──────── 从对手池采样对手 (2人/4人局随机) ←────────────────────────────┘
```

- Rollout 阶段: 8卡并行跑自我博弈，每局从对手池采样对手
- 训练阶段: 收集 (obs, action, reward, value) 数据，PPO 更新
- Checkpoint: 每 N 步保存到对手池
- 评估: 新策略 vs 池中最强对手，确认进化方向

---

## 2. 观察编码 (Observation Encoding)

### 2.1 输入特征

**星球特征 (Planet Features)**:

| 特征 | 说明 | 归一化 |
|------|------|--------|
| owner_one_hot [5] | owner 0-3 + neutral | - |
| x, y | 位置 | / 100 |
| radius | 星球半径 | / 10 |
| ships | 当前驻军 | / 100 |
| production | 产出率 (1-5) | / 5 |
| is_comet | 是否彗星 | binary |
| is_orbiting | 是否公转 | binary |
| distance_to_sun | 到太阳距离 | / 50 |

**预计算威胁特征 (Threat Features, 解析计算后拼接到星球)**:

| 特征 | 说明 |
|------|------|
| incoming_enemy_ships | 即将到达的敌方总飞船数 |
| incoming_enemy_eta | 最早敌方到达时间 (步数) |
| incoming_friendly_ships | 己方增援数量 |
| incoming_friendly_eta | 己方增援到达时间 |
| threat_level | threat_strength / (garrison + eta * production) |
| is_contested | 多方舰队都朝此星球飞 |
| planet_economic_value | production * remaining_turns |
| comet_roi | (仅彗星) production * remaining_life - capture_cost |

威胁预测解析计算:
```
对每个敌方舰队:
    speed = 1.0 + 5.0 * (log(ships)/log(1000))^1.5
    trajectory = extrapolate(x, y, angle, speed, max_steps=50)
    对公转星球: 预测公转后位置，计算精确拦截窗口
    匹配最近碰撞星球 → predicted_target, eta, threat_strength
```

**舰队特征 (Fleet Features)**:

| 特征 | 说明 | 归一化 |
|------|------|--------|
| owner_one_hot [5] | owner 0-3 + neutral | - |
| x, y | 位置 | / 100 |
| angle | 飞行方向 | / π |
| ships | 舰队规模 | / 100 |
| speed | 计算的速度 | / 6 |
| predicted_target_dist | 预计到达目标距离 | / 100 |

**全局特征 (Global Features)**:

| 特征 | 说明 |
|------|------|
| angular_velocity | 公转角速度 |
| turn / max_turns | 当前回合进度 |
| own_total_ships | 己方总飞船 |
| own_total_production | 己方总产出 |
| enemy_total_ships | 敌方总飞船 |
| own_planet_count | 己方星球数 |
| num_players | 当前玩家数 (2 或 4) |

### 2.2 编码器

```
Planet Encoder:  TransformerEncoder(d_model, nhead, num_layers)
Fleet Encoder:   TransformerEncoder(d_model, nhead, num_layers)
Global:          MLP → d_model

Fusion:
    owned_planet_embeddings = planet_encoder[owner == player]
    context = all_planet_embeddings + fleet_embeddings + global
    fused = CrossAttention(queries=owned_planets, keys=context, values=context)
```

所有位置坐标相对于己方母星做归一化（位置不变策略）。

---

## 3. 动作空间设计 (两阶段解码)

### 3.1 阶段一: 目标选择 (Target Selection)

对每个己方星球，输出攻击/防御目标:
```
target_logits [N_owned, N_all_planets] → softmax → 目标星球 ID
```

用 Gumbel-Softmax 训练时保持可微，推理时 argmax。

### 3.2 阶段二: 兵力分配 (Force Allocation)

选定目标后，决定发送多少飞船:
```
num_ships = sigmoid(MLP(source_embedding, target_embedding)) * source.ships
```

**协调机制**: Transformer self-attention 让每个星球的 embedding 已包含全局信息，所以即使 action head 共享参数，输出也是协调的。可选增加一轮 cross-attention 在 action 之间做精炼。

### 3.3 输出格式转换

```
network output: [(target_planet_id, num_ships) for each owned planet]
→ 转换为 kaggle 格式: [from_planet_id, angle, num_ships]
    angle = atan2(target.y - source.y, target.x - source.x)
```

---

## 4. 奖励设计

### 4.1 终局奖励

| 条件 | 奖励 |
|------|------|
| 胜利 (排名 1) | +1.0 |
| 排名 2 | +0.3 |
| 排名 3 | -0.3 |
| 排名 4 / 失败 | -1.0 |

### 4.2 中间奖励 (经济模型, 归一化后权重 0.2)

**Ship 经济价值公式**:
```python
def planet_value(planet, turn, max_turns=500):
    remaining = max_turns - turn
    if is_comet(planet):
        remaining = min(remaining, comet_remaining_life(planet))
    return planet.production * remaining
```

| 奖励项 | 公式 | 说明 |
|--------|------|------|
| 占领净收益 | planet_value(target) - ships_sent - travel_time * cost | 占领是否划算 |
| 损失惩罚 | - ships_lost_in_failed_attack | 进攻失败代价 |
| 防御价值 | planet_value(defended) * 防御比例 | 成功防御的经济价值 |
| 舰队冻结成本 | in_transit_ships * (1 - discount) | 在途舰队的时间成本 |
| 生产力优势变化 | Δ(total_production) * remaining_turns | 生产力变化的长远价值 |
| 彗星 ROI | 仅 comet_roi > 0 时奖励 | 避免抢即将消失的彗星 |

---

## 5. 对手建模 (辅助任务)

```
共享 Backbone (Transformer)
    │
    ├── Policy Head (己方星球 → action)       ← 主任务, PPO loss
    ├── Value Head (全局 → 排名概率分布)       ← 主任务, value loss
    └── Opponent Head (敌方星球 → 预测动作)     ← 辅助任务, cross-entropy loss

L_total = L_ppo_policy + c1 * L_value + c2 * L_opponent_pred
```

对手预测头: 对每个敌方星球预测其 (target, num_ships)，用实际对手动作做监督。共享 backbone 被迫学会理解对手策略。

---

## 6. 自我博弈与对手池

### 6.1 对手池 (Opponent Pool)

```python
PoolEntry = {
    "policy_weights": state_dict,
    "elo": float,
    "generation": int,
    "timestamp": datetime
}
```

### 6.2 对手采样策略

| 比例 | 来源 | 目的 |
|------|------|------|
| 40% | 最新 checkpoint | 持续进化 |
| 30% | 对手池随机历史版本 | 防策略退化 |
| 20% | 池中 Elo 最高版本 | 学打败最强对手 |
| 10% | 简单启发式 agent | 保持基本功 |

### 6.3 2 人 / 4 人局随机

每局随机选择 2 人或 4 人模式:
- 2 人: 从池中采样 1 个对手
- 4 人: 从池中采样 3 个对手

### 6.4 Checkpoint 入池条件

- 新策略 vs 池中最强 win_rate > 0.45
- 或与池中已有策略的多样性足够 (策略 embedding 余弦距离 > 阈值)

---

## 7. 网络架构 (配置化)

### 7.1 配置项

```yaml
model:
  d_model: 128
  planet_encoder_layers: 3    # 可调 1-6
  fleet_encoder_layers: 2     # 可调 1-4
  fusion_layers: 2            # 可调 1-4
  nhead: 4                    # 可调 2-8
  mlp_ratio: 4
  dropout: 0.1
  use_opponent_head: true
  use_autoregressive_refine: false
  activation: "gelu"

training:
  framework: "pytorch"
  tracker: "swanlab"
  learning_rate: 3e-4
  lr_scheduler: "cosine"
  gamma: 0.99
  gae_lambda: 0.95
  ppo_clip: 0.2
  ppo_epochs: 4
  entropy_coef: 0.01
  value_coef: 0.5
  opponent_pred_coef: 0.1
  max_grad_norm: 0.5
  batch_size: 2048
  num_parallel_games: 128
  save_interval: 50

self_play:
  pool_size: 20
  sample_latest_ratio: 0.4
  sample_random_ratio: 0.3
  sample_best_ratio: 0.2
  sample_heuristic_ratio: 0.1
  two_player_prob: 0.3        # 30% 概率打 2 人局

reward:
  terminal_weight: 1.0
  intermediate_weight: 0.2
  capture_reward_weight: 0.3
  loss_penalty_weight: 0.2
  defense_reward_weight: 0.15
  transit_cost_weight: 0.1
  production_advantage_weight: 0.15
  comet_roi_weight: 0.1
```

### 7.2 网络结构

```
Input:
  planet_features [N_planets, D_planet]  (含预计算威胁特征)
  fleet_features  [N_fleets, D_fleet]
  global_features [D_global]

Planet Encoder:  TransformerEncoder(d_model, nhead, num_layers=planet_encoder_layers)
Fleet Encoder:   TransformerEncoder(d_model, nhead, num_layers=fleet_encoder_layers)
Global:          MLP → d_model

Fusion: CrossAttention(queries=owned_planets, keys/values=all_planets+fleets+global)
        num_layers=fusion_layers

Output Heads (per owned planet, 共享参数 MLP):
  target_logits  → softmax over all planets
  num_ships      → sigmoid * max_ships

Auxiliary Heads:
  value_head     → MLP → rank_prob [num_players] (softmax)
  opponent_head  → per enemy planet: (target_logits, num_ships)
```

---

## 8. 训练流程 (伪代码)

```python
pool = OpponentPool(max_size=20)

for iteration in range(MAX_ITERATIONS):
    # 1. 采样对手，随机 2/4 人局
    num_players = sample_num_players(two_player_prob)
    opponents = pool.sample(num_players - 1)

    # 2. 并行 rollout (8 卡)
    trajectories = parallel_rollout(
        current_policy, opponents,
        num_games=num_parallel_games,
        gpus=8
    )

    # 3. 计算中间奖励
    trajectories = compute_rewards(trajectories, reward_config)

    # 4. 计算 GAE
    advantages = compute_gae(trajectories, gamma, gae_lambda)

    # 5. PPO 更新
    for epoch in range(ppo_epochs):
        loss = ppo_update(trajectories, advantages)
        swanlab.log(loss)

    # 6. 评估 & 入池
    if iteration % save_interval == 0:
        win_rate = evaluate(current_policy, pool.best, num_games=200)
        if should_add_to_pool(current_policy, pool, win_rate):
            pool.add(current_policy, elo=compute_elo(win_rate))
        save_checkpoint(current_policy)
        swanlab.log({"elo": pool.best.elo, "win_rate": win_rate})
```

---

## 9. 推理格式转换

训练用 (target_planet_id, num_ships)，推理时转换为 kaggle 格式:

```python
def to_kaggle_action(owned_planet, target_planet, num_ships):
    angle = math.atan2(target_planet.y - owned_planet.y,
                       target_planet.x - owned_planet.x)
    return [owned_planet.id, angle, int(num_ships)]
```

Kaggle 提交格式: 单个 main.py + 模型权重文件打包为 tar.gz。

---

## 10. 实现任务清单

### Phase 1: 基础框架

- [ ] T1: 游戏环境封装 — 封装 kaggle_environments 为 gym-like 接口 (step, reset, observation_space)
- [ ] T2: 特征工程模块 — 实现威胁预测解析计算、经济价值计算、彗星 ROI
- [ ] T3: 观察编码器 — Planet Encoder + Fleet Encoder + Global Encoder (Transformer)
- [ ] T4: 配置系统 — YAML 配置 + dataclass 加载，模型/训练/奖励参数可配置

### Phase 2: 动作与策略

- [ ] T5: 两阶段解码器 — Target Selection Head + Force Allocation Head
- [ ] T6: 动作格式转换 — 网络输出 → kaggle action format 转换
- [ ] T7: Value Head — 排名概率分布输出 (支持 2/4 人)
- [ ] T8: Opponent Prediction Head — 辅助任务，预测对手动作

### Phase 3: 训练循环

- [ ] T9: PPO Trainer — 策略损失 + 价值损失 + entropy loss
- [ ] T10: 奖励计算模块 — 终局奖励 + 经济模型中间奖励
- [ ] T11: GAE 计算 — Generalized Advantage Estimation
- [ ] T12: 并行 Rollout — torch.multiprocessing 多 GPU 并行自我博弈

### Phase 4: 自我博弈系统

- [ ] T13: 对手池管理 — Checkpoint 存储/加载/采样/Elo 追踪
- [ ] T14: 对手采样策略 — 40/30/20/10 采样比例实现
- [ ] T15: 2人/4人局随机 — 训练中随机切换玩家数
- [ ] T16: 启发式基准 Agent — 实现简单启发式 agent 作为对手池种子

### Phase 5: 实验管理

- [ ] T17: SwanLab 集成 — 训练指标、Elo 曲线、胜率追踪
- [ ] T18: Checkpoint 管理 — 保存/加载/版本管理
- [ ] T19: 对局回放分析 — 下载 kaggle replay，分析败因

### Phase 6: 提交与优化

- [ ] T20: Kaggle 提交脚本 — main.py + 权重打包 + 提交
- [ ] T21: 推理优化 — 确保 1 秒/回合 (模型大小调整、torch.compile)
- [ ] T22: 超参搜索 — 网格搜索关键超参

---

## 备选方案

### 备选 A: AlphaZero MCTS

- MCTS + 神经网络，需要离散化角度 (16-32 方向)
- 4 人博弈 minimax 不如 2 人干净
- 实现复杂度高，但在搜索质量上上限更高
- 如果 PPO 方案遇到瓶颈可切换

### 备选 B: RL + Beam Search

- PPO 训练 + 推理时 beam search 增强
- 用 value head 评估 top-K 候选动作序列
- 复杂度介于方案 A 和 AlphaZero 之间
- 1 秒限制内可控
