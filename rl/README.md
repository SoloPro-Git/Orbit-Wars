# Orbit Wars PPO Training Framework

## Overview
Complete PPO training framework with **pointer network** for target selection (key innovation over BC model).

## Files Created

### 1. `rl/env.py` (324 lines)
Simplified Orbit Wars environment for RL training.

**Features:**
- Self-contained game engine (no kaggle-environments dependency)
- Simplified mechanics: distance-based collision, no comets
- 4-fold symmetric map generation
- Planet rotation (inner planets orbit the sun)
- Fleet speed scaling: `speed = 1 + 5 * (log(ships)/log(1000))^1.5`
- Combat resolution with multi-player support

**Built-in Opponents:**
- `StarterBot`: Simple rules (attack nearest, send half ships)
- `RandomBot`: Random actions
- `AggressiveBot`: Aggressive expansion (targets weak enemies, sends 70% ships)

**Usage:**
```python
env = OrbitWarsEnv(num_players=2)
obs = env.reset(seed=42)
obs, rewards, dones, infos = env.step([actions_p0, actions_p1])
```

### 2. `rl/agent.py` (325 lines)
PPO agent with pointer network for target selection.

**Key Innovation:**
- **Pointer Network**: Policy outputs `target_planet_idx` (discrete choice over all planets)
- **NOT raw angle**: Angle is computed from selected target's position
- This solves the BC model's problem of predicting random angles

**Architecture:**
```
Encoder (can warm-start from BC):
  - planet_enc: [B, N, 7] → [B, N, D]
  - fleet_enc: [B, M, 7] → [B, M, D]
  - global_enc: [B, 13+num_players] → [B, D]

Policy Head (per owned planet):
  - send_head: logit for send/no-send (Bernoulli)
  - ptr_query/ptr_key: pointer network over all planets (Categorical)
  - ship_head: ship ratio 0-1 (sigmoid)

Value Head:
  - V(s) scalar for PPO baseline
```

**Parameters:** ~40K (embed_dim=64)

**Usage:**
```python
agent = PPOAgent(embed_dim=64)
actions, log_prob, entropy, value, info = agent.get_action_and_value(
    planets, fleets, player_id, step, planet_mask, fleet_mask, owned_mask)
```

### 3. `rl/ppo_trainer.py` (267 lines)
PPO trainer with GAE and opponent pool.

**Features:**
- **GAE** (Generalized Advantage Estimation): λ=0.95
- **Clipped surrogate objective**: ε=0.2
- **Value function clipping**: prevents large value updates
- **Entropy bonus**: encourages exploration
- **Opponent pool**: rotate through different opponents
- **Parallel rollout collection**: efficient data gathering

**Hyperparameters:**
```python
gamma = 0.99          # discount factor
gae_lambda = 0.95     # GAE lambda
clip_ratio = 0.2      # PPO clip
ppo_epochs = 4        # epochs per rollout
batch_size = 64       # mini-batch size
lr = 3e-4             # learning rate
ent_coef = 0.01       # entropy coefficient
vf_coef = 0.5         # value loss coefficient
```

**Usage:**
```python
trainer = PPOTrainer(env, agent, opponent_pool=['starter', 'random', 'aggressive'])
trainer.collect_rollouts(num_steps=2048)
stats = trainer.train_step()
eval_stats = trainer.evaluate(opponent='starter', num_games=50)
```

### 4. `rl/train.py` (189 lines)
Main training script with CLI interface.

**Features:**
- BC warm start (loads encoder from `checkpoints/bc_1v1_v2_best.pt`)
- Periodic evaluation against opponent pool
- Checkpoint saving (best + periodic)
- Comprehensive logging

**Usage:**
```bash
# Basic training
python3 rl/train.py --total-timesteps 1000000

# With custom options
python3 rl/train.py \
  --total-timesteps 500000 \
  --rollout-steps 2048 \
  --eval-freq 10 \
  --opponents starter random aggressive \
  --device cuda \
  --exp-name ppo_v1

# Without BC warm start
python3 rl/train.py --no-warm-start
```

**Output:**
```
checkpoints/ppo_v1/
  ├── best.pt           # Best model by average win rate
  ├── final.pt          # Final model
  └── checkpoint_*.pt   # Periodic checkpoints
```

## Test Results

```
✓ Environment: 40 planets, 2 players, working game loop
✓ Agent: 39,619 parameters, correct output shapes
✓ Pointer network: target_logits [1, 40, 40] over all planets
✓ Action sampling: Bernoulli (send) + Categorical (target)
✓ PPO trainer: rollout collection and training working
```

## Key Differences from BC Model

| Aspect | BC Model | PPO Agent |
|--------|----------|-----------|
| **Target Selection** | Raw angle prediction | Pointer network (discrete choice) |
| **Training** | Supervised (replay data) | Reinforcement learning (self-play) |
| **Reward Signal** | None (imitation) | Win/loss + shaped rewards |
| **Exploration** | None | Entropy bonus |
| **Opponent** | Fixed (replay data) | Diverse pool (starter/random/aggressive) |

## Next Steps

1. **Run training:**
   ```bash
   python3 rl/train.py --total-timesteps 500000 --device cpu
   ```

2. **Monitor progress:**
   - Training logs show loss, entropy, rewards
   - Evaluation logs show win rates vs each opponent
   - Best model saved automatically

3. **Expected progression:**
   - Initial: ~20-30% win rate vs starter
   - After 100K steps: ~50-60% win rate
   - After 500K steps: ~70-80% win rate
   - Final: 80%+ win rate vs all opponents

4. **Integration:**
   - Load trained model in `main.py`
   - Replace BC model with PPO agent
   - Test against Kaggle starter bot

## Architecture Diagram

```
Observation
    ↓
[Planet Encoder] → planet_emb [B, N, D]
    ↓
[Fleet Encoder] → fleet_emb [B, M, D]
    ↓
[Global Encoder] → global_feat [B, D]
    ↓
[Action Head] → action_feat [B, N, D]
    ↓
    ├─→ [Send Head] → send_logits [B, N, 1] → Bernoulli
    ├─→ [Ship Head] → ship_ratio [B, N, 1] → sigmoid
    └─→ [Pointer Network] → target_logits [B, N, N] → Categorical
                              ↓
                    Select target planet index
                              ↓
                    Compute angle from target position
                              ↓
                    Action: [planet_id, angle, ships]
```

## Files Summary

| File | Lines | Purpose |
|------|-------|---------|
| `rl/env.py` | 324 | Simplified game environment + bots |
| `rl/agent.py` | 325 | PPO agent with pointer network |
| `rl/ppo_trainer.py` | 267 | PPO training logic |
| `rl/train.py` | 189 | Main training script |
| **Total** | **1,105** | Complete RL framework |

All files <400 lines as requested. Code is documented, tested, and ready to run.
