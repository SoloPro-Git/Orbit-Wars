#!/bin/bash
# 快速测试训练脚本（1分钟）

set -e

source ../.venv/bin/activate
export SWANLAB_API_KEY="2tljFkPrQC4udTXKKfuwC"

echo "=========================================="
echo "  快速测试训练（1分钟）"
echo "=========================================="

# 修改配置为快速测试
cat > /tmp/test_quick.yaml << 'EOF'
model:
  d_model: 64
  nhead: 2
  mlp_ratio: 2
  planet_encoder_layers: 1
  fleet_encoder_layers: 1
  fusion_layers: 1
  dropout: 0.1
  activation: gelu
  use_opponent_head: false

training:
  max_iterations: 5
  num_parallel_games: 2
  batch_size: 32
  ppo_epochs: 2
  learning_rate: 0.001
  gamma: 0.995
  gae_lambda: 0.95
  ppo_clip: 0.2
  entropy_coef: 0.01
  value_coef: 0.5
  opponent_pred_coef: 0.1
  max_grad_norm: 1.0
  save_interval: 50

  swanlab_project: orbit-wars
  swanlab_experiment: ppo-test
  swanlab_mode: local
  resume_from_checkpoint: false
  max_checkpoints: 2
  keep_best_n: 1

environment:
  board_size: 100.0
  sun_radius: 10.0
  ship_speed: 6.0
  episode_steps: 100

reward:
  terminal_weight: 1.0
  intermediate_weight: 0.1
  capture_reward_weight: 0.5
  loss_penalty_weight: 0.3
  defense_reward_weight: 0.2
  transit_cost_weight: 0.01
  production_advantage_weight: 0.1
  comet_roi_weight: 0.2

self_play:
  pool_size: 5
  sample_latest_ratio: 0.4
  sample_random_ratio: 0.2
  sample_best_ratio: 0.3
  sample_heuristic_ratio: 0.1
  add_to_pool_win_rate: 0.4
  diversity_threshold: 0.1
  two_player_prob: 0.5
EOF

python train_ddp.py /tmp/test_quick.yaml

echo "测试完成！"
