"""Configuration management for Orbit Wars RL training."""
from __future__ import annotations

import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelConfig:
    d_model: int = 128
    planet_encoder_layers: int = 3
    fleet_encoder_layers: int = 2
    fusion_layers: int = 2
    nhead: int = 4
    mlp_ratio: int = 4
    dropout: float = 0.1
    use_opponent_head: bool = True
    use_autoregressive_refine: bool = False
    activation: str = "gelu"


@dataclass
class TrainingConfig:
    framework: str = "pytorch"
    tracker: str = "swanlab"
    swanlab_project: str = "orbit-wars"
    swanlab_experiment: str = "ppo-self-play"
    learning_rate: float = 3e-4
    lr_scheduler: str = "cosine"
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    ppo_epochs: int = 4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    opponent_pred_coef: float = 0.1
    intermediate_reward_weight: float = 0.2
    max_grad_norm: float = 0.5
    batch_size: int = 2048
    num_parallel_games: int = 128
    save_interval: int = 50
    max_iterations: int = 10000
    warmup_steps: int = 100


@dataclass
class SelfPlayConfig:
    pool_size: int = 20
    sample_latest_ratio: float = 0.4
    sample_random_ratio: float = 0.3
    sample_best_ratio: float = 0.2
    sample_heuristic_ratio: float = 0.1
    two_player_prob: float = 0.3
    add_to_pool_win_rate: float = 0.45
    diversity_threshold: float = 0.3


@dataclass
class RewardConfig:
    terminal_weight: float = 1.0
    intermediate_weight: float = 0.2
    capture_reward_weight: float = 0.3
    loss_penalty_weight: float = 0.2
    defense_reward_weight: float = 0.15
    transit_cost_weight: float = 0.1
    production_advantage_weight: float = 0.15
    comet_roi_weight: float = 0.1


@dataclass
class EnvironmentConfig:
    episode_steps: int = 500
    act_timeout: float = 1.0
    ship_speed: float = 6.0
    sun_radius: float = 10.0
    board_size: float = 100.0
    comet_speed: float = 4.0


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "AppConfig":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            self_play=SelfPlayConfig(**data.get("self_play", {})),
            reward=RewardConfig(**data.get("reward", {})),
            environment=EnvironmentConfig(**data.get("environment", {})),
        )

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()
