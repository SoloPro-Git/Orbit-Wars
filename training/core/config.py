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
class ExpertDataConfig:
    """专家演示数据配置。"""
    enabled: bool = True  # 是否启用专家数据预训练
    data_dir: str = "data/expert_demonstrations"  # 专家数据目录
    extra_data_dirs: list[str] = field(default_factory=list)  # 额外数据目录（可并行加载）
    num_pretrain_iterations: int = 100  # 预训练迭代次数（最大）
    pretrain_batch_size: int = 256  # 预训练批次大小
    pretrain_learning_rate: float = 1e-4  # 预训练学习率
    behavior_clone_loss_coef: float = 1.0  # 行为克隆损失系数
    mix_expert_data_ratio: float = 0.3  # 在 RL 训练中混合专家数据的比例
    use_until_iteration: int = 100  # 前 N 次迭代使用专家数据（-1 表示一直使用）

    # Ray 分布式配置
    num_pretrain_workers: int = 4  # 预训练 worker 数量
    gpus_per_worker: float = 0.5  # 每个 worker 的 GPU 数量
    sync_interval: int = 10  # 每隔多少个 iter 合并一次参数
    auto_proceed: bool = True  # 评估未达标是否自动继续（True）或停止（False）
    pretrain_resume_from: Optional[str] = None  # 预训练恢复路径（为 None 则从头训练）

    # 专家策略评估门控
    eval_gate_enabled: bool = True  # 启用真实对局评估门控
    eval_gate_interval: int = 100  # 每 N iteration 评估一次
    eval_gate_num_games: int = 20  # 每次评估的对局数
    eval_gate_win_rate_threshold: float = 0.45  # 胜率阈值（>= 此值通过门控）
    eval_gate_device: str = "cpu"  # 评估设备（CPU 不干扰 GPU 训练）
    staged_mode: bool = False  # 是否启用 A/B/C 分阶段单独训练模式
    active_stage: str = "A"  # 当前运行阶段: A/B/C
    stage_eval_interval: int = 10  # 每多少 iter 做一次阶段门控评估
    stage_min_iterations: int = 50  # 最少训练迭代，防止过早停止
    stop_after_pretrain: bool = False  # 预训练后是否直接停止（不进入 RL）
    pretrain_backend: str = "ray"  # ray 或 local（staged_mode 推荐 local）
    stage_a_send_acc_threshold: float = 0.90
    stage_b_ship_mae_threshold: float = 0.12
    stage_c_target_top1_threshold: float = 0.65
    stage_a_num_players: int = 2  # 0=不过滤，2=只用2人局样本，4=只用4人局样本
    stage_b_num_players: int = 2
    stage_c_num_players: int = 4
    stage_a_eval_opponent: str = "sniper"  # sniper/expert/none
    stage_b_eval_opponent: str = "sniper"
    stage_c_eval_opponent: str = "expert"
    stage_a_eval_games: int = 12
    stage_b_eval_games: int = 12
    stage_c_eval_games: int = 20
    stage_a_ratio: float = 0.2  # Stage A: 只训练是否发送
    stage_b_ratio: float = 0.3  # Stage B: 训练发送数量
    stage_c_ratio: float = 0.5  # Stage C: 训练目标分配（实时标签）


@dataclass
class TrainingConfig:
    framework: str = "pytorch"
    tracker: str = "swanlab"
    swanlab_project: str = "orbit-wars"
    swanlab_experiment: str = "ppo-self-play"
    swanlab_mode: str = "cloud"  # cloud 或 local
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
    num_feature_workers: int = 8  # 特征提取工作进程数，-1表示自动设置（CPU核心数//4）
    enable_rollout_timing: bool = False  # 是否启用rollout详细计时统计
    use_gpu_features: bool = False  # 是否使用GPU向量化特征提取（推荐）
    save_interval: int = 50
    max_iterations: int = 10000
    warmup_steps: int = 100
    max_checkpoints: int = 5
    keep_best_n: int = 2
    resume_from_checkpoint: bool = True  # 默认加载最新checkpoint
    output_dir: str = "checkpoints"         # checkpoint输出目录


@dataclass
class SelfPlayConfig:
    pool_size: int = 64
    sample_expert_ratio: float = 0.1
    sample_heuristic_ratio: float = 0.1
    sample_checkpoint_ratio: float = 0.8
    checkpoint_latest_ratio: float = 0.3
    checkpoint_best_ratio: float = 0.3
    checkpoint_random_ratio: float = 0.4
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
class RayConfig:
    """Ray 分布式训练配置。"""
    num_rollout_workers: int = 8          # rollout worker 数量
    num_gpus_per_worker: float = 0.25     # 每个 worker 的 GPU 数量
    trainer_num_gpus: int = 1             # trainer 的 GPU 数量
    games_per_rollout: int = 64           # 每次 rollout 的游戏数
    max_rollout_retries: int = 3          # rollout 失败重试次数


@dataclass
class AppConfig:
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    self_play: SelfPlayConfig = field(default_factory=SelfPlayConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    ray: RayConfig = field(default_factory=RayConfig)
    expert_data: ExpertDataConfig = field(default_factory=ExpertDataConfig)

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
            ray=RayConfig(**data.get("ray", {})),
            expert_data=ExpertDataConfig(**data.get("expert_data", {})),
        )

    @classmethod
    def default(cls) -> "AppConfig":
        return cls()
