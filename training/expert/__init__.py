"""Orbit Wars 专家策略模块。

提供多个高性能专家智能体，用于生成冷启动训练数据。
这些策略基于 Kaggle 竞赛中的高分策略实现。
"""

from training.expert.base_agent import ExpertAgent
from training.expert.kaggle_expert import KaggleExpertAgent

try:
    from training.expert.data_generator import (
        ExpertDataGenerator,
        ExpertDataset,
        print_statistics,
        generate_expert_data,
        load_expert_dataset,
    )
except Exception:
    ExpertDataGenerator = None
    ExpertDataset = None
    print_statistics = None
    generate_expert_data = None
    load_expert_dataset = None

try:
    from training.expert.pretraining import ExpertPretrainer, mix_expert_data_with_rl
except Exception:
    ExpertPretrainer = None
    mix_expert_data_with_rl = None

try:
    from training.expert.ray_pretrainer import RayDistributedPretrainer, PretrainingWorker
except Exception:
    RayDistributedPretrainer = None
    PretrainingWorker = None

__all__ = [
    "ExpertAgent",
    "KaggleExpertAgent",
    "ExpertDataGenerator",
    "ExpertDataset",
    "print_statistics",
    "generate_expert_data",
    "load_expert_dataset",
    "ExpertPretrainer",
    "RayDistributedPretrainer",
    "PretrainingWorker",
    "mix_expert_data_with_rl",
]
