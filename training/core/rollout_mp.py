"""多进程并行 Rollout Worker - 利用多核CPU加速数据收集。"""
from __future__ import annotations

import copy
from typing import Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np
import torch

from core.config import ModelConfig, EnvironmentConfig
from core.env_wrapper import OrbitWarsEnv
from core.feature_engineering import FeatureEngineer
from core.model import OrbitWarsModel
from core.action import decode_actions, sample_actions
from core.reward import RewardCalculator
from core.ppo import PPOBuffer


def _rollout_single_game_worker(args):
    """单个游戏的工作函数（在子进程中运行）。

    Args:
        args: (game_id, num_players, temperature, max_steps, player_id,
               serialized_model_state, feature_engineer_config, reward_config)

    Returns:
        (game_id, buffer_data)
    """
    import sys
    sys.path.insert(0, '/data2/solo/Orbit-Wars/training')

    from core.env_wrapper import OrbitWarsEnv
    from core.feature_engineering import FeatureEngineer
    from core.model import OrbitWarsModel
    from core.reward import RewardCalculator, RewardConfig
    from core.action import sample_actions
    import torch

    game_id, num_players, temperature, max_steps, player_id, \
    model_state_dict, feature_engineer_config, reward_config = args

    # 初始化子进程的模型和环境
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # 重建模型
    from core.config import ModelConfig
    model = OrbitWarsModel(
        ModelConfig(d_model=128, nhead=4, mlp_ratio=4,
                   planet_encoder_layers=3, fleet_encoder_layers=2,
                   fusion_layers=2, dropout=0.1, activation="gelu",
                   use_opponent_head=True),
        n_planets=40,
        max_players=num_players,
    ).to(device)

    model.load_state_dict(model_state_dict)
    model.eval()

    # 重建特征工程和奖励计算器
    feature_engineer = FeatureEngineer(
        board_size=feature_engineer_config['board_size'],
        sun_radius=feature_engineer_config['sun_radius'],
        max_speed=feature_engineer_config['max_speed'],
        max_turns=feature_engineer_config['max_turns'],
    )

    reward_calculator = RewardCalculator(
        config=RewardConfig(**reward_config),
        max_turns=feature_engineer_config['max_turns'],
    )

    # 运行游戏
    env = OrbitWarsEnv(num_players=num_players)
    observations = env.reset()
    buffer_data = {
        'planet_features': [],
        'fleet_features': [],
        'global_features': [],
        'owned_masks': [],
        'enemy_masks': [],
        'num_players_list': [],
        'planet_ships_list': [],
        'target_indices': [],
        'num_ships_actual': [],
        'log_probs': [],
        'values': [],
        'rewards': [],
        'dones': [],
    }

    for step in range(max_steps):
        if env.done:
            break

        # 为每个玩家选择动作
        all_actions = {}
        step_data = {}

        for pid in range(num_players):
            obs_dict = env.get_raw_observation(pid)

            # 提取特征
            planet_feat, fleet_feat, global_feat, _ = feature_engineer.compute(obs_dict, pid)

            # 构建模型输入
            planet_feat_t = torch.from_numpy(planet_feat).unsqueeze(0).to(device)
            if fleet_feat.shape[0] > 0:
                fleet_feat_t = torch.from_numpy(fleet_feat).unsqueeze(0).to(device)
            else:
                fleet_feat_t = torch.zeros(1, 0, 11, device=device)

            global_feat_t = torch.from_numpy(global_feat).unsqueeze(0).to(device)

            # 构建masks
            raw_planets = obs_dict.get("planets", [])
            n = len(raw_planets)
            owned_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
            enemy_mask = torch.zeros(1, n, dtype=torch.bool, device=device)

            for i, p in enumerate(raw_planets):
                owner = int(p[1])
                if owner == pid:
                    owned_mask[0, i] = True
                elif owner != -1:
                    enemy_mask[0, i] = True

            num_players_t = torch.tensor([num_players], dtype=torch.long, device=device)
            planet_ships = torch.tensor(
                [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=device
            )

            with torch.no_grad():
                target_logits, num_ships_out, value, _, _ = model(
                    planet_features=planet_feat_t,
                    fleet_features=fleet_feat_t,
                    global_features=global_feat_t,
                    owned_mask=owned_mask,
                    enemy_mask=enemy_mask,
                    num_players=num_players_t,
                    planet_ships=planet_ships,
                )

            target_logits_np = target_logits[0].cpu().numpy()
            num_ships_np = num_ships_out[0].cpu().numpy()
            value_np = value[0].cpu().numpy()

            # 采样或argmax
            if pid == player_id:
                target_indices, sampled_ships = sample_actions(
                    target_logits_np, num_ships_np, temperature=temperature
                )
                log_prob = 0.0
            else:
                target_indices = np.argmax(target_logits_np, axis=-1)
                sampled_ships = num_ships_np.squeeze(-1)
                log_prob = 0.0

            # 构建动作
            all_planets_dicts = [
                {"id": int(p[0]), "x": float(p[2]), "y": float(p[3]),
                 "ships": float(p[5]), "owner": int(p[1]), "production": float(p[6])}
                for p in raw_planets
            ]
            owned_planets_dicts = [p for p in all_planets_dicts if p["owner"] == pid]

            actions = _indices_to_kaggle_actions(
                target_indices, sampled_ships, owned_planets_dicts, all_planets_dicts
            )
            all_actions[pid] = actions

            # 记录训练数据（仅训练策略）
            if pid == player_id:
                step_data[pid] = {
                    "planet_feat": planet_feat,
                    "fleet_feat": fleet_feat,
                    "global_feat": global_feat,
                    "owned_mask": owned_mask[0].cpu().numpy(),
                    "enemy_mask": enemy_mask[0].cpu().numpy(),
                    "num_players": num_players,
                    "planet_ships": planet_ships[0].cpu().numpy(),
                    "target_indices": target_indices,
                    "num_ships_actual": sampled_ships,
                    "log_prob": log_prob,
                    "value": float(value_np.mean()),
                }

        # 执行step
        obs_before = {pid: env.get_raw_observation(pid) for pid in range(num_players)}
        observations, rewards, dones, infos = env.step(all_actions)
        done = env.done
        obs_after = {pid: env.get_raw_observation(pid) for pid in range(num_players)}

        # 计算reward
        if player_id in step_data:
            sd = step_data[player_id]
            reward = reward_calculator.compute(
                obs_before[player_id],
                obs_after[player_id],
                player_id,
                all_actions.get(player_id, []),
                done,
            )

            # 添加到buffer
            buffer_data['planet_features'].append(sd["planet_feat"])
            buffer_data['fleet_features'].append(sd["fleet_feat"])
            buffer_data['global_features'].append(sd["global_feat"])
            buffer_data['owned_masks'].append(sd["owned_mask"])
            buffer_data['enemy_masks'].append(sd["enemy_mask"])
            buffer_data['num_players_list'].append(sd["num_players"])
            buffer_data['planet_ships_list'].append(sd["planet_ships"])
            buffer_data['target_indices'].append(sd["target_indices"])
            buffer_data['num_ships_actual'].append(sd["num_ships_actual"])
            buffer_data['log_probs'].append(sd["log_prob"])
            buffer_data['values'].append(sd["value"])
            buffer_data['rewards'].append(reward)
            buffer_data['dones'].append(done)

    return (game_id, buffer_data)


def _indices_to_kaggle_actions(
    target_indices: np.ndarray,
    num_ships: np.ndarray,
    owned_planets: list[dict],
    all_planets: list[dict],
) -> list[list]:
    """将模型输出转换为 Kaggle 环境动作格式。"""
    actions = []

    for i, src in enumerate(owned_planets):
        if i >= len(target_indices):
            break

        target_idx = target_indices[i]
        if target_idx < 0 or target_idx >= len(all_planets):
            continue

        ships = num_ships[i] if i < len(num_ships) else 0
        if ships <= 0:
            continue

        target_planet = all_planets[target_idx]
        if not target_planet:
            continue

        angle = np.arctan2(
            target_planet["y"] - src["y"],
            target_planet["x"] - src["x"]
        )

        actions.append([src["id"], float(angle), float(ships)])

    return actions


class MultiProcessRolloutWorker:
    """多进程并行 Rollout Worker - 充分利用多核CPU。"""

    def __init__(
        self,
        model: OrbitWarsModel,
        feature_engineer: FeatureEngineer,
        reward_calculator: RewardCalculator,
        device: str = "cuda",
        num_workers: int = None,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.reward_calculator = reward_calculator
        self.device = device

        # 默认使用CPU核心数的50%（留一些给主进程和其他任务）
        if num_workers is None:
            num_workers = max(1, cpu_count() // 2)

        self.num_workers = num_workers
        print(f"[MultiProcessRolloutWorker] 使用 {num_workers} 个worker进程")

    def rollout_self_play(
        self,
        num_games: int = 1,
        num_players: int = 4,
        temperature: float = 1.0,
        max_steps: int = 200,
    ) -> PPOBuffer:
        """多进程并行运行多个游戏。

        预期加速：~8x（使用64个worker）
        """
        import time
        start = time.time()

        # 准备参数
        player_id = 0  # 训练player 0

        # 获取模型状态字典（用于传递给子进程）
        model_state_dict = self.model.state_dict()

        # 特征工程配置
        feature_engineer_config = {
            'board_size': self.feature_engineer.board_size,
            'sun_radius': self.feature_engineer.sun_radius,
            'max_speed': self.feature_engineer.max_speed,
            'max_turns': self.feature_engineer.max_turns,
        }

        # 奖励配置
        from core.reward import RewardConfig
        reward_config = self.reward_calculator.config.__dict__.copy()

        # 准备所有游戏的参数
        all_args = [
            (game_id, num_players, temperature, max_steps, player_id,
             model_state_dict, feature_engineer_config, reward_config)
            for game_id in range(num_games)
        ]

        # 使用进程池并行运行
        with Pool(processes=self.num_workers) as pool:
            results = pool.map(_rollout_single_game_worker, all_args)

        # 合并所有buffer
        merged_buffer = PPOBuffer()

        for game_id, buffer_data in results:
            for i in range(len(buffer_data['rewards'])):
                merged_buffer.add(
                    planet_feat=buffer_data['planet_features'][i],
                    fleet_feat=buffer_data['fleet_features'][i],
                    global_feat=buffer_data['global_features'][i],
                    owned_mask=buffer_data['owned_masks'][i],
                    enemy_mask=buffer_data['enemy_masks'][i],
                    num_players=buffer_data['num_players_list'][i],
                    planet_ships=buffer_data['planet_ships_list'][i],
                    target_idx=buffer_data['target_indices'][i],
                    num_ships_act=buffer_data['num_ships_actual'][i],
                    log_prob=buffer_data['log_probs'][i],
                    value=buffer_data['values'][i],
                    reward=buffer_data['rewards'][i],
                    done=buffer_data['dones'][i],
                )

        elapsed = time.time() - start
        throughput = len(merged_buffer) / elapsed if elapsed > 0 else 0

        print(f"[MultiProcessRolloutWorker] 完成 {num_games} 局游戏，"
              f"收集 {len(merged_buffer)} 个样本，"
              f"耗时 {elapsed:.2f}s，"
              f"吞吐量 {throughput:.0f} samples/s")

        return merged_buffer


# 导出函数
def parallel_rollout_mp(
    worker,
    num_games: int,
    num_players: int,
    temperature: float,
) -> PPOBuffer:
    """多进程并行rollout入口。"""
    # 如果worker是MultiProcessRolloutWorker，直接使用
    if isinstance(worker, MultiProcessRolloutWorker):
        return worker.rollout_self_play(
            num_games=num_games,
            num_players=num_players,
            temperature=temperature,
            max_steps=200,
        )

    # 否则回退到原始worker
    from core.rollout import RolloutWorker
    if isinstance(worker, RolloutWorker):
        return worker.rollout_self_play(
            num_games=num_games,
            num_players=num_players,
            temperature=temperature,
        )

    raise ValueError(f"Unknown worker type: {type(worker)}")
