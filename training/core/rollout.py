"""Rollout 管理 — 自我博弈数据收集。"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from training.core.config import ModelConfig, EnvironmentConfig
from training.core.env_wrapper import OrbitWarsEnv
from training.core.feature_engineering import FeatureEngineer
from training.core.model import OrbitWarsModel
from training.core.action import decode_actions, sample_actions
from training.core.reward import RewardCalculator
from training.core.ppo import PPOBuffer


class RolloutWorker:
    """跑自我博弈并收集训练数据。"""

    def __init__(
        self,
        model: OrbitWarsModel,
        feature_engineer: FeatureEngineer,
        reward_calculator: RewardCalculator,
        device: str = "cuda",
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.reward_calculator = reward_calculator
        self.device = device

    def rollout_game(
        self,
        env: OrbitWarsEnv,
        player_id: int = 0,
        temperature: float = 1.0,
        max_steps: int = 500,
    ) -> PPOBuffer:
        """跑一局游戏，收集训练数据。

        player_id: 训练策略对应的玩家 ID，其他位置用同一模型 self-play。
        """
        buffer = PPOBuffer()
        self.model.eval()

        observations = env.reset()
        raw_obs = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

        for step in range(max_steps):
            if env.done:
                break

            # 为每个玩家选择动作
            all_actions: dict[int, list] = {}
            step_data: dict[int, dict] = {}

            for pid in range(env.num_players):
                obs_dict = env.get_raw_observation(pid)

                # 提取特征
                planet_feat, fleet_feat, global_feat, metadata = (
                    self.feature_engineer.compute(obs_dict, pid)
                )

                # 构建模型输入
                planet_feat_t = torch.from_numpy(planet_feat).unsqueeze(0).to(self.device)

                if fleet_feat.shape[0] > 0:
                    fleet_feat_t = torch.from_numpy(fleet_feat).unsqueeze(0).to(self.device)
                else:
                    fleet_feat_t = torch.zeros(1, 0, fleet_feat.shape[1] if fleet_feat.ndim == 2 else 11, device=self.device)

                global_feat_t = torch.from_numpy(global_feat).unsqueeze(0).to(self.device)

                # 构建 masks
                raw_planets = obs_dict.get("planets", [])
                n_planets = len(raw_planets)
                owned_mask = torch.zeros(1, n_planets, dtype=torch.bool, device=self.device)
                enemy_mask = torch.zeros(1, n_planets, dtype=torch.bool, device=self.device)

                for i, p in enumerate(raw_planets):
                    owner = int(p[1])
                    if owner == pid:
                        owned_mask[0, i] = True
                    elif owner != -1:
                        enemy_mask[0, i] = True

                num_players = torch.tensor([env.num_players], dtype=torch.long, device=self.device)
                planet_ships = torch.tensor(
                    [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=self.device
                )

                with torch.no_grad():
                    target_logits, num_ships_out, value, _, _ = self.model(
                        planet_features=planet_feat_t,
                        fleet_features=fleet_feat_t,
                        global_features=global_feat_t,
                        owned_mask=owned_mask,
                        enemy_mask=enemy_mask,
                        num_players=num_players,
                        planet_ships=planet_ships,
                    )

                target_logits_np = target_logits[0].cpu().numpy()  # [N_owned, N_planets]
                num_ships_np = num_ships_out[0].cpu().numpy()  # [N_owned, 1]
                value_np = value[0].cpu().numpy()

                # 采样或 argmax
                if pid == player_id:
                    target_indices, sampled_ships = sample_actions(
                        target_logits_np, num_ships_np, temperature=temperature
                    )
                    log_prob = self._compute_log_prob_from_sample(
                        target_logits[0], num_ships_out[0],
                        torch.from_numpy(target_indices).to(self.device),
                        torch.from_numpy(sampled_ships).float().to(self.device),
                        owned_mask[0],
                    ).item()
                else:
                    target_indices = np.argmax(target_logits_np, axis=-1)
                    sampled_ships = num_ships_np.squeeze(-1)
                    log_prob = 0.0

                # 转为 kaggle 动作
                all_planets_dicts = [
                    {"id": int(p[0]), "x": float(p[2]), "y": float(p[3]),
                     "ships": float(p[5]), "owner": int(p[1]), "production": float(p[6])}
                    for p in raw_planets
                ]
                owned_planets_dicts = [p for p in all_planets_dicts if p["owner"] == pid]

                # 从 target_indices 和 sampled_ships 构建动作
                actions = self._indices_to_kaggle_actions(
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
                        "num_players": env.num_players,
                        "planet_ships": planet_ships[0].cpu().numpy(),
                        "target_indices": target_indices,
                        "num_ships_actual": sampled_ships,
                        "log_prob": log_prob,
                        "value": float(value_np.mean()),
                    }

            # 保存 obs_before 用于 reward 计算
            obs_before = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

            # 执行 step
            observations, rewards, dones, infos = env.step(all_actions)
            done = env.done

            # 计算 reward
            obs_after = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

            if player_id in step_data:
                sd = step_data[player_id]
                reward = self.reward_calculator.compute(
                    obs_before[player_id],
                    obs_after[player_id],
                    player_id,
                    all_actions.get(player_id, []),
                    done,
                )

                buffer.add(
                    planet_feat=sd["planet_feat"],
                    fleet_feat=sd["fleet_feat"],
                    global_feat=sd["global_feat"],
                    owned_mask=sd["owned_mask"],
                    enemy_mask=sd["enemy_mask"],
                    num_players=sd["num_players"],
                    planet_ships=sd["planet_ships"],
                    target_idx=sd["target_indices"],
                    num_ships_act=sd["num_ships_actual"],
                    log_prob=sd["log_prob"],
                    value=sd["value"],
                    reward=reward,
                    done=done,
                )

        self.model.train()
        return buffer

    def rollout_self_play(
        self,
        num_games: int = 1,
        num_players: int = 4,
        temperature: float = 1.0,
        env_config: dict | None = None,
    ) -> PPOBuffer:
        """跑多局自我博弈。"""
        combined_buffer = PPOBuffer()

        for game_idx in range(num_games):
            env = OrbitWarsEnv(num_players=num_players, config=env_config)
            game_buffer = self.rollout_game(
                env, player_id=0, temperature=temperature
            )

            # 合并 buffer
            for attr in [
                "planet_features", "fleet_features", "global_features",
                "owned_masks", "enemy_masks", "num_players_list",
                "planet_ships_list", "target_indices", "num_ships_actual",
                "log_probs", "values", "rewards", "dones",
                "opp_target_indices", "opp_num_ships_actual",
            ]:
                src = getattr(game_buffer, attr)
                dst = getattr(combined_buffer, attr)
                dst.extend(src)

        return combined_buffer

    def _indices_to_kaggle_actions(
        self,
        target_indices: np.ndarray,
        num_ships: np.ndarray,
        owned_planets: list[dict],
        all_planets: list[dict],
        threshold: float = 1.0,
    ) -> list[list]:
        """将 target indices + num_ships 转为 kaggle 动作格式。"""
        import math
        actions = []
        for i, src in enumerate(owned_planets):
            if i >= len(target_indices) or i >= len(num_ships):
                break
            ships = float(num_ships[i]) if num_ships.ndim == 1 else float(num_ships[i, 0])
            if ships < threshold or src["ships"] <= 0:
                continue

            ships = min(int(max(ships, 1)), int(src["ships"]))

            tgt_idx = int(target_indices[i])
            if tgt_idx >= len(all_planets):
                continue
            tgt = all_planets[tgt_idx]
            angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
            actions.append([src["id"], angle, ships])
        return actions

    @staticmethod
    def _compute_log_prob_from_sample(
        target_logits: torch.Tensor,
        num_ships_pred: torch.Tensor,
        target_indices: torch.Tensor,
        num_ships_actual: torch.Tensor,
        owned_mask: torch.Tensor,
    ) -> torch.Tensor:
        """计算采样动作的 log probability。"""
        import torch.nn.functional as F

        log_softmax = F.log_softmax(target_logits, dim=-1)
        n_planets = target_logits.shape[-1]
        target_log_prob = log_softmax.gather(
            1, target_indices.clamp(0, n_planets - 1).unsqueeze(-1)
        ).squeeze(-1)

        sigma = 0.1
        ships_log_prob = -0.5 * ((num_ships_actual - num_ships_pred.squeeze(-1)) / sigma).pow(2)

        valid = (~owned_mask[:target_logits.shape[0]]).float()
        log_prob = (target_log_prob + ships_log_prob) * valid
        return log_prob.sum() / valid.sum().clamp(min=1)


def parallel_rollout(
    worker: RolloutWorker,
    num_games: int,
    num_players: int = 4,
    temperature: float = 1.0,
    env_config: dict | None = None,
) -> PPOBuffer:
    """并行 rollout（单进程版本，后续可扩展为多进程）。"""
    return worker.rollout_self_play(
        num_games=num_games,
        num_players=num_players,
        temperature=temperature,
        env_config=env_config,
    )
