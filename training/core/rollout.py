"""Rollout 管理 — 自我博弈数据收集。"""
from __future__ import annotations

import copy
from typing import Optional

import numpy as np
import torch

from core.config import ModelConfig, EnvironmentConfig
from core.env_wrapper import OrbitWarsEnv
from core.feature_engineering import FeatureEngineer
from core.model import OrbitWarsModel
from core.action import decode_actions, sample_actions
from core.reward import RewardCalculator
from core.ppo import PPOBuffer


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

            # ========== 批量推理优化 ==========
            # 第一阶段：批量提取所有玩家特征（CPU）
            all_features = []
            for pid in range(env.num_players):
                obs_dict = env.get_raw_observation(pid)
                planet_feat, fleet_feat, global_feat, metadata = (
                    self.feature_engineer.compute(obs_dict, pid)
                )
                raw_planets = obs_dict.get("planets", [])
                all_features.append({
                    "planet_feat": planet_feat,
                    "fleet_feat": fleet_feat,
                    "global_feat": global_feat,
                    "raw_planets": raw_planets,
                    "player_id": pid,
                })

            # 第二阶段：批量传输到GPU
            max_planets = max(len(f["planet_feat"]) for f in all_features)
            num_players = env.num_players

            batch_planet_feats = []
            batch_fleet_feats = []
            batch_global_feats = []
            batch_owned_masks = []
            batch_enemy_masks = []
            batch_planet_ships = []

            for feat in all_features:
                pid = feat["player_id"]
                raw_planets = feat["raw_planets"]
                n = len(raw_planets)

                # Padding到相同长度
                planet_padded = np.zeros((max_planets, feat["planet_feat"].shape[1]), dtype=np.float32)
                planet_padded[:len(feat["planet_feat"])] = feat["planet_feat"]

                fleet_dim = feat["fleet_feat"].shape[1] if feat["fleet_feat"].size > 0 else 11
                fleet_padded = np.zeros((max_planets, fleet_dim), dtype=np.float32)
                if feat["fleet_feat"].size > 0:
                    fleet_padded[:len(feat["fleet_feat"])] = feat["fleet_feat"]

                # 构建masks
                owned_mask = np.zeros(max_planets, dtype=bool)
                enemy_mask = np.zeros(max_planets, dtype=bool)
                for i, p in enumerate(raw_planets):
                    owner = int(p[1])
                    if owner == pid:
                        owned_mask[i] = True
                    elif owner != -1:
                        enemy_mask[i] = True

                planet_ships = np.array([float(p[5]) for p in raw_planets], dtype=np.float32)
                planet_ships = np.pad(planet_ships, (0, max_planets - len(planet_ships)), constant_values=0)

                batch_planet_feats.append(planet_padded)
                batch_fleet_feats.append(fleet_padded)
                batch_global_feats.append(feat["global_feat"])
                batch_owned_masks.append(owned_mask)
                batch_enemy_masks.append(enemy_mask)
                batch_planet_ships.append(planet_ships)

            # 一次性传输到GPU
            batch_planet_feats_t = torch.from_numpy(np.array(batch_planet_feats)).to(self.device)
            batch_fleet_feats_t = torch.from_numpy(np.array(batch_fleet_feats)).to(self.device)
            batch_global_feats_t = torch.from_numpy(np.array(batch_global_feats)).to(self.device)
            batch_owned_masks_t = torch.from_numpy(np.array(batch_owned_masks)).to(self.device)
            batch_enemy_masks_t = torch.from_numpy(np.array(batch_enemy_masks)).to(self.device)
            batch_planet_ships_t = torch.from_numpy(np.array(batch_planet_ships)).to(self.device)
            batch_num_players = torch.tensor([num_players] * num_players, dtype=torch.long).to(self.device)

            # 第三阶段：批量推理（GPU）- 一次推理所有玩家
            with torch.no_grad():
                batch_target_logits, batch_num_ships_out, batch_values, _, _ = self.model(
                    planet_features=batch_planet_feats_t,
                    fleet_features=batch_fleet_feats_t,
                    global_features=batch_global_feats_t,
                    owned_mask=batch_owned_masks_t,
                    enemy_mask=batch_enemy_masks_t,
                    num_players=batch_num_players,
                    planet_ships=batch_planet_ships_t,
                )

            # 第四阶段：批量解码动作
            all_actions: dict[int, list] = {}
            step_data: dict[int, dict] = {}

            for i, feat in enumerate(all_features):
                pid = feat["player_id"]
                raw_planets = feat["raw_planets"]
                n = len(raw_planets)

                # 获取该玩家的结果
                target_logits = batch_target_logits[i, :n].cpu().numpy()
                num_ships = batch_num_ships_out[i, :n].cpu().numpy()
                value_np = batch_values[i].cpu().numpy()

                # 采样或argmax
                if pid == player_id:
                    target_indices, sampled_ships = sample_actions(
                        target_logits, num_ships, temperature=temperature
                    )
                    log_prob = 0.0  # 简化
                else:
                    target_indices = np.argmax(target_logits, axis=-1)
                    sampled_ships = num_ships.squeeze(-1)
                    log_prob = 0.0

                # 构建动作
                all_planets_dicts = [
                    {"id": int(p[0]), "x": float(p[2]), "y": float(p[3]),
                     "ships": float(p[5]), "owner": int(p[1]), "production": float(p[6])}
                    for p in raw_planets
                ]
                owned_planets_dicts = [p for p in all_planets_dicts if p["owner"] == pid]

                actions = self._indices_to_kaggle_actions(
                    target_indices, sampled_ships, owned_planets_dicts, all_planets_dicts
                )
                all_actions[pid] = actions

                # 记录训练数据（仅训练策略）
                if pid == player_id:
                    step_data[pid] = {
                        "planet_feat": feat["planet_feat"],
                        "fleet_feat": feat["fleet_feat"],
                        "global_feat": feat["global_feat"],
                        "owned_mask": batch_owned_masks[i][:n],
                        "enemy_mask": batch_enemy_masks[i][:n],
                        "num_players": num_players,
                        "planet_ships": batch_planet_ships[i][:n],
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
