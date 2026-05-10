"""Rollout 管理 — 自我博弈数据收集。"""
from __future__ import annotations

import copy
import time
from typing import Optional
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
import multiprocessing

import numpy as np
import torch

from core.config import ModelConfig, EnvironmentConfig
from core.env_wrapper import OrbitWarsEnv
from core.feature_engineering import FeatureEngineer
from core.model import OrbitWarsModel
from core.action import decode_actions, sample_actions
from core.reward import RewardCalculator
from core.ppo import PPOBuffer


def _extract_features_worker(args):
    """工作线程中执行的特征提取函数。"""
    obs, pid, board_size, sun_radius, max_speed, max_turns = args

    # 在工作线程中创建 FeatureEngineer（线程安全的）
    from core.feature_engineering import FeatureEngineer
    feature_engineer = FeatureEngineer(
        board_size=board_size,
        sun_radius=sun_radius,
        max_speed=max_speed,
        max_turns=max_turns,
    )

    pf, ff, gf, metadata = feature_engineer.compute(obs, pid)
    return pf, ff, gf, obs.get("planets", []), pid, metadata


class RolloutWorker:
    """跑自我博弈并收集训练数据。

    支持CPU多进程或GPU向量化特征提取。
    """

    def __init__(
        self,
        model: OrbitWarsModel,
        feature_engineer: FeatureEngineer,
        reward_calculator: RewardCalculator,
        device: str = "cuda",
        num_feature_workers: int = -1,
        enable_rollout_timing: bool = False,
        use_gpu_features: bool = False,
    ):
        self.model = model
        self.feature_engineer = feature_engineer
        self.reward_calculator = reward_calculator
        self.device = device
        self.enable_rollout_timing = enable_rollout_timing
        self.use_gpu_features = use_gpu_features

        # GPU特征提取器
        if self.use_gpu_features:
            from core.feature_engineering_gpu import FeatureEngineerGPU
            self.feature_engineer_gpu = FeatureEngineerGPU(
                board_size=feature_engineer.board_size,
                sun_radius=feature_engineer.sun_radius,
                max_speed=feature_engineer.max_speed,
                max_turns=feature_engineer.max_turns,
                device=device,
            )
            print(f"[RolloutWorker] 使用GPU向量化特征提取")
            self.feature_pool = None
        else:
            # 创建线程池用于并行特征提取
            if num_feature_workers == -1:
                self.num_feature_workers = max(4, multiprocessing.cpu_count() // 8)
            else:
                self.num_feature_workers = num_feature_workers
            self.feature_pool = ThreadPool(processes=self.num_feature_workers)

            # 保存特征工程配置参数，用于传递给工作线程
            self.feature_engine_config = {
                "board_size": feature_engineer.board_size,
                "sun_radius": feature_engineer.sun_radius,
                "max_speed": feature_engineer.max_speed,
                "max_turns": feature_engineer.max_turns,
            }

            print(f"[RolloutWorker] 使用 {self.num_feature_workers} 线程并行提取特征")

    def rollout_game(
        self,
        env: OrbitWarsEnv,
        player_id: int = 0,
        temperature: float = 1.0,
        max_steps: int = 500,
        opponent_agents: dict[int, Any] | None = None,
    ) -> PPOBuffer:
        """跑一局游戏，收集训练数据。

        player_id: 训练策略对应的玩家 ID。
        opponent_agents: {pid: agent_fn} 非训练位置的对手 agent。
            agent_fn 签名: (obs: dict) -> list[list]  (kaggle action 格式)
            如果为 None，则所有位置都用当前模型 self-play。
        """
        buffer = PPOBuffer()
        self.model.eval()

        # 计时统计
        total_feature_time = 0.0
        total_transfer_time = 0.0
        total_inference_time = 0.0
        total_decode_time = 0.0
        total_step_time = 0.0
        total_steps = 0

        observations = env.reset()
        raw_obs = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

        # 预分配缓冲区用于批量特征提取（避免频繁的进程间通信）
        BATCH_STEPS = 32  # 每次批量处理 32 步的特征提取

        for step in range(max_steps):
            if env.done:
                break

            # 每100步打印一次进度（调试用）
            # if step > 0 and step % 100 == 0:
            #     print(f"[Rollout] Step {step}/{max_steps}...", flush=True)

            # ========== 批量推理优化 ==========
            # 第一阶段：批量提取所有玩家特征
            t_feature_start = time.time()
            obs_list = [env.get_raw_observation(pid) for pid in range(env.num_players)]
            pid_list = list(range(env.num_players))

            if self.use_gpu_features:
                # GPU向量化特征提取（零拷贝，已在GPU上）
                all_features = []
                for obs, pid in zip(obs_list, pid_list):
                    planet_feat, fleet_feat, global_feat, metadata = self.feature_engineer_gpu.compute(
                        obs, pid
                    )
                    # 已经是GPU tensor，无需传输
                    all_features.append({
                        "planet_feat": planet_feat,
                        "fleet_feat": fleet_feat,
                        "global_feat": global_feat,
                        "raw_planets": obs.get("planets", []),
                        "player_id": pid,
                        "metadata": metadata,
                        "owned_indices": metadata.get("owned_planet_indices", []),
                    })
            else:
                # CPU多进程并行特征提取
                worker_args = [
                    (obs, pid,
                     self.feature_engine_config["board_size"],
                     self.feature_engine_config["sun_radius"],
                     self.feature_engine_config["max_speed"],
                     self.feature_engine_config["max_turns"])
                    for obs, pid in zip(obs_list, pid_list)
                ]

                # 批量提交任务并等待完成（比逐个 submit 更高效）
                results = list(self.feature_pool.map(_extract_features_worker, worker_args))

                # 组装特征
                all_features = []
                for planet_feat, fleet_feat, global_feat, raw_planets, pid, metadata in results:
                    all_features.append({
                        "planet_feat": planet_feat,
                        "fleet_feat": fleet_feat,
                        "global_feat": global_feat,
                        "raw_planets": raw_planets,
                        "player_id": pid,
                        "metadata": metadata,
                        "owned_indices": metadata.get("owned_planet_indices", []),
                    })

            feature_time = time.time() - t_feature_start
            total_feature_time += feature_time

            # 第二阶段：批量传输到GPU（或直接使用GPU tensor）
            t_transfer_start = time.time()
            max_planets = max(len(f["planet_feat"]) for f in all_features)
            max_fleets = max(
                len(f["fleet_feat"]) if f["fleet_feat"].shape[0] > 0 else 0
                for f in all_features
            )
            num_players = env.num_players

            if self.use_gpu_features:
                # GPU特征：已经在GPU上，只需padding
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

                    # GPU tensor padding
                    planet_feat = feat["planet_feat"]  # [N, D]
                    fleet_feat = feat["fleet_feat"]    # [M, D] 或 [0, D]

                    # Padding到相同长度
                    planet_padded = torch.zeros(
                        max_planets, planet_feat.shape[1],
                        dtype=torch.float32, device=self.device
                    )
                    planet_padded[:planet_feat.shape[0]] = planet_feat

                    if fleet_feat.shape[0] > 0:
                        fleet_padded = torch.zeros(
                            max_fleets, fleet_feat.shape[1],
                            dtype=torch.float32, device=self.device
                        )
                        fleet_padded[:fleet_feat.shape[0]] = fleet_feat
                    else:
                        fleet_padded = torch.zeros(
                            max_fleets, 11,
                            dtype=torch.float32, device=self.device
                        )

                    # 构建masks
                    owned_mask = torch.zeros(max_planets, dtype=torch.bool, device=self.device)
                    enemy_mask = torch.zeros(max_planets, dtype=torch.bool, device=self.device)
                    for i, p in enumerate(raw_planets):
                        owner = int(p[1])
                        if owner == pid:
                            owned_mask[i] = True
                        elif owner != -1:
                            enemy_mask[i] = True

                    planet_ships = torch.tensor(
                        [float(p[5]) for p in raw_planets],
                        dtype=torch.float32, device=self.device
                    )
                    planet_ships_padded = torch.zeros(max_planets, dtype=torch.float32, device=self.device)
                    planet_ships_padded[:len(planet_ships)] = planet_ships

                    batch_planet_feats.append(planet_padded)
                    batch_fleet_feats.append(fleet_padded)
                    batch_global_feats.append(feat["global_feat"])
                    batch_owned_masks.append(owned_mask)
                    batch_enemy_masks.append(enemy_mask)
                    batch_planet_ships.append(planet_ships_padded)

                # Stack为batch（已经在GPU上）
                batch_planet_feats_t = torch.stack(batch_planet_feats, dim=0)
                batch_fleet_feats_t = torch.stack(batch_fleet_feats, dim=0)
                batch_global_feats_t = torch.stack(batch_global_feats, dim=0)
                batch_owned_masks_t = torch.stack(batch_owned_masks, dim=0)
                batch_enemy_masks_t = torch.stack(batch_enemy_masks, dim=0)
                batch_planet_ships_t = torch.stack(batch_planet_ships, dim=0)
            else:
                # CPU特征：需要传输到GPU
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
                    fleet_padded = np.zeros((max_fleets, fleet_dim), dtype=np.float32)
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

            # num_players tensor (两种模式都需要)
            batch_num_players = torch.tensor([num_players] * num_players, dtype=torch.long).to(self.device)

            # 传输完成，计算耗时
            transfer_time = time.time() - t_transfer_start
            total_transfer_time += transfer_time

            # 第三阶段：批量推理（GPU）- 一次推理所有玩家
            t_inference_start = time.time()
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

            # 推理完成，计算耗时
            inference_time = time.time() - t_inference_start
            total_inference_time += inference_time

            # 第四阶段：批量解码动作
            t_decode_start = time.time()
            all_actions: dict[int, list] = {}
            step_data: dict[int, dict] = {}

            for i, feat in enumerate(all_features):
                pid = feat["player_id"]
                raw_planets = feat["raw_planets"]
                n = len(raw_planets)

                # 如果有外部对手 agent，直接调用，不经过模型推理
                if opponent_agents and pid in opponent_agents:
                    opp_obs = env.get_raw_observation(pid)
                    try:
                        opp_actions = opponent_agents[pid](opp_obs)
                    except Exception:
                        opp_actions = []
                    all_actions[pid] = opp_actions
                    continue

                # 获取该玩家的结果
                # 重要：模型输出中，只有前 N_owned 行是有效的（后面是 padding）
                # 需要使用 owned_mask 找到实际的拥有星球数量
                mask_item = batch_owned_masks[i]
                if isinstance(mask_item, np.ndarray):
                    owned_mask_np = mask_item
                else:
                    owned_mask_np = mask_item.cpu().numpy()

                n_owned = int(owned_mask_np.sum())  # 实际拥有星球数

                if n_owned > 0:
                    target_logits = batch_target_logits[i, :n_owned].cpu().numpy()
                    num_ships = batch_num_ships_out[i, :n_owned].cpu().numpy()
                else:
                    # 没有拥有星球，跳过
                    continue

                value_np = batch_values[i].cpu().numpy()

                # 采样或argmax
                if pid == player_id:
                    target_indices, sampled_ships, sampled_log_probs = sample_actions(
                        target_logits, num_ships, temperature=temperature, return_log_probs=True
                    )
                    log_prob = float(np.mean(sampled_log_probs))
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

                # 检查一致性
                if len(owned_planets_dicts) != n_owned:
                    # 如果不一致，使用前 n_owned 个拥有的星球
                    owned_planets_dicts = owned_planets_dicts[:n_owned]

                actions = self._indices_to_kaggle_actions(
                    target_indices, sampled_ships, owned_planets_dicts, all_planets_dicts, target_logits=target_logits
                )
                all_actions[pid] = actions

                # 记录训练数据（仅训练策略）
                if pid == player_id:
                    # 关键：保留所有星球的特征，模型需要看到所有信息来判断敌人行为
                    n = len(raw_planets)
                    n_owned = len(target_indices)

                    if n_owned > 0:
                        step_data[pid] = {
                            "planet_feat": feat["planet_feat"],  # 所有星球 [N_all, D]
                            "fleet_feat": feat["fleet_feat"],
                            "global_feat": feat["global_feat"],
                            "owned_mask": batch_owned_masks[i][:n],  # 所有星球的 mask [N_all]
                            "enemy_mask": batch_enemy_masks[i][:n],
                            "num_players": num_players,
                            "planet_ships": batch_planet_ships[i][:n],
                            "target_indices": target_indices,  # 拥有星球的行动 [N_owned]
                            "num_ships_actual": sampled_ships,  # sigmoid 比例 [0,1]
                            "log_prob": log_prob,               # 真实计算的 log probability
                            "value": float(value_np),           # 标量价值
                        }
                    else:
                        # 没有拥有星球，跳过
                        step_data[pid] = None

            # 保存 obs_before 用于 reward 计算
            obs_before = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

            # 计算解码耗时
            decode_time = time.time() - t_decode_start
            total_decode_time += decode_time

            # 执行 step
            t_step_start = time.time()
            observations, rewards, dones, infos = env.step(all_actions)
            done = env.done
            step_time = time.time() - t_step_start
            total_step_time += step_time

            total_steps += 1

            # 计算 reward
            obs_after = {pid: env.get_raw_observation(pid) for pid in range(env.num_players)}

            if player_id in step_data and step_data[player_id] is not None:
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

        # 打印计时统计（仅在启用时）
        if self.enable_rollout_timing and total_steps > 0:
            total_time = total_feature_time + total_transfer_time + total_inference_time + total_decode_time + total_step_time
            print(f"[Rollout] 计时统计 (共 {total_steps} 步):", flush=True)
            print(f"  特征提取: {total_feature_time:.3f}s ({total_feature_time/total_time*100:.1f}%)", flush=True)
            print(f"  数据传输: {total_transfer_time:.3f}s ({total_transfer_time/total_time*100:.1f}%)", flush=True)
            print(f"  GPU推理:   {total_inference_time:.3f}s ({total_inference_time/total_time*100:.1f}%)", flush=True)
            print(f"  动作解码: {total_decode_time:.3f}s ({total_decode_time/total_time*100:.1f}%)", flush=True)
            print(f"  环境步:   {total_step_time:.3f}s ({total_step_time/total_time*100:.1f}%)", flush=True)
            print(f"  总计:     {total_time:.3f}s", flush=True)
            print(f"  每步平均: {total_time/total_steps:.4f}s", flush=True)

        return buffer

    def rollout_self_play(
        self,
        num_games: int = 1,
        num_players: int = 4,
        temperature: float = 1.0,
        env_config: dict | None = None,
        show_progress: bool = True,
        opponent_pool: Any | None = None,
    ) -> PPOBuffer:
        """跑多局自我博弈。"""
        combined_buffer = PPOBuffer()

        game_iter = range(num_games)
        if show_progress:
            game_iter = tqdm(game_iter, desc="[Rollout] ", unit="game")

        for game_idx in game_iter:
            env = OrbitWarsEnv(num_players=num_players, config=env_config)

            # 为这局采样对手
            opponents = None
            if opponent_pool is not None:
                opponents = opponent_pool.sample_opponents_for_game(
                    num_players, training_player=0
                )

            game_buffer = self.rollout_game(
                env, player_id=0, temperature=temperature,
                opponent_agents=opponents,
            )

            # 更新进度条信息
            if show_progress and isinstance(game_iter, tqdm):
                game_iter.set_postfix({
                    'steps': len(game_buffer),
                    'players': num_players,
                    'total': len(combined_buffer)
                })

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

        if show_progress and isinstance(game_iter, tqdm):
            game_iter.close()

        return combined_buffer

    def _indices_to_kaggle_actions(
        self,
        target_indices: np.ndarray,
        num_ships_ratio: np.ndarray,
        owned_planets: list[dict],
        all_planets: list[dict],
        threshold_ratio: float = 0.1,
        target_logits: np.ndarray | None = None,
    ) -> list[list]:
        """将 target indices + sigmoid 比例转为 kaggle 动作格式。

        Args:
            target_indices: [N_owned] 目标星球索引
            num_ships_ratio: [N_owned] 或 [N_owned, 1] sigmoid 比例 [0, 1]
            owned_planets: 己方星球列表
            all_planets: 所有星球列表
            threshold_ratio: 比例阈值，低于此值不发射

        Returns:
            Kaggle格式的动作列表 [[from_planet_id, angle, num_ships], ...]
        """
        import math
        actions = []
        for i, src in enumerate(owned_planets):
            if i >= len(target_indices) or i >= len(num_ships_ratio):
                break

            # num_ships_ratio 是 sigmoid 比例，乘以源星球飞船数得到绝对数量
            ratio = float(num_ships_ratio[i]) if num_ships_ratio.ndim == 1 else float(num_ships_ratio[i, 0])
            if ratio < threshold_ratio or src["ships"] <= 0:
                continue

            ships = ratio * src["ships"]
            ships = max(1.0, min(ships, src["ships"]))

            tgt_idx = int(target_indices[i])
            if tgt_idx >= len(all_planets):
                continue
            tgt = all_planets[tgt_idx]
            angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
            primary_ships = int(ships)
            actions.append([src["id"], angle, primary_ships])

            # 允许同一星球二次发射（规则允许多次发射），用于缓解“单动作”限制。
            # 条件：剩余飞船充足，且模型对第二目标也有较强偏好。
            if target_logits is not None and i < target_logits.shape[0]:
                src_total = float(src["ships"])
                rem = int(src_total) - primary_ships
                if rem >= 8 and ratio >= 0.35:
                    top2 = np.argsort(target_logits[i])[-2:]
                    alt_idx = int(top2[0]) if int(top2[1]) == tgt_idx else int(top2[1])
                    if 0 <= alt_idx < len(all_planets) and alt_idx != tgt_idx:
                        alt = all_planets[alt_idx]
                        alt_angle = math.atan2(alt["y"] - src["y"], alt["x"] - src["x"])
                        split = max(1, int(rem * 0.4))
                        if split >= 3:
                            actions.append([src["id"], alt_angle, split])
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

    def __del__(self):
        """清理线程池资源。"""
        if hasattr(self, 'feature_pool') and self.feature_pool is not None:
            self.feature_pool.close()
            self.feature_pool.join()


def parallel_rollout(
    worker: RolloutWorker,
    num_games: int,
    num_players: int = 4,
    temperature: float = 1.0,
    env_config: dict | None = None,
    show_progress: bool = True,
    opponent_pool: Any | None = None,
) -> PPOBuffer:
    """并行 rollout（单进程版本，后续可扩展为多进程）。"""
    return worker.rollout_self_play(
        num_games=num_games,
        num_players=num_players,
        temperature=temperature,
        env_config=env_config,
        show_progress=show_progress,
        opponent_pool=opponent_pool,
    )
