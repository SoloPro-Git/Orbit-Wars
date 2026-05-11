"""Ray 分布式专家数据预训练。

使用 Ray 多 GPU 并行进行专家数据预训练。
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# 设置导入路径（支持从 training/expert 目录运行）
script_dir = Path(__file__).parent.parent
project_root = script_dir.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# 禁用日志
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KAGGLE_ENGINES_LOG_LEVEL'] = '0'
os.environ['SWANLAB_NO_INTERACTIVE'] = '1'
os.environ['SWANLAB_DISABLE_INTERACTIVE'] = '1'

import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

# SwanLab
try:
    import swanlab
except ImportError:
    swanlab = None

# Ray
try:
    import ray
    from ray.util.queue import Queue as RayQueue
except ImportError:
    print("Ray 未安装")
    ray = None

import math
from typing import Dict, List

from training.core.config import ExpertDataConfig, ModelConfig
from training.core.model import OrbitWarsModel
from training.core.feature_engineering import FeatureEngineer
from training.core.action import decode_actions
from training.expert.data_generator import ExpertDataset
from training.expert.action_labeling import infer_target_planet_id
from training.core.reward import RewardCalculator, RewardConfig

# ===========================================================================
# 内存模型 Agent（用于评估对局）
# ===========================================================================

class InMemoryModelAgent:
    """直接从内存中的 state_dict 创建的 Kaggle agent，用于评估对局。"""

    def __init__(
        self,
        model_state_dict: dict,
        model_config: ModelConfig,
        n_planets: int = 40,
        device: str = "cpu",
    ):
        self.device = torch.device(device)
        self.feature_engineer = FeatureEngineer()
        self.model = OrbitWarsModel(model_config, n_planets=n_planets)
        self.model.load_state_dict(model_state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.n_planets = n_planets
        self._patch_positional_encoding()

    def _patch_positional_encoding(self):
        """扩展位置编码，避免超过 max_len 报错。"""
        original_pe = self.model.pos_enc
        max_len = 2048
        d_model = original_pe.pe.shape[-1]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        original_pe.pe = pe

    def __call__(self, obs, configuration=None):
        """Kaggle agent 接口。"""
        try:
            return self._predict(obs)
        except Exception:
            return []

    def _predict(self, obs):
        """模型推理：特征提取 → 前向传播 → 动作解码。"""
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets

        # 特征提取
        planet_features, fleet_features, global_features, metadata = (
            self.feature_engineer.compute(obs, player_id)
        )

        # 转 tensor
        planet_feat_t = torch.from_numpy(planet_features).unsqueeze(0).to(self.device)
        global_feat_t = torch.from_numpy(global_features).unsqueeze(0).to(self.device)

        if fleet_features.shape[0] > 0:
            fleet_feat_t = torch.from_numpy(fleet_features).unsqueeze(0).to(self.device)
        else:
            fleet_feat_t = torch.zeros(
                1, 0, fleet_features.shape[1], device=self.device
            )

        # 构建 mask
        n_planets_actual = len(raw_planets)
        owned_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)
        enemy_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)

        for i, p in enumerate(raw_planets):
            owner = int(p[1])
            if owner == player_id:
                owned_mask[0, i] = True
            elif owner != -1:
                enemy_mask[0, i] = True

        # num_players
        owners = set()
        for p in raw_planets:
            o = int(p[1])
            if 0 <= o <= 3:
                owners.add(o)
        num_players_val = max(len(owners), 2)
        num_players = torch.tensor([num_players_val], dtype=torch.long, device=self.device)

        # planet_ships
        planet_ships = torch.tensor(
            [[float(p[5]) for p in raw_planets]],
            dtype=torch.float32,
            device=self.device,
        )

        # 模型推理
        with torch.no_grad():
            target_logits, num_ships_out, value, opp_target, opp_num_ships = self.model(
                planet_features=planet_feat_t,
                fleet_features=fleet_feat_t,
                global_features=global_feat_t,
                owned_mask=owned_mask,
                enemy_mask=enemy_mask,
                num_players=num_players,
                planet_ships=planet_ships,
            )

            # 截断或填充 target_logits 以匹配实际星球数量
            n_model_planets = target_logits.shape[2]
            if n_planets_actual < n_model_planets:
                target_logits = target_logits[:, :, :n_planets_actual]
            elif n_planets_actual > n_model_planets:
                pad_size = n_planets_actual - n_model_planets
                padding = torch.zeros(
                    1, target_logits.shape[1], pad_size, device=self.device
                )
                target_logits = torch.cat([target_logits, padding], dim=2)

        # 转 numpy
        target_logits_np = target_logits.cpu().numpy()
        num_ships_np = num_ships_out.cpu().numpy()

        if target_logits_np.shape[1] == 0:
            return []

        target_logits_np = target_logits_np[0]
        num_ships_np = num_ships_np[0]

        # 构建星球列表
        all_planets = []
        owned_planets = []
        for p in raw_planets:
            pid, owner, x, y, radius, ships, production = p
            planet_dict = {
                "id": int(pid),
                "owner": int(owner),
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "ships": float(ships),
                "production": float(production),
            }
            all_planets.append(planet_dict)
            if int(owner) == player_id:
                owned_planets.append(planet_dict)

        if len(owned_planets) == 0 or len(all_planets) == 0:
            return []

        if target_logits_np.shape[1] != len(all_planets):
            return []

        actions = self._decode_actions_with_split(
            target_logits=target_logits_np,
            num_ships_raw=num_ships_np,
            owned_planets=owned_planets,
            all_planets=all_planets,
            threshold=0.08,
        )

        return actions

    @staticmethod
    def _decode_actions_with_split(
        target_logits,
        num_ships_raw,
        owned_planets,
        all_planets,
        threshold: float = 0.08,
    ):
        actions = []
        for i, src in enumerate(owned_planets):
            if i >= len(target_logits) or i >= len(num_ships_raw):
                break
            src_ships = float(src.get("ships", 0.0))
            if src_ships <= 0:
                continue

            ratio = float(num_ships_raw[i, 0])
            if ratio < threshold:
                continue

            tgt_idx = int(np.argmax(target_logits[i]))
            if tgt_idx < 0 or tgt_idx >= len(all_planets):
                continue
            tgt = all_planets[tgt_idx]
            angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])

            send = int(max(1.0, min(src_ships, ratio * src_ships)))
            actions.append([src["id"], angle, send])

            # 允许次级分兵（和 rollout 对齐）
            rem = int(src_ships) - send
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


# ===========================================================================
# 专家策略胜率评估
# ===========================================================================

def evaluate_win_rate_against_expert(
    model_state_dict: dict,
    model_config: ModelConfig,
    n_planets: int = 40,
    num_games: int = 20,
    device: str = "cpu",
) -> Dict:
    """当前模型与专家策略对局，计算胜率。

    Args:
        model_state_dict: 当前模型参数
        model_config: 模型配置
        n_planets: 模型构建时的星球数
        num_games: 对局数
        device: 推理设备

    Returns:
        包含 win_rate, wins, losses, draws, num_games 的字典
    """
    from kaggle_environments import make
    from training.expert.kaggle_expert import KaggleExpertAgent

    # 创建模型 agent
    model_agent = InMemoryModelAgent(
        model_state_dict=model_state_dict,
        model_config=model_config,
        n_planets=n_planets,
        device=device,
    )

    # 创建专家 agent
    expert = KaggleExpertAgent()
    def expert_agent(obs, configuration=None):
        return expert.get_actions(obs)

    wins = 0
    losses = 0
    draws = 0

    for game_idx in range(num_games):
        seed = 42 + game_idx
        env = make("orbit_wars", configuration={"seed": seed}, debug=False)
        model_is_p0 = (game_idx % 2 == 0)
        if model_is_p0:
            env.run([model_agent, expert_agent])
        else:
            env.run([expert_agent, model_agent])

        final = env.steps[-1]
        p0_reward = final[0].reward
        p1_reward = final[1].reward
        model_reward = p0_reward if model_is_p0 else p1_reward
        expert_reward = p1_reward if model_is_p0 else p0_reward

        if model_reward > expert_reward:
            wins += 1
        elif model_reward < expert_reward:
            losses += 1
        else:
            draws += 1

        print(f"    第 {game_idx+1}/{num_games} 局 (seed={seed}): "
              f"模型={model_reward} vs 专家={expert_reward} "
              f"({'先手' if model_is_p0 else '后手'})"
              f" -> {'胜' if model_reward > expert_reward else '负' if model_reward < expert_reward else '平'}")

    win_rate = wins / num_games
    return {
        "win_rate": win_rate,
        "wins": wins,
        "losses": losses,
        "draws": draws,
        "num_games": num_games,
    }


# ===========================================================================
# Ray Worker
# ===========================================================================

@ray.remote
class PretrainingWorker:
    """预训练 Worker - 使用 Ray 分布式训练。"""

    def __init__(
        self,
        worker_id: int,
        config: ExpertDataConfig,
        model_config,
        data_files: list = None,
    ):
        """初始化 Worker。

        Args:
            worker_id: Worker ID
            config: 专家数据配置
            model_config: 模型配置
            data_files: 分配给该 worker 的数据文件列表
        """
        self.worker_id = worker_id
        self.config = config

        # 在 worker 内部检测 CUDA 设备
        if torch.cuda.is_available():
            self.device = "cuda:0"
            gpu_id = torch.cuda.current_device()
            print(f"[PretrainingWorker {worker_id}] 初始化 on cuda:0 (GPU {gpu_id})")
        else:
            self.device = "cpu"
            print(f"[PretrainingWorker {worker_id}] 初始化 on CPU")

        # 加载模型
        self.model_n_planets = 40
        self.model = OrbitWarsModel(model_config, n_planets=self.model_n_planets).to(self.device)
        self.model.train()

        # 只加载分配的文件
        if data_files:
            print(f"[PretrainingWorker {worker_id}] 加载 {len(data_files)} 个文件...")
            from training.expert.data_generator import ExpertDataGenerator
            generator = ExpertDataGenerator()
            self.train_data = []
            for fp in data_files:
                self.train_data.extend(generator.load_trajectories(Path(fp)))
            print(f"[PretrainingWorker {worker_id}] 加载完成: {len(self.train_data)} 样本")
        else:
            # 兼容：加载全部数据
            dataset = ExpertDataset(data_dir=config.data_dir)
            self.train_data = dataset.data

        lr = float(config.pretrain_learning_rate)  # 确保是 float 类型
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
        )

        # 特征工程
        self.feature_engineer = FeatureEngineer()

        # 奖励计算器（用于重新计算 reward）
        self.reward_calculator = RewardCalculator(
            config=RewardConfig(
                terminal_weight=1.0,
                intermediate_weight=0.2,
                capture_reward_weight=0.3,
                loss_penalty_weight=0.2,
                defense_reward_weight=0.15,
                transit_cost_weight=0.1,
                production_advantage_weight=0.15,
                comet_roi_weight=0.1,
            ),
            max_turns=500,
        )

    def _infer_num_players(self, raw_planets: list) -> int:
        """从原始行星数据推断玩家数量。"""
        players = set()
        for p in raw_planets:
            owner = int(p[1])
            if owner >= 0:  # 忽略中立行星（owner = -1）
                players.add(owner)
        return max(len(players), 2)  # 至少 2 个玩家

    def train_multi_iterations(
        self,
        model_state_dict: dict,
        num_iters: int,
        start_iteration: int = 0,
    ) -> Dict:
        """连续执行多次训练迭代，只在最后返回模型参数。

        Args:
            model_state_dict: 当前模型参数
            num_iters: 连续训练的迭代次数
            start_iteration: 起始迭代编号（用于进度条）

        Returns:
            最终的训练统计和模型参数
        """
        self.model.load_state_dict(model_state_dict)

        total_loss = 0.0
        total_valid = 0
        total_acted_ratio_expert = 0.0
        total_acted_ratio_pred = 0.0
        total_ship_ratio_mae = 0.0
        total_target_label_valid_ratio = 0.0
        total_action_map_ratio = 0.0
        total_bc_loss = 0.0
        total_value_loss = 0.0

        for local_iter in range(num_iters):
            it = start_iteration + local_iter
            iter_loss = 0.0
            iter_valid = 0

            pbar = tqdm(
                range(500),
                desc=f"[W{self.worker_id} iter{it}]",
                unit="step",
            )

            for step in pbar:
                sample = random.choice(self.train_data)
                try:
                    planet_features, fleet_features, global_features, metadata = self.feature_engineer.compute(
                        sample["observation"],
                        sample["player_id"],
                    )

                    planet_features = torch.from_numpy(planet_features).float().unsqueeze(0).to(self.device)
                    fleet_features = torch.from_numpy(fleet_features).float().unsqueeze(0).to(self.device) if len(fleet_features) > 0 else None
                    global_features = torch.from_numpy(global_features).float().unsqueeze(0).to(self.device)

                    n_planets = planet_features.size(1)
                    owned_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                    if metadata.get("owned_planet_indices"):
                        owned_mask[metadata["owned_planet_indices"]] = True

                    enemy_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                    if metadata.get("enemy_planet_indices"):
                        enemy_mask[metadata["enemy_planet_indices"]] = True

                    raw_planets = sample["observation"].get("planets", [])
                    num_players = self._infer_num_players(raw_planets)
                    num_players_tensor = torch.tensor([num_players], dtype=torch.long).to(self.device)

                    expert_actions = sample["actions"]
                    if not expert_actions:
                        continue

                    features_dict = {
                        "planet_features": planet_features,
                        "fleet_features": fleet_features,
                        "global_features": global_features,
                        "owned_mask": owned_mask.unsqueeze(0),
                        "enemy_mask": enemy_mask.unsqueeze(0),
                        "num_players": num_players_tensor,
                    }

                    loss, diag = self._compute_loss(
                        features_dict,
                        expert_actions,
                        sample["observation"],
                        sample.get("player_id"),
                        sample.get("reward", 0.0),
                    )

                    self.optimizer.zero_grad()
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()

                    iter_loss += loss.item()
                    iter_valid += 1
                    total_acted_ratio_expert += diag.get("acted_ratio_expert", 0.0)
                    total_acted_ratio_pred += diag.get("acted_ratio_pred", 0.0)
                    total_ship_ratio_mae += diag.get("ship_ratio_mae", 0.0)
                    total_target_label_valid_ratio += diag.get("target_label_valid_ratio", 0.0)
                    total_action_map_ratio += diag.get("action_map_ratio", 0.0)
                    total_bc_loss += diag.get("bc_loss", 0.0)
                    total_value_loss += diag.get("value_loss", 0.0)

                    if step % 50 == 0 and iter_valid > 0:
                        pbar.set_postfix(loss=f"{iter_loss / iter_valid:.4f}", valid=iter_valid)

                except Exception as e:
                    msg = str(e)
                    if (
                        "CUDA error" in msg
                        or "device-side assert" in msg
                        or "index out of range" in msg
                        or "Target" in msg
                    ):
                        raise
                    continue

            pbar.close()
            total_loss += iter_loss
            total_valid += iter_valid

        avg_loss = total_loss / max(1, total_valid)

        return {
            "worker_id": self.worker_id,
            "model_state": self.model.state_dict(),
            "loss": avg_loss,
            "num_samples": total_valid,
            "acted_ratio_expert": total_acted_ratio_expert / max(1, total_valid),
            "acted_ratio_pred": total_acted_ratio_pred / max(1, total_valid),
            "ship_ratio_mae": total_ship_ratio_mae / max(1, total_valid),
            "target_label_valid_ratio": total_target_label_valid_ratio / max(1, total_valid),
            "action_map_ratio": total_action_map_ratio / max(1, total_valid),
            "bc_loss": total_bc_loss / max(1, total_valid),
            "value_loss": total_value_loss / max(1, total_valid),
        }

    def train_iteration(self, model_state_dict: dict, iteration: int = 0) -> Dict:
        """执行一次训练迭代。

        每个 iter 采样固定步数，不遍历全部数据。

        Args:
            model_state_dict: 当前模型参数
            iteration: 当前迭代编号（用于进度条显示）

        Returns:
            训练统计信息
        """
        self.model.load_state_dict(model_state_dict)

        steps_per_iter = 500
        total_loss = 0.0
        num_valid = 0
        total_acted_ratio_expert = 0.0
        total_acted_ratio_pred = 0.0
        total_ship_ratio_mae = 0.0
        total_target_label_valid_ratio = 0.0
        total_action_map_ratio = 0.0

        pbar = tqdm(
            range(steps_per_iter),
            desc=f"[W{self.worker_id} iter{iteration}]",
            unit="step",
            disable=False,
        )

        for step in pbar:
            sample = random.choice(self.train_data)

            try:
                planet_features, fleet_features, global_features, metadata = self.feature_engineer.compute(
                    sample["observation"],
                    sample["player_id"],
                )

                planet_features = torch.from_numpy(planet_features).float().unsqueeze(0).to(self.device)
                fleet_features = torch.from_numpy(fleet_features).float().unsqueeze(0).to(self.device) if len(fleet_features) > 0 else None
                global_features = torch.from_numpy(global_features).float().unsqueeze(0).to(self.device)

                n_planets = planet_features.size(1)
                owned_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                if metadata.get("owned_planet_indices"):
                    owned_mask[metadata["owned_planet_indices"]] = True

                enemy_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                if metadata.get("enemy_planet_indices"):
                    enemy_mask[metadata["enemy_planet_indices"]] = True

                raw_planets = sample["observation"].get("planets", [])
                num_players = self._infer_num_players(raw_planets)
                num_players_tensor = torch.tensor([num_players], dtype=torch.long).to(self.device)

                expert_actions = sample["actions"]
                if not expert_actions:
                    continue

                features_dict = {
                    "planet_features": planet_features,
                    "fleet_features": fleet_features,
                    "global_features": global_features,
                    "owned_mask": owned_mask.unsqueeze(0),
                    "enemy_mask": enemy_mask.unsqueeze(0),
                    "num_players": num_players_tensor,
                }

                loss = self._compute_loss(
                    features_dict,
                    expert_actions,
                    sample["observation"],
                    sample.get("player_id"),
                    sample.get("reward", 0.0),
                )

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()

                total_loss += loss.item()
                num_valid += 1

                # 每 50 步更新一次进度条
                if step % 50 == 0 and num_valid > 0:
                    avg = total_loss / num_valid
                    pbar.set_postfix(loss=f"{avg:.4f}", valid=num_valid)

            except Exception as e:
                msg = str(e)
                if (
                    "CUDA error" in msg
                    or "device-side assert" in msg
                    or "index out of range" in msg
                    or "Target" in msg
                ):
                    raise
                continue

        pbar.close()
        avg_loss = total_loss / max(1, num_valid)

        return {
            "worker_id": self.worker_id,
            "model_state": self.model.state_dict(),
            "loss": avg_loss,
            "num_samples": num_valid,
        }

    def validate(self, model_state_dict: dict) -> Dict:
        """验证模型。

        Args:
            model_state_dict: 模型参数

        Returns:
            验证统计信息
        """
        self.model.load_state_dict(model_state_dict)
        self.model.eval()

        val_samples = random.sample(self.train_data, min(100, len(self.train_data)))
        total_loss = 0.0
        num_valid = 0
        total_acted_ratio_expert = 0.0
        total_acted_ratio_pred = 0.0
        total_ship_ratio_mae = 0.0
        total_target_label_valid_ratio = 0.0
        total_action_map_ratio = 0.0

        with torch.no_grad():
            for sample in val_samples:
                try:
                    # 提取特征
                    planet_features, fleet_features, global_features, metadata = self.feature_engineer.compute(
                        sample["observation"],
                        sample["player_id"],
                    )

                    # 转换为张量
                    planet_features = torch.from_numpy(planet_features).float().unsqueeze(0).to(self.device)
                    fleet_features = torch.from_numpy(fleet_features).float().unsqueeze(0).to(self.device) if len(fleet_features) > 0 else None
                    global_features = torch.from_numpy(global_features).float().unsqueeze(0).to(self.device)

                    # 构建 masks
                    n_planets = planet_features.size(1)
                    owned_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                    owned_indices = metadata.get("owned_planet_indices", [])
                    if owned_indices:
                        owned_mask[owned_indices] = True

                    enemy_mask = torch.zeros(n_planets, dtype=torch.bool).to(self.device)
                    enemy_indices = metadata.get("enemy_planet_indices", [])
                    if enemy_indices:
                        enemy_mask[enemy_indices] = True

                    owned_mask = owned_mask.unsqueeze(0)
                    enemy_mask = enemy_mask.unsqueeze(0)

                    # 推断玩家数量
                    raw_planets = sample["observation"].get("planets", [])
                    num_players_val = self._infer_num_players(raw_planets)
                    num_players = torch.tensor([num_players_val], dtype=torch.long).to(self.device)

                    # 构建特征字典
                    features_dict = {
                        "planet_features": planet_features,
                        "fleet_features": fleet_features,
                        "global_features": global_features,
                        "owned_mask": owned_mask,
                        "enemy_mask": enemy_mask,
                        "num_players": num_players,
                    }

                    expert_actions = sample["actions"]
                    if not expert_actions:
                        continue

                    loss, diag = self._compute_loss(
                        features_dict,
                        expert_actions,
                        sample["observation"],
                        sample.get("player_id"),
                        sample.get("reward", 0.0),
                    )
                    total_loss += loss.item()
                    num_valid += 1
                    total_acted_ratio_expert += diag.get("acted_ratio_expert", 0.0)
                    total_acted_ratio_pred += diag.get("acted_ratio_pred", 0.0)
                    total_ship_ratio_mae += diag.get("ship_ratio_mae", 0.0)
                    total_target_label_valid_ratio += diag.get("target_label_valid_ratio", 0.0)
                    total_action_map_ratio += diag.get("action_map_ratio", 0.0)

                except Exception:
                    continue

        self.model.train()
        return {
            "worker_id": self.worker_id,
            "val_loss": total_loss / max(1, num_valid),
            "val_acted_ratio_expert": total_acted_ratio_expert / max(1, num_valid),
            "val_acted_ratio_pred": total_acted_ratio_pred / max(1, num_valid),
            "val_ship_ratio_mae": total_ship_ratio_mae / max(1, num_valid),
            "val_target_label_valid_ratio": total_target_label_valid_ratio / max(1, num_valid),
            "val_action_map_ratio": total_action_map_ratio / max(1, num_valid),
        }

    def _compute_loss(
        self,
        features_dict: dict,
        expert_actions: list,
        observation: dict,
        player_id: int | None = None,
        reward: float = 0.0,
    ) -> tuple[torch.Tensor, dict]:
        """计算行为克隆损失 + value function 损失。

        Args:
            features_dict: 特征字典，包含：
                - planet_features: [batch, N_planets, D_PLANET]
                - fleet_features: [batch, N_fleets, D_FLEET] or None
                - global_features: [batch, D_GLOBAL]
                - owned_mask: [batch, N_planets]
                - enemy_mask: [batch, N_planets]
                - num_players: [batch]
            expert_actions: 专家动作列表 [[from_planet_id, angle, num_ships], ...]
            observation: 原始观测数据

        Returns:
            总损失
        """
        # 运行模型前向传播
        model_output = self.model(
            planet_features=features_dict["planet_features"],
            fleet_features=features_dict["fleet_features"],
            global_features=features_dict["global_features"],
            owned_mask=features_dict["owned_mask"],
            enemy_mask=features_dict["enemy_mask"],
            num_players=features_dict["num_players"],
        )

        # 提取模型输出
        target_logits = model_output[0]  # [batch, N_owned, N_planets]
        pred_ships = model_output[1]     # [batch, N_owned, 1]
        value_pred = model_output[2]     # [batch, max_players]

        # 1. 行为克隆损失
        bc_loss, bc_diag = self._compute_behavior_cloning_loss(
            target_logits, pred_ships, expert_actions, observation, player_id
        )

        # 2. Value loss 在预训练阶段容易引入噪声，降到很低权重。
        value_loss = self._compute_value_loss(value_pred, reward)

        # 总损失：以行为克隆为主
        total_loss = self.config.behavior_clone_loss_coef * bc_loss + 0.02 * value_loss
        bc_diag["value_loss"] = float(value_loss.detach().item())
        bc_diag["bc_loss"] = float(bc_loss.detach().item())
        return total_loss, bc_diag

    def _compute_behavior_cloning_loss(
        self,
        target_logits: torch.Tensor,
        pred_ships: torch.Tensor,
        expert_actions: list,
        observation: dict,
        player_id: int | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """计算行为克隆损失。

        将专家的 angle 映射到目标行星 ID，然后计算分类损失。
        """
        if not expert_actions:
            return torch.tensor(0.0, device=self.device), {
                "acted_ratio_expert": 0.0,
                "acted_ratio_pred": 0.0,
                "ship_ratio_mae": 0.0,
                "target_label_valid_ratio": 0.0,
                "action_map_ratio": 0.0,
            }

        raw_planets = observation.get("planets", [])
        if not raw_planets:
            return torch.tensor(0.0, device=self.device), {
                "acted_ratio_expert": 0.0,
                "acted_ratio_pred": 0.0,
                "ship_ratio_mae": 0.0,
                "target_label_valid_ratio": 0.0,
                "action_map_ratio": 0.0,
            }

        all_planets = [
            {
                "id": int(p[0]),
                "owner": int(p[1]),
                "x": float(p[2]),
                "y": float(p[3]),
                "ships": float(p[5]),
            }
            for p in raw_planets
        ]
        id_to_col = {p["id"]: i for i, p in enumerate(all_planets)}

        if player_id is None:
            player_id = observation.get("player", 0)

        # 找到该玩家拥有的行星
        owned_planets = [p for p in all_planets if p["owner"] == player_id]
        owned_planet_ids = [p["id"] for p in owned_planets]

        if not owned_planet_ids:
            return torch.tensor(0.0, device=self.device), {
                "acted_ratio_expert": 0.0,
                "acted_ratio_pred": 0.0,
                "ship_ratio_mae": 0.0,
                "target_label_valid_ratio": 0.0,
                "action_map_ratio": 0.0,
            }

        # 将专家动作转换为模型格式
        # 专家动作：[[from_planet_id, angle, num_ships], ...]
        # 模型输出：对于每个 owned planet，预测目标 planet ID 和 ships 比例

        # 创建目标张量
        # 假设 batch_size = 1（单样本）
        batch_size = 1
        n_owned = len(owned_planet_ids)
        n_planets = len(all_planets)

        if n_owned == 0 or target_logits.size(1) < n_owned:
            return torch.tensor(0.0, device=self.device)

        # 创建源行星 ID 到索引的映射
        owned_id_to_idx = {pid: idx for idx, pid in enumerate(owned_planet_ids)}

        # 为每个 owned planet 创建目标标签
        target_labels = []      # [n_owned] 目标行星 ID
        ship_ratios = []        # [n_owned] 派遣船只比例
        acted_flags = []        # [n_owned] 是否行动 (0/1)

        # 首先，找到每个 owned planet 的总船只数
        planet_ships = {p["id"]: p["ships"] for p in owned_planets}

        # 默认：不派遣任何船只
        for pid in owned_planet_ids:
            target_labels.append(-1)  # -1 表示不行动
            ship_ratios.append(0.0)
            acted_flags.append(0.0)

        # 根据专家动作更新
        # 同一 source 行星若有多次发射，保留舰队规模最大的动作，
        # 避免“后写覆盖”随机吞掉主动作标签。
        best_action_by_src = {}
        for action in expert_actions:
            if len(action) < 3:
                continue
            from_pid = int(action[0])
            ships = int(action[2])
            prev = best_action_by_src.get(from_pid)
            if prev is None or ships > int(prev[2]):
                best_action_by_src[from_pid] = action

        mapped_count = 0
        for action in best_action_by_src.values():
            if len(action) < 3:
                continue
            from_pid = int(action[0])
            angle = float(action[1])
            ships = int(action[2])

            if from_pid not in owned_id_to_idx:
                continue

            if from_pid not in id_to_col:
                continue

            best_tid = infer_target_planet_id(
                observation=observation,
                from_planet_id=from_pid,
                angle=angle,
                num_ships=float(ships),
            )

            # 更新标签
            if best_tid is not None and best_tid in id_to_col:
                idx = owned_id_to_idx[from_pid]
                target_labels[idx] = id_to_col[best_tid]
                mapped_count += 1
                # 计算派遣比例
                if from_pid in planet_ships and planet_ships[from_pid] > 0:
                    ratio = min(ships / planet_ships[from_pid], 1.0)
                    ship_ratios[idx] = ratio
                    acted_flags[idx] = 1.0
            else:
                # 即便目标无法映射，也保留“该星球有行动”和出兵比例监督
                idx = owned_id_to_idx[from_pid]
                if from_pid in planet_ships and planet_ships[from_pid] > 0:
                    ratio = min(ships / planet_ships[from_pid], 1.0)
                    ship_ratios[idx] = max(ship_ratios[idx], ratio)
                acted_flags[idx] = 1.0

        # 转换为张量
        target_labels = torch.tensor(target_labels, dtype=torch.long, device=self.device)
        ship_targets = torch.tensor(ship_ratios, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)  # [1, n_owned, 1]
        acted_targets = torch.tensor(acted_flags, dtype=torch.float32, device=self.device).unsqueeze(0).unsqueeze(-1)  # [1, n_owned, 1]

        # 只对有效动作计算损失，且目标索引必须在当前 logits 的真实范围内。
        # 不要只依赖 self.model_n_planets，避免 CE 目标越界导致 CUDA assert。
        logits_n_planets = int(target_logits.size(-1))
        valid_mask = (target_labels >= 0) & (target_labels < logits_n_planets)
        if not valid_mask.any():
            pred_owned = pred_ships[0, :n_owned, :]
            tgt_owned = ship_targets[0, :n_owned, :]
            act_owned = acted_targets[0, :n_owned, :]
            acted_count = float(act_owned.sum().item())
            diag = {
                "acted_ratio_expert": float(act_owned.mean().item()),
                "acted_ratio_pred": float(pred_owned.mean().item()),
                "ship_ratio_mae": float((pred_owned - tgt_owned).abs().mean().item()),
                "target_label_valid_ratio": 0.0,
                "action_map_ratio": float(mapped_count / max(acted_count, 1.0)),
            }
            return torch.tensor(0.0, device=self.device), diag

        # 1. 目标选择损失（交叉熵）
        # target_logits: [1, n_owned, logits_n_planets]
        # 取出有效的 owned planets
        valid_indices = torch.where(valid_mask)[0]
        valid_logits = target_logits[0, valid_indices, :logits_n_planets]  # [n_valid, logits_n_planets]
        valid_targets = target_labels[valid_indices]        # [n_valid]
        if valid_targets.numel() == 0 or valid_logits.size(-1) <= 1:
            pred_owned = pred_ships[0, :n_owned, :]
            tgt_owned = ship_targets[0, :n_owned, :]
            act_owned = acted_targets[0, :n_owned, :]
            acted_count = float(act_owned.sum().item())
            diag = {
                "acted_ratio_expert": float(act_owned.mean().item()),
                "acted_ratio_pred": float(pred_owned.mean().item()),
                "ship_ratio_mae": float((pred_owned - tgt_owned).abs().mean().item()),
                "target_label_valid_ratio": 0.0,
                "action_map_ratio": float(mapped_count / max(acted_count, 1.0)),
            }
            return torch.tensor(0.0, device=self.device), diag

        target_loss = F.cross_entropy(valid_logits, valid_targets)

        # 2. 出兵比例损失（对所有 owned 星球监督：行动=ratio，不行动=0）
        pred_owned = pred_ships[0, :n_owned, :]  # [n_owned, 1]
        tgt_owned = ship_targets[0, :n_owned, :]  # [n_owned, 1]
        act_owned = acted_targets[0, :n_owned, :]  # [n_owned, 1]

        ratio_weights = 1.0 + 3.0 * act_owned
        ratio_err = (pred_owned - tgt_owned).pow(2)
        ships_loss = (ratio_err * ratio_weights).sum() / ratio_weights.sum().clamp(min=1.0)

        # 3. 行动/不行动监督（帮助学会节奏）
        act_bce = F.binary_cross_entropy(
            pred_owned.clamp(min=1e-5, max=1 - 1e-5),
            act_owned,
        )

        # 总行为克隆损失
        bc_loss = target_loss + 0.3 * ships_loss + 1.0 * act_bce

        acted_count = float(act_owned.sum().item())
        valid_count = float(valid_mask.sum().item())
        diag = {
            "acted_ratio_expert": float(act_owned.mean().item()),
            "acted_ratio_pred": float(pred_owned.mean().item()),
            "ship_ratio_mae": float((pred_owned - tgt_owned).abs().mean().item()),
            "target_label_valid_ratio": float(valid_count / max(acted_count, 1.0)),
            "action_map_ratio": float(mapped_count / max(acted_count, 1.0)),
        }
        return bc_loss, diag

    def _compute_value_loss(
        self,
        value_pred: torch.Tensor,
        reward: float,
    ) -> torch.Tensor:
        """计算标量 value loss，与当前模型的标量 value head 对齐。"""
        value_target = torch.tensor([float(reward)], dtype=torch.float32, device=self.device)
        return F.mse_loss(value_pred.view(-1), value_target)

# ===========================================================================
# 分布式预训练器
# ===========================================================================

class RayDistributedPretrainer:
    """Ray 分布式专家数据预训练器。

    使用多个 GPU 并行进行专家数据预训练。
    """

    def __init__(
        self,
        model: OrbitWarsModel,
        config: ExpertDataConfig,
        swanlab_run=None,
    ):
        """初始化分布式预训练器。

        Args:
            model: 模型
            config: 配置
            swanlab_run: SwanLab 运行实例
        """
        self.config = config
        self.swanlab = swanlab
        self.model_config = model.config
        self.n_planets = 40  # 与 PretrainingWorker 一致

        # 初始化 Ray
        if not ray.is_initialized():
            # 配置 Ray 临时目录到 /data2 分区（避免根分区空间不足）
            import tempfile
            ray_temp_dir = Path("/data2/solo/Orbit-Wars/training/.ray_temp")
            ray_temp_dir.mkdir(parents=True, exist_ok=True)
            print(f"[Ray] 临时目录: {ray_temp_dir}")

            # 获取可见 GPU 数量
            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print(f"[Ray] 检测到 {num_gpus} 个 GPU")

            ray.init(
                num_gpus=num_gpus,
                ignore_reinit_error=True,
                logging_level="WARNING",
                _temp_dir=str(ray_temp_dir),
                include_dashboard=False,  # 禁用 Dashboard（Python 3.12 兼容性问题）
            )
            print(f"[Ray] 初始化完成: {ray.cluster_resources()}")

        print(f"\n{'='*60}")
        print(f"Ray 分布式专家数据预训练")
        print(f"{'='*60}")
        print(f"Worker 数量: {config.num_pretrain_workers}")
        print(f"每个 Worker GPU: {config.gpus_per_worker}")
        print(f"总 GPU 数量: {config.num_pretrain_workers * config.gpus_per_worker:.1f}")
        print(f"最大迭代次数: {config.num_pretrain_iterations}")
        print(f"自动继续: {config.auto_proceed}")
        print(f"{'='*60}\n")

        # 优化：按文件分片，每个 worker 只加载自己的文件，并行加载
        data_dir = Path(config.data_dir)
        data_files = sorted(list(data_dir.glob("*.jsonl")) + list(data_dir.glob("*.pkl")))
        total_workers = config.num_pretrain_workers
        print(f"数据文件: {len(data_files)} 个，分配给 {total_workers} 个 workers")

        # 按文件均匀分配
        worker_file_slices = [[] for _ in range(total_workers)]
        for i, f in enumerate(data_files):
            worker_file_slices[i % total_workers].append(str(f))
        for wid, files in enumerate(worker_file_slices):
            print(f"  Worker {wid}: {len(files)} 个文件")

        # 创建 Workers
        self.workers = []
        for worker_id in range(config.num_pretrain_workers):
            worker = PretrainingWorker.options(
                num_gpus=config.gpus_per_worker,
            ).remote(
                worker_id,
                config,
                model.config,
                worker_file_slices[worker_id],  # 传递文件列表
            )
            self.workers.append(worker)

        # 全局模型状态
        self.global_model_state = model.state_dict()
        self.checkpoint_dir = Path("training/checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def pretrain(self) -> Dict:
        """执行分布式预训练。

        支持从 checkpoint 恢复：如果 config.pretrain_resume_from 指定了有效的
        checkpoint 路径，则从该 checkpoint 的迭代次数继续训练。

        Returns:
            训练统计信息
        """
        max_iterations = self.config.num_pretrain_iterations

        # 检查是否需要从 checkpoint 恢复
        start_iteration = 0
        resume_path = self.config.pretrain_resume_from
        if resume_path:
            resume_ckpt = Path(resume_path)
            if resume_ckpt.exists():
                print(f"从 checkpoint 恢复: {resume_ckpt}")
                ckpt = torch.load(resume_ckpt, map_location="cpu", weights_only=False)
                self.global_model_state = ckpt["model_state_dict"]
                start_iteration = ckpt.get("iteration", 0)
                print(f"  已恢复到第 {start_iteration} 次迭代")
            else:
                print(f"⚠️  checkpoint 不存在: {resume_ckpt}，从头开始训练")

        if start_iteration >= max_iterations:
            print(f"✓ 已完成 {start_iteration} 次迭代（目标 {max_iterations}），跳过预训练")
            return {"train_loss": [], "val_loss": []}

        remaining = max_iterations - start_iteration
        print(f"开始预训练（从第 {start_iteration} 到 {max_iterations}，共 {remaining} 次迭代）...\n")

        stats = {
            "train_loss": [],
            "val_loss": [],
        }

        sync_interval = self.config.sync_interval
        pbar_sync = tqdm(
            range(start_iteration, max_iterations, sync_interval),
            desc="[预训练]",
            unit="sync",
        )

        for sync_start in pbar_sync:
            t0 = time.time()
            sync_end = min(sync_start + sync_interval, max_iterations)
            num_local_iters = sync_end - sync_start

            # 1. 每个 worker 连续训练 num_local_iters 个 iter
            futures = [
                worker.train_multi_iterations.remote(
                    self.global_model_state,
                    num_iters=num_local_iters,
                    start_iteration=sync_start,
                )
                for worker in self.workers
            ]

            results = ray.get(futures)

            # 2. 合并模型参数
            self._aggregate_models(results)

            # 3. 统计
            avg_train_loss = np.mean([r["loss"] for r in results])
            stats["train_loss"].append(avg_train_loss)
            avg_acted_ratio_expert = float(np.mean([r.get("acted_ratio_expert", 0.0) for r in results]))
            avg_acted_ratio_pred = float(np.mean([r.get("acted_ratio_pred", 0.0) for r in results]))
            avg_ship_ratio_mae = float(np.mean([r.get("ship_ratio_mae", 0.0) for r in results]))
            avg_target_label_valid_ratio = float(np.mean([r.get("target_label_valid_ratio", 0.0) for r in results]))
            avg_action_map_ratio = float(np.mean([r.get("action_map_ratio", 0.0) for r in results]))
            avg_bc_loss = float(np.mean([r.get("bc_loss", 0.0) for r in results]))
            avg_value_loss = float(np.mean([r.get("value_loss", 0.0) for r in results]))
            sync_time = time.time() - t0

            # 4. 验证
            val_loss_str = ""
            val_futures = [
                worker.validate.remote(self.global_model_state)
                for worker in self.workers
            ]
            val_results = ray.get(val_futures)
            avg_val_loss = np.mean([r["val_loss"] for r in val_results])
            stats["val_loss"].append(avg_val_loss)
            val_loss_str = f" Val:{avg_val_loss:.4f}"
            avg_val_acted_ratio_expert = float(np.mean([r.get("val_acted_ratio_expert", 0.0) for r in val_results]))
            avg_val_acted_ratio_pred = float(np.mean([r.get("val_acted_ratio_pred", 0.0) for r in val_results]))
            avg_val_ship_ratio_mae = float(np.mean([r.get("val_ship_ratio_mae", 0.0) for r in val_results]))
            avg_val_target_label_valid_ratio = float(np.mean([r.get("val_target_label_valid_ratio", 0.0) for r in val_results]))
            avg_val_action_map_ratio = float(np.mean([r.get("val_action_map_ratio", 0.0) for r in val_results]))

            # 记录 SwanLab
            if self.swanlab:
                self.swanlab.log({
                    "pretrain_iteration": sync_end - 1,
                    "train_loss": avg_train_loss,
                    "val_loss": avg_val_loss,
                    "pretrain_bc_loss": avg_bc_loss,
                    "pretrain_value_loss": avg_value_loss,
                    "pretrain_acted_ratio_expert": avg_acted_ratio_expert,
                    "pretrain_acted_ratio_pred": avg_acted_ratio_pred,
                    "pretrain_ship_ratio_mae": avg_ship_ratio_mae,
                    "pretrain_target_label_valid_ratio": avg_target_label_valid_ratio,
                    "pretrain_action_map_ratio": avg_action_map_ratio,
                    "pretrain_val_acted_ratio_expert": avg_val_acted_ratio_expert,
                    "pretrain_val_acted_ratio_pred": avg_val_acted_ratio_pred,
                    "pretrain_val_ship_ratio_mae": avg_val_ship_ratio_mae,
                    "pretrain_val_target_label_valid_ratio": avg_val_target_label_valid_ratio,
                    "pretrain_val_action_map_ratio": avg_val_action_map_ratio,
                })

            # 更新进度条
            pbar_sync.set_postfix(
                Iter=f"{sync_start}-{sync_end-1}",
                Loss=f"{avg_train_loss:.4f}{val_loss_str}",
                Time=f"{sync_time:.1f}s",
            )

            # 详细汇总
            worker_losses = [f"W{r['worker_id']}:{r['loss']:.4f}" for r in results]
            total_samples = sum(r["num_samples"] for r in results)
            pbar_sync.write(
                f"[Iter {sync_start}-{sync_end-1}] "
                f"Loss:{avg_train_loss:.4f}{val_loss_str} "
                f"Act(pred/expert):{avg_acted_ratio_pred:.3f}/{avg_acted_ratio_expert:.3f} "
                f"ShipMAE:{avg_ship_ratio_mae:.3f} "
                f"ValidLbl:{avg_target_label_valid_ratio:.3f} "
                f"Samples:{total_samples} "
                f"Time:{sync_time:.1f}s "
                f"| {' '.join(worker_losses)}"
            )

            # 5. 保存 checkpoint
            if sync_end % 50 == 0 or sync_end >= max_iterations:
                ckpt_path = self.checkpoint_dir / f"pretrain_iter_{sync_end}.pkl"
                self._save_checkpoint(ckpt_path, sync_end)
                pbar_sync.write(f"  ✓ Checkpoint: {ckpt_path}")

            # 6. 专家策略评估门控
            gate_passed = False
            if (
                self.config.eval_gate_enabled
                and sync_end % self.config.eval_gate_interval == 0
                and sync_end > 0
            ):
                pbar_sync.write(f"  [Gate] 运行专家评估 (iter {sync_end})...")
                try:
                    eval_result = evaluate_win_rate_against_expert(
                        model_state_dict=self.global_model_state,
                        model_config=self.model_config,
                        n_planets=self.n_planets,
                        num_games=self.config.eval_gate_num_games,
                        device=self.config.eval_gate_device,
                    )

                    win_rate = eval_result["win_rate"]
                    stats.setdefault("win_rates", []).append(win_rate)
                    stats["final_win_rate"] = win_rate

                    # 记录到 SwanLab
                    if self.swanlab:
                        self.swanlab.log({
                            "pretrain_expert_win_rate": win_rate,
                            "pretrain_iteration": sync_end,
                        })

                    pbar_sync.write(
                        f"  [Gate] 胜率: {win_rate:.1%} "
                        f"({eval_result['wins']}胜/{eval_result['losses']}负/{eval_result['draws']}平) "
                        f"- 阈值: {self.config.eval_gate_win_rate_threshold:.0%}"
                    )

                    if win_rate >= self.config.eval_gate_win_rate_threshold:
                        gate_passed = True
                        pbar_sync.write(
                            f"  [Gate] ✓ 通过! 胜率 {win_rate:.1%} >= "
                            f"{self.config.eval_gate_win_rate_threshold:.0%}，"
                            f"在 iter {sync_end} 提前结束预训练"
                        )
                except Exception as e:
                    pbar_sync.write(f"  [Gate] 评估失败: {e}，继续训练")

            if gate_passed:
                stats["gate_passed"] = True
                stats["final_iteration"] = sync_end
                break

        pbar_sync.close()

        # 保存最终模型
        final_iter = stats.get("final_iteration", max_iterations)
        final_ckpt = self.checkpoint_dir / "pretrained_model.pkl"
        self._save_checkpoint(final_ckpt, final_iter)

        if stats.get("gate_passed"):
            print(f"\n✓ 预训练通过门控！胜率 {stats.get('final_win_rate', 0):.1%}，"
                  f"在 iter {final_iter} 结束")
        else:
            print(f"\n✓ 预训练完成（达到最大迭代 {max_iterations}）")
        print(f"  模型已保存: {final_ckpt}")
        print(f"{'='*60}\n")

        stats.setdefault("gate_passed", False)
        stats.setdefault("final_win_rate", 0.0)
        return stats

    def _aggregate_models(self, results: List[Dict]):
        """聚合多个 Worker 的模型参数。

        Args:
            results: Worker 训练结果
        """
        # 获取所有模型状态
        model_states = [r["model_state"] for r in results]

        # 平均参数
        aggregated_state = {}
        for key in model_states[0].keys():
            # 收集所有 worker 的参数
            params = [state[key].cpu() for state in model_states]

            # 计算平均
            avg_param = torch.stack(params).mean(dim=0)
            aggregated_state[key] = avg_param

        # 更新全局状态
        self.global_model_state = aggregated_state

    def _save_checkpoint(self, path: Path, iteration: int):
        """保存 checkpoint。

        Args:
            path: 保存路径
            iteration: 当前迭代次数
        """
        torch.save({
            "model_state_dict": self.global_model_state,
            "iteration": iteration,
            "config": self.config,
        }, path)

    def shutdown(self):
        """关闭 Ray Workers。"""
        for worker in self.workers:
            ray.kill(worker)
        self.workers.clear()
