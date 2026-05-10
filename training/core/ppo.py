"""PPO 训练器 — 使用标准 PPO 算法，兼容自定义 Transformer 策略。"""
from __future__ import annotations

import copy
from typing import Optional
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.parallel import DistributedDataParallel as DDP

from core.config import ModelConfig, TrainingConfig
from core.model import OrbitWarsModel


class PPOBuffer:
    """存储 rollout 数据，支持 GAE 计算。"""

    def __init__(self):
        self.clear()

    def clear(self):
        self.planet_features = []
        self.fleet_features = []
        self.global_features = []
        self.owned_masks = []
        self.enemy_masks = []
        self.num_players_list = []
        self.planet_ships_list = []
        self.target_indices = []
        self.num_ships_actual = []
        self.log_probs = []
        self.values = []
        self.rewards = []
        self.dones = []
        self.opp_target_indices = []
        self.opp_num_ships_actual = []

    def add(
        self,
        planet_feat: np.ndarray,
        fleet_feat: np.ndarray,
        global_feat: np.ndarray,
        owned_mask: np.ndarray,
        enemy_mask: np.ndarray,
        num_players: int,
        planet_ships: np.ndarray,
        target_idx: np.ndarray,
        num_ships_act: np.ndarray,
        log_prob: float,
        value: np.ndarray,
        reward: float,
        done: bool,
        opp_target_idx: Optional[np.ndarray] = None,
        opp_num_ships_act: Optional[np.ndarray] = None,
    ):
        self.planet_features.append(planet_feat)
        self.fleet_features.append(fleet_feat)
        self.global_features.append(global_feat)
        self.owned_masks.append(owned_mask)
        self.enemy_masks.append(enemy_mask)
        self.num_players_list.append(num_players)
        self.planet_ships_list.append(planet_ships)
        self.target_indices.append(target_idx)
        self.num_ships_actual.append(num_ships_act)
        self.log_probs.append(log_prob)
        self.values.append(value)
        self.rewards.append(reward)
        self.dones.append(done)
        self.opp_target_indices.append(opp_target_idx)
        self.opp_num_ships_actual.append(opp_num_ships_act)

    def __len__(self) -> int:
        return len(self.rewards)


def compute_gae(
    rewards: list[float],
    values: list[np.ndarray],
    dones: list[bool],
    gamma: float = 0.99,
    gae_lambda: float = 0.95,
) -> tuple[np.ndarray, np.ndarray]:
    """计算 GAE (Generalized Advantage Estimation) 和 returns。

    Args:
        rewards: 每步奖励
        values: 每步价值估计 (标量)
        dones: 每步是否结束
        gamma: 折扣因子
        gae_lambda: GAE lambda

    Returns:
        advantages: [T]
        returns: [T]
    """
    T = len(rewards)
    advantages = np.zeros(T, dtype=np.float32)
    returns = np.zeros(T, dtype=np.float32)
    last_gae = 0.0

    # 转为标量列表
    vals = [float(v) if isinstance(v, np.ndarray) else v for v in values]

    for t in reversed(range(T)):
        if t == T - 1:
            next_value = 0.0
        else:
            next_value = vals[t + 1]

        next_non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * next_non_terminal - vals[t]
        last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + np.array(vals, dtype=np.float32)
    return advantages, returns


class PPOTrainer:
    """PPO 训练器，管理策略更新。"""

    def __init__(
        self,
        model: OrbitWarsModel,
        config: TrainingConfig,
        device: str = "cuda",
    ):
        self.model = model
        self.config = config
        self.device = device

        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=config.learning_rate, eps=1e-5
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=config.max_iterations
        )

    def update(self, buffer: PPOBuffer, show_progress: bool = True) -> dict:
        """从 buffer 数据执行 PPO 更新。

        Returns:
            训练指标 dict (policy_loss, value_loss, entropy, total_loss, etc.)
        """
        if len(buffer) == 0:
            return {}

        # 1. 计算 GAE
        advantages, returns = compute_gae(
            buffer.rewards,
            buffer.values,
            buffer.dones,
            gamma=self.config.gamma,
            gae_lambda=self.config.gae_lambda,
        )

        # 2. 构建训练数据
        old_log_probs = torch.tensor(
            np.array(buffer.log_probs, dtype=np.float32), device=self.device
        )
        advantages_t = torch.tensor(advantages, device=self.device)
        returns_t = torch.tensor(returns, device=self.device)

        # 优势归一化
        if advantages_t.std() > 1e-8:
            advantages_t = (advantages_t - advantages_t.mean()) / (
                advantages_t.std() + 1e-8
            )

        # 3. 收集所有观测数据
        planet_feats = [torch.tensor(f, dtype=torch.float32) for f in buffer.planet_features]
        fleet_feats = [torch.tensor(f, dtype=torch.float32) for f in buffer.fleet_features]
        global_feats = [torch.tensor(f, dtype=torch.float32) for f in buffer.global_features]
        owned_masks_t = [torch.tensor(m, dtype=torch.bool) for m in buffer.owned_masks]
        enemy_masks_t = [torch.tensor(m, dtype=torch.bool) for m in buffer.enemy_masks]
        num_players_t = torch.tensor(buffer.num_players_list, dtype=torch.long, device=self.device)
        planet_ships_t = [torch.tensor(s, dtype=torch.float32) for s in buffer.planet_ships_list]
        target_indices_t = [torch.tensor(t, dtype=torch.long) for t in buffer.target_indices]
        num_ships_actual_t = [torch.tensor(s, dtype=torch.float32) for s in buffer.num_ships_actual]

        # 4. PPO epochs
        total_metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "opp_loss": 0.0,
            "total_loss": 0.0,
        }

        T = len(buffer)
        batch_size = min(self.config.batch_size, T)

        # 创建 epoch 进度条
        epoch_iter = range(self.config.ppo_epochs)
        if show_progress:
            epoch_iter = tqdm(epoch_iter, desc="[PPO Update]", unit="epoch")

        for epoch in epoch_iter:
            indices = torch.randperm(T, device=self.device)[:batch_size]

            # 小批量前向
            metrics = self._ppo_step(
                planet_feats, fleet_feats, global_feats,
                owned_masks_t, enemy_masks_t, num_players_t, planet_ships_t,
                target_indices_t, num_ships_actual_t,
                old_log_probs, advantages_t, returns_t,
                indices,
            )

            # 更新进度条信息
            if show_progress and isinstance(epoch_iter, tqdm):
                epoch_iter.set_postfix({
                    'policy_loss': f"{metrics.get('policy_loss', 0):.4f}",
                    'value_loss': f"{metrics.get('value_loss', 0):.4f}",
                    'entropy': f"{metrics.get('entropy', 0):.4f}",
                    'batch_size': batch_size
                })

            for k, v in metrics.items():
                total_metrics[k] += v

        if show_progress and isinstance(epoch_iter, tqdm):
            epoch_iter.close()

        # 平均
        for k in total_metrics:
            total_metrics[k] /= self.config.ppo_epochs

        # LR schedule
        self.scheduler.step()
        total_metrics["lr"] = self.scheduler.get_last_lr()[0]

        return total_metrics

    def _ppo_step(
        self,
        planet_feats, fleet_feats, global_feats,
        owned_masks, enemy_masks, num_players, planet_ships,
        target_indices, num_ships_actual,
        old_log_probs, advantages, returns,
        indices,
    ) -> dict:
        """单步 PPO 更新。"""
        # Pad 序列到相同长度并取 mini-batch
        batch_pf = self._pad_and_index(planet_feats, indices).to(self.device)
        batch_ff = self._pad_and_index_fleets(fleet_feats, indices).to(self.device)
        batch_gf = torch.stack([global_feats[i] for i in indices]).to(self.device)
        batch_om = self._pad_bool_and_index(owned_masks, indices).to(self.device)
        batch_em = self._pad_bool_and_index(enemy_masks, indices).to(self.device)
        batch_np = num_players[indices]
        batch_ps = self._pad_and_index(planet_ships, indices).to(self.device)
        batch_ti = self._pad_long_and_index(target_indices, indices).to(self.device)
        batch_ns = self._pad_and_index(num_ships_actual, indices).to(self.device)

        batch_old_lp = old_log_probs[indices]
        batch_adv = advantages[indices]
        batch_ret = returns[indices]

        # 前向
        target_logits, num_ships_pred, value, opp_target, opp_num_ships = self.model(
            planet_features=batch_pf,
            fleet_features=batch_ff,
            global_features=batch_gf,
            owned_mask=batch_om,
            enemy_mask=batch_em,
            num_players=batch_np,
            planet_ships=batch_ps,
        )

        # Policy loss
        log_probs = self._compute_log_probs(
            target_logits, num_ships_pred, batch_ti, batch_ns
        )
        ratio = torch.exp(log_probs - batch_old_lp)
        surr1 = ratio * batch_adv
        surr2 = ratio.clamp(
            1.0 - self.config.ppo_clip, 1.0 + self.config.ppo_clip
        ) * batch_adv
        policy_loss = -torch.min(surr1, surr2).mean()

        # Value loss — value 已经是 [batch] 标量
        value_loss = F.mse_loss(value, batch_ret)

        # Entropy bonus
        entropy = self._compute_entropy(target_logits)
        entropy_loss = -self.config.entropy_coef * entropy

        # Opponent prediction loss (auxiliary)
        opp_loss = torch.tensor(0.0, device=self.device)
        if opp_target is not None:
            # 处理 DDP 包装的情况
            model = self.model.module if isinstance(self.model, DDP) else self.model
            if model.config.use_opponent_head:
                opp_loss = self._compute_opponent_loss(
                    opp_target, opp_num_ships, indices
                )

        # Total
        total_loss = (
            policy_loss
            + self.config.value_coef * value_loss
            + entropy_loss
            + self.config.opponent_pred_coef * opp_loss
        )

        self.optimizer.zero_grad()
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.max_grad_norm
        )
        self.optimizer.step()

        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
            "opp_loss": opp_loss.item(),
            "total_loss": total_loss.item(),
        }

    def compute_buffer_stats(self, buffer: PPOBuffer) -> dict:
        """计算buffer的统计信息，用于监控训练进度。

        Args:
            buffer: PPOBuffer经验数据

        Returns:
            统计信息dict，包含各种reward和性能指标
        """
        if len(buffer) == 0:
            return {}

        rewards = np.array(buffer.rewards)

        # 基础统计
        stats = {
            "mean_reward": float(np.mean(rewards)),
            "std_reward": float(np.std(rewards)),
            "min_reward": float(np.min(rewards)),
            "max_reward": float(np.max(rewards)),
            "total_reward": float(np.sum(rewards)),
            "num_steps": len(rewards),
        }

        # 分位数统计
        stats["median_reward"] = float(np.median(rewards))
        stats["percentile_25_reward"] = float(np.percentile(rewards, 25))
        stats["percentile_75_reward"] = float(np.percentile(rewards, 75))

        # 正奖励比例
        stats["positive_reward_ratio"] = float(np.mean(rewards > 0))

        # 大奖励比例（假设reward > 1.0为好表现）
        stats["high_reward_ratio"] = float(np.mean(rewards > 1.0))

        # 负奖励比例
        stats["negative_reward_ratio"] = float(np.mean(rewards < 0))

        return stats

    def _compute_log_probs(
        self,
        target_logits: torch.Tensor,
        num_ships_pred: torch.Tensor,
        target_indices: torch.Tensor,
        num_ships_actual: torch.Tensor,
    ) -> torch.Tensor:
        """计算动作 log probability。所有值都在 sigmoid 比例空间。

        target_logits:    [batch, N_owned, N_planets]
        num_ships_pred:   [batch, N_owned, 1]  sigmoid 比例
        target_indices:   [batch, N_owned]  采样的目标索引
        num_ships_actual: [batch, N_owned, 1] 或 [batch, N_owned]  采样后的 sigmoid 比例
        """
        B, N_owned, N_planets = target_logits.shape

        # target log prob
        log_softmax = F.log_softmax(target_logits, dim=-1)
        target_log_prob = log_softmax.gather(
            2, target_indices.clamp(0, N_planets - 1).unsqueeze(-1)
        ).squeeze(-1)  # [batch, N_owned]

        # num_ships log prob (高斯, sigmoid 比例空间)
        sigma = 0.15
        if num_ships_actual.dim() == 3:
            ship_diff = num_ships_actual - num_ships_pred  # [B, N_owned, 1]
        else:
            ship_diff = (num_ships_actual - num_ships_pred.squeeze(-1))  # [B, N_owned]
        ships_log_prob = -0.5 * (ship_diff / sigma).pow(2)

        # 合并 log probs 并取每样本平均
        log_prob = target_log_prob + ships_log_prob.squeeze(-1)  # [B, N_owned]
        return log_prob.mean(dim=-1)  # [batch]

    def _compute_entropy(self, target_logits: torch.Tensor) -> torch.Tensor:
        """计算策略熵。"""
        probs = F.softmax(target_logits, dim=-1)
        log_probs = F.log_softmax(target_logits, dim=-1)
        entropy = -(probs * log_probs).sum(dim=-1)
        return entropy.mean()

    def _compute_opponent_loss(self, opp_target, opp_num_ships, indices):
        """对手预测辅助损失 (placeholder, 需要 opp_target_indices)。"""
        # 简单的 entropy 正则化
        entropy = self._compute_entropy(opp_target)
        return -entropy * 0.1

    # --- Padding utilities ---

    @staticmethod
    def _pad_and_index(tensors: list[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        """Pad 变长 tensor 列表并按 indices 取 mini-batch。"""
        selected = [tensors[i] for i in indices.cpu().tolist()]

        # 检查是否为 1D 张量（如 planet_ships）
        if selected[0].dim() == 1:
            # 1D 张量：添加特征维度并 padding
            max_len = max(s.size(0) for s in selected)
            padded = torch.zeros(len(selected), max_len)
            for i, s in enumerate(selected):
                padded[i, :s.size(0)] = s
            return padded
        else:
            # 2D 张量：正常 padding
            max_len = max(s.size(0) for s in selected)
            feat_dim = selected[0].size(-1)
            padded = torch.zeros(len(selected), max_len, feat_dim)
            for i, s in enumerate(selected):
                padded[i, :s.size(0)] = s
            return padded

    @staticmethod
    def _pad_and_index_fleets(tensors: list[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        selected = [tensors[i] for i in indices.cpu().tolist()]
        if all(s.size(0) == 0 for s in selected):
            return torch.zeros(len(selected), 0, tensors[0].size(-1))
        max_len = max(s.size(0) for s in selected)
        feat_dim = selected[0].size(-1)
        padded = torch.zeros(len(selected), max_len, feat_dim)
        for i, s in enumerate(selected):
            if s.size(0) > 0:
                padded[i, :s.size(0)] = s
        return padded

    @staticmethod
    def _pad_bool_and_index(tensors: list[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        selected = [tensors[i] for i in indices.cpu().tolist()]
        max_len = max(s.size(0) for s in selected)
        padded = torch.zeros(len(selected), max_len, dtype=torch.bool)
        for i, s in enumerate(selected):
            padded[i, :s.size(0)] = s
        return padded

    @staticmethod
    def _pad_long_and_index(tensors: list[torch.Tensor], indices: torch.Tensor) -> torch.Tensor:
        selected = [tensors[i] for i in indices.cpu().tolist()]
        max_len = max(s.size(0) for s in selected)
        padded = torch.zeros(len(selected), max_len, dtype=torch.long)
        for i, s in enumerate(selected):
            padded[i, :s.size(0)] = s
        return padded
