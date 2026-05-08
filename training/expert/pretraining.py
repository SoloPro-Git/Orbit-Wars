"""专家数据预训练模块。

使用专家演示数据进行行为克隆预训练，为强化学习提供良好的初始化。
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from training.core.config import ExpertDataConfig
from training.expert.data_generator import ExpertDataset


class ExpertPretrainer:
    """专家数据预训练器。

    使用行为克隆训练模型模仿专家策略。
    """

    def __init__(
        self,
        model: nn.Module,
        config: ExpertDataConfig,
        device: str = "cuda",
    ):
        """初始化预训练器。

        Args:
            model: 要训练的模型。
            config: 专家数据配置。
            device: 训练设备。
        """
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 加载数据集
        print(f"加载专家数据: {config.data_dir}")
        self.dataset = ExpertDataset(data_dir=config.data_dir)
        self.train_ds, self.val_ds = self.dataset.split(train_ratio=0.8)

        print(f"训练集: {len(self.train_ds)} 样本")
        print(f"验证集: {len(self.val_ds)} 样本")

        # 创建优化器
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.pretrain_learning_rate,
        )

    def pretrain(
        self,
        feature_extractor,
        num_iterations: int | None = None,
    ) -> dict:
        """执行预训练。

        Args:
            feature_extractor: 特征提取器。
            num_iterations: 训练迭代次数，None 则使用配置值。

        Returns:
            训练统计信息。
        """
        if num_iterations is None:
            num_iterations = self.config.num_pretrain_iterations

        print(f"\n开始专家数据预训练 ({num_iterations} 次迭代)")
        print("=" * 60)

        stats = {
            "train_loss": [],
            "val_loss": [],
        }

        for iteration in range(num_iterations):
            # 训练阶段
            train_loss = self._train_iteration(feature_extractor)
            stats["train_loss"].append(train_loss)

            # 验证阶段
            if iteration % 10 == 0:
                val_loss = self._validate_iteration(feature_extractor)
                stats["val_loss"].append(val_loss)

                print(
                    f"Iteration {iteration}/{num_iterations} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

        print("\n✓ 预训练完成")
        return stats

    def _train_iteration(self, feature_extractor) -> float:
        """执行一次训练迭代。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        # 采样批次
        batch_size = self.config.pretrain_batch_size
        samples = self.train_ds.get_batch(batch_size, shuffle=True)

        for sample in samples:
            # 提取特征
            obs = sample["observation"]
            features = feature_extractor.extract_features(obs, sample["player_id"])

            # 转换为张量
            features = torch.from_numpy(features).float().to(self.device)

            # 获取专家动作
            expert_actions = sample["actions"]
            if not expert_actions:
                continue

            # 前向传播
            self.optimizer.zero_grad()

            # 模型输出 (batch_size, num_planets * 2)
            # 假设模型输出目标星球的 logits 和飞船比例
            model_output = self.model(features.unsqueeze(0))

            # 计算行为克隆损失
            loss = self._compute_behavior_clone_loss(
                model_output,
                expert_actions,
                obs,
                sample["player_id"],
            )

            # 反向传播
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=1.0,
            )
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _validate_iteration(self, feature_extractor) -> float:
        """执行一次验证迭代。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            batch_size = self.config.pretrain_batch_size
            samples = self.val_ds.get_batch(batch_size, shuffle=True)

            for sample in samples:
                obs = sample["observation"]
                features = feature_extractor.extract_features(obs, sample["player_id"])
                features = torch.from_numpy(features).float().to(self.device)

                expert_actions = sample["actions"]
                if not expert_actions:
                    continue

                model_output = self.model(features.unsqueeze(0))
                loss = self._compute_behavior_clone_loss(
                    model_output,
                    expert_actions,
                    obs,
                    sample["player_id"],
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def _compute_behavior_clone_loss(
        self,
        model_output: torch.Tensor,
        expert_actions: list,
        observation: dict,
        player_id: int,
    ) -> torch.Tensor:
        """计算行为克隆损失。

        将专家动作转换为模型需要的格式并计算损失。
        """
        # 解析观测
        planets = observation.get("planets", [])
        fleets = observation.get("fleets", [])

        # 获取我方星球和所有星球
        owned_planets = []
        all_planets = []

        for p in planets:
            planet_dict = {
                "id": int(p[0]),
                "owner": int(p[1]),
                "x": float(p[2]),
                "y": float(p[3]),
                "radius": float(p[4]),
                "ships": float(p[5]),
                "production": float(p[6]),
            }
            all_planets.append(planet_dict)
            if planet_dict["owner"] == player_id:
                owned_planets.append(planet_dict)

        if not owned_planets or not all_planets:
            return torch.tensor(0.0, device=self.device)

        num_owned = len(owned_planets)
        num_all = len(all_planets)

        # 构建专家动作的目标
        # 专家动作格式: [[from_planet_id, angle, num_ships], ...]
        # 我们需要为每个源星球创建目标分布

        # 创建源星球ID到索引的映射
        owned_id_to_idx = {p["id"]: i for i, p in enumerate(owned_planets)}
        all_id_to_idx = {p["id"]: i for i, p in enumerate(all_planets)}

        # 初始化目标矩阵
        target_targets = torch.zeros(num_owned, num_all, device=self.device)
        target_ships = torch.zeros(num_owned, 1, device=self.device)

        # 填充专家动作
        for action in expert_actions:
            from_id, angle, num_ships = action

            if from_id not in owned_id_to_idx:
                continue

            source_idx = owned_id_to_idx[from_id]

            # 计算目标星球（基于角度）
            source_planet = owned_planets[source_idx]

            # 找到角度对应的目标星球
            best_target_idx = None
            best_angle_diff = float("inf")

            for target_planet in all_planets:
                if target_planet["id"] == from_id:
                    continue

                dx = target_planet["x"] - source_planet["x"]
                dy = target_planet["y"] - source_planet["y"]
                target_angle = np.arctan2(dy, dx)

                angle_diff = abs(target_angle - angle)
                if angle_diff > np.pi:
                    angle_diff = 2 * np.pi - angle_diff

                if angle_diff < best_angle_diff:
                    best_angle_diff = angle_diff
                    best_target_idx = all_id_to_idx[target_planet["id"]]

            if best_target_idx is not None:
                target_targets[source_idx, best_target_idx] = 1.0
                # 飞船数比例
                if source_planet["ships"] > 0:
                    ship_ratio = num_ships / source_planet["ships"]
                    target_ships[source_idx, 0] = ship_ratio

        # 模型输出应该包含:
        # 1. 目标星球的 logits (num_owned, num_all)
        # 2. 飞船比例 (num_owned, 1)

        # 假设 model_output 的形状为 (1, num_owned * (num_all + 1))
        # 前 num_owned * num_all 是目标 logits
        # 最后 num_owned 是飞船比例

        output = model_output.squeeze(0)  # (num_owned * (num_all + 1),)

        # 分离输出
        target_logits = output[: num_owned * num_all].view(num_owned, num_all)
        ship_logits = output[num_owned * num_all :].view(num_owned, 1)

        # 飞船比例使用 sigmoid
        ship_pred = torch.sigmoid(ship_logits)

        # 计算损失
        # 目标分类损失 (交叉熵)
        target_loss = F.cross_entropy(target_logits, target_targets.argmax(dim=1))

        # 飞船数量损失 (MSE)
        ship_loss = F.mse_loss(ship_pred, target_ships)

        # 总损失
        total_loss = (
            self.config.behavior_clone_loss_coef * (target_loss + ship_loss)
        )

        return total_loss


def mix_expert_data_with_rl(
    rl_batch: list,
    expert_dataset: ExpertDataset,
    mix_ratio: float,
) -> list:
    """混合 RL 数据和专家数据。

    Args:
        rl_batch: RL 采集的数据批次。
        expert_dataset: 专家数据集。
        mix_ratio: 专家数据的混合比例。

    Returns:
        混合后的数据批次。
    """
    if mix_ratio <= 0:
        return rl_batch

    batch_size = len(rl_batch)
    expert_size = int(batch_size * mix_ratio)

    # 采样专家数据
    expert_samples = expert_dataset.get_batch(expert_size, shuffle=True)

    # 随机打乱并合并
    mixed_batch = rl_batch + expert_samples
    random.shuffle(mixed_batch)

    return mixed_batch
