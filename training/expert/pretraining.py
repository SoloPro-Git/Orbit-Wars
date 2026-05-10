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
from training.core.feature_engineering import FeatureEngineer
from training.core.model import OrbitWarsModel
from training.expert.data_generator import ExpertDataset


class ExpertPretrainer:
    """专家数据预训练器。

    使用行为克隆训练模型模仿专家策略。
    """

    def __init__(
        self,
        model: OrbitWarsModel,
        config: ExpertDataConfig,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.config = config
        self.device = device

        # 特征提取器
        self.feature_engineer = FeatureEngineer()

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

        # SwanLab（延迟导入）
        self.swanlab = None

    def pretrain(
        self,
        num_iterations: int | None = None,
        checkpoint_dir: str = "training/checkpoints",
    ) -> dict:
        """执行预训练。

        Args:
            num_iterations: 训练迭代次数，None 则使用配置值。
            checkpoint_dir: checkpoint 保存目录。

        Returns:
            训练统计信息。
        """
        if num_iterations is None:
            num_iterations = self.config.num_pretrain_iterations

        # 从 checkpoint 恢复（如果配置了 pretrain_resume_from）
        resume_path = self.config.pretrain_resume_from
        if resume_path:
            resume_path = Path(resume_path)
            if resume_path.exists():
                ckpt = torch.load(resume_path, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state_dict, strict=False)
                # 如果有 optimizer 状态也恢复
                if isinstance(ckpt, dict) and "optimizer_state_dict" in ckpt:
                    self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
                start_iter = ckpt.get("iteration", 0) if isinstance(ckpt, dict) else 0
                print(f"[Pretrain] 已加载 checkpoint: {resume_path} (iteration={start_iter})")
            else:
                print(f"[Pretrain] 警告: checkpoint 不存在: {resume_path}，从头训练")

        # 延迟导入 SwanLab（需要外部先调用 swanlab.init()）
        self.swanlab = None
        try:
            import swanlab
            # 检查是否已初始化
            if hasattr(swanlab, 'get_run') and swanlab.get_run() is not None:
                self.swanlab = swanlab
        except (ImportError, Exception):
            pass

        print(f"\n开始专家数据预训练 ({num_iterations} 次迭代)")
        print(f"设备: {self.device}")
        print(f"Checkpoint 目录: {checkpoint_dir}")
        print("=" * 60)

        checkpoint_path = Path(checkpoint_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "train_loss": [],
            "val_loss": [],
        }

        for iteration in range(num_iterations):
            train_loss = self._train_iteration()
            stats["train_loss"].append(train_loss)

            if iteration % 10 == 0:
                val_loss = self._validate_iteration()
                stats["val_loss"].append(val_loss)

                print(
                    f"Iteration {iteration}/{num_iterations} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f}"
                )

                if self.swanlab:
                    self.swanlab.log({
                        "pretrain_iteration": iteration,
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                    })

            # 保存 checkpoint（每 50 次迭代）
            if (iteration + 1) % 50 == 0:
                ckpt_file = checkpoint_path / f"pretrain_iter_{iteration+1}.pkl"
                self._save_checkpoint(ckpt_file, iteration)
                print(f"  ✓ 保存 checkpoint: {ckpt_file}")

        # 保存最终模型
        final_ckpt = checkpoint_path / "pretrained_model.pkl"
        self._save_checkpoint(final_ckpt, num_iterations)
        print(f"\n✓ 预训练完成，模型已保存: {final_ckpt}")
        print("=" * 60)

        return stats

    def _save_checkpoint(self, path: Path, iteration: int):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "iteration": iteration,
            "config": self.config,
        }, path)

    def _prepare_batch_input(
        self, sample: dict
    ) -> tuple[torch.Tensor, ...] | None:
        """将单个样本转为模型输入张量。

        Returns:
            (planet_features, fleet_features, global_features,
             owned_mask, enemy_mask, num_players, planet_ships)
             或 None（如果数据无效）
        """
        obs = sample["observation"]
        player_id = sample.get("player_id", 0)

        # 特征提取
        planet_feat, fleet_feat, global_feat, metadata = (
            self.feature_engineer.compute(obs, player_id)
        )

        raw_planets = obs.get("planets", [])
        n = len(raw_planets)
        if n == 0:
            return None

        # 转 tensor
        planet_feat_t = torch.from_numpy(planet_feat).unsqueeze(0).float().to(self.device)

        if fleet_feat.shape[0] > 0:
            fleet_feat_t = torch.from_numpy(fleet_feat).unsqueeze(0).float().to(self.device)
        else:
            fleet_feat_t = torch.zeros(1, 0, fleet_feat.shape[1] if fleet_feat.size > 0 else 11,
                                        device=self.device)

        global_feat_t = torch.from_numpy(global_feat).unsqueeze(0).float().to(self.device)

        # 构建 mask
        owned_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
        enemy_mask = torch.zeros(1, n, dtype=torch.bool, device=self.device)
        for i, p in enumerate(raw_planets):
            owner = int(p[1])
            if owner == player_id:
                owned_mask[0, i] = True
            elif owner != -1:
                enemy_mask[0, i] = True

        planet_ships = torch.tensor(
            [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=self.device
        )
        owners = {int(p[1]) for p in raw_planets if int(p[1]) >= 0}
        num_players_t = torch.tensor([max(len(owners), 2)], dtype=torch.long, device=self.device)

        return (planet_feat_t, fleet_feat_t, global_feat_t,
                owned_mask, enemy_mask, num_players_t, planet_ships)

    def _train_iteration(self) -> float:
        """执行一次训练迭代。"""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        batch_size = self.config.pretrain_batch_size
        samples = self.train_ds.get_batch(batch_size, shuffle=True)

        for sample in samples:
            expert_actions = sample.get("actions", [])
            if not expert_actions:
                continue

            # 准备模型输入
            inputs = self._prepare_batch_input(sample)
            if inputs is None:
                continue

            # 前向传播
            self.optimizer.zero_grad()
            target_logits, num_ships_pred, value, _, _ = self.model(*inputs)

            # 计算行为克隆损失
            loss = self._compute_behavior_clone_loss(
                target_logits, num_ships_pred,
                expert_actions,
                sample["observation"],
                sample.get("player_id", 0),
            )

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1

        return total_loss / max(1, num_batches)

    def _validate_iteration(self) -> float:
        """执行一次验证迭代。"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0

        with torch.no_grad():
            batch_size = self.config.pretrain_batch_size
            samples = self.val_ds.get_batch(batch_size, shuffle=True)

            for sample in samples:
                expert_actions = sample.get("actions", [])
                if not expert_actions:
                    continue

                inputs = self._prepare_batch_input(sample)
                if inputs is None:
                    continue

                target_logits, num_ships_pred, value, _, _ = self.model(*inputs)

                loss = self._compute_behavior_clone_loss(
                    target_logits, num_ships_pred,
                    expert_actions,
                    sample["observation"],
                    sample.get("player_id", 0),
                )

                total_loss += loss.item()
                num_batches += 1

        return total_loss / max(1, num_batches)

    def _compute_behavior_clone_loss(
        self,
        target_logits: torch.Tensor,   # [1, N_owned, N_planets]
        num_ships_pred: torch.Tensor,   # [1, N_owned, 1] sigmoid 比例
        expert_actions: list,           # [[from_planet_id, angle, num_ships], ...]
        observation: dict,
        player_id: int,
    ) -> torch.Tensor:
        """计算行为克隆损失。

        将专家动作映射到模型的 target_logits 索引空间，然后计算
        target cross-entropy + ship ratio MSE。
        """
        raw_planets = observation.get("planets", [])
        if not raw_planets:
            return torch.tensor(0.0, device=self.device)

        # 构建星球索引映射
        all_planets_dicts = [
            {"id": int(p[0]), "owner": int(p[1]),
             "x": float(p[2]), "y": float(p[3]),
             "ships": float(p[5])}
            for p in raw_planets
        ]
        owned_planets = [p for p in all_planets_dicts if p["owner"] == player_id]
        if not owned_planets:
            return torch.tensor(0.0, device=self.device)

        # 模型输出是 gathered owned planets
        # owned_planets 的顺序应与 target_logits 的行对应
        all_id_to_idx = {p["id"]: i for i, p in enumerate(all_planets_dicts)}
        owned_id_to_row = {p["id"]: i for i, p in enumerate(owned_planets)}

        n_owned = len(owned_planets)
        n_all = len(all_planets_dicts)
        device = self.device

        target_loss = torch.tensor(0.0, device=device)
        ship_loss = torch.tensor(0.0, device=device)
        n_valid = 0

        for action in expert_actions:
            from_id, angle, num_ships = action[0], action[1], action[2]

            if from_id not in owned_id_to_row:
                continue

            src_row = owned_id_to_row[from_id]
            src = owned_planets[src_row]

            # 通过角度找到目标星球索引
            best_target_idx = None
            best_angle_diff = float("inf")

            for tgt in all_planets_dicts:
                if tgt["id"] == from_id:
                    continue
                dx = tgt["x"] - src["x"]
                dy = tgt["y"] - src["y"]
                tgt_angle = np.arctan2(dy, dx)
                diff = abs(tgt_angle - float(angle))
                if diff > np.pi:
                    diff = 2 * np.pi - diff
                if diff < best_angle_diff:
                    best_angle_diff = diff
                    best_target_idx = all_id_to_idx[tgt["id"]]

            if best_target_idx is None:
                continue

            # target cross-entropy loss
            if src_row < target_logits.shape[1]:
                logits_row = target_logits[0, src_row]  # [N_planets]
                target_loss += F.cross_entropy(
                    logits_row.unsqueeze(0),
                    torch.tensor([best_target_idx], device=device),
                )

            # ship ratio MSE
            if src["ships"] > 0:
                expert_ratio = min(float(num_ships) / src["ships"], 1.0)
            else:
                expert_ratio = 0.0

            if src_row < num_ships_pred.shape[1]:
                pred_ratio = num_ships_pred[0, src_row, 0]
                ship_loss += F.mse_loss(pred_ratio, torch.tensor(expert_ratio, device=device))

            n_valid += 1

        if n_valid == 0:
            return torch.tensor(0.0, device=self.device)

        total_loss = (target_loss + ship_loss) / n_valid
        return self.config.behavior_clone_loss_coef * total_loss


def mix_expert_data_with_rl(
    rl_batch: list,
    expert_dataset: ExpertDataset,
    mix_ratio: float,
) -> list:
    """混合 RL 数据和专家数据。"""
    if mix_ratio <= 0:
        return rl_batch

    batch_size = len(rl_batch)
    expert_size = int(batch_size * mix_ratio)

    expert_samples = expert_dataset.get_batch(expert_size, shuffle=True)
    mixed_batch = rl_batch + expert_samples
    random.shuffle(mixed_batch)

    return mixed_batch
