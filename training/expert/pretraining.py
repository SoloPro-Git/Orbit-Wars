"""专家数据预训练模块。

使用专家演示数据进行行为克隆预训练，为强化学习提供良好的初始化。
"""
from __future__ import annotations

import random
from pathlib import Path
from typing import Any
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from training.core.config import ExpertDataConfig
from training.core.feature_engineering import FeatureEngineer
from training.core.model import OrbitWarsModel
from training.expert.action_labeling import infer_target_planet_id
from training.expert.data_generator import ExpertDataset


def resolve_stage_checkpoint_dir(base_dir: str | Path, config: ExpertDataConfig, stage: str | None = None) -> Path:
    """按 stage + 人局过滤构造 ckpt 目录名。"""
    base = Path(base_dir)
    stg = (stage or getattr(config, "active_stage", "AUTO")).upper().strip()
    if stg == "A":
        npf = int(getattr(config, "stage_a_num_players", 0))
    elif stg == "B":
        npf = int(getattr(config, "stage_b_num_players", 0))
    elif stg == "C":
        npf = int(getattr(config, "stage_c_num_players", 0))
    else:
        npf = 0
    suffix = f"{npf}p" if npf > 0 else "allp"
    return base / f"stage_{stg}_{suffix}"


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
        self.dataset = ExpertDataset(
            data_dir=config.data_dir,
            extra_data_dirs=getattr(config, "extra_data_dirs", []),
        )
        self.train_ds, self.val_ds = self.dataset.split(train_ratio=0.8)

        print(f"训练集: {len(self.train_ds)} 样本")
        print(f"验证集: {len(self.val_ds)} 样本")

        # 创建优化器
        lr = float(config.pretrain_learning_rate)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=lr,
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

        stage_for_dir = str(self.config.active_stage).upper().strip() if self.config.staged_mode else "AUTO"
        checkpoint_path = resolve_stage_checkpoint_dir(checkpoint_dir, self.config, stage_for_dir)
        checkpoint_path.mkdir(parents=True, exist_ok=True)

        stats = {
            "train_loss": [],
            "val_loss": [],
        }

        if self.config.staged_mode:
            active_stage = str(self.config.active_stage).upper().strip()
            print(f"[Pretrain] staged_mode=True | active_stage={active_stage}")
        else:
            stage_spans = self._build_stage_spans(num_iterations)
            print(f"[Pretrain] Stage spans: {stage_spans}")

        gate_passed = False
        last_val_metrics: dict[str, float] = {}

        for iteration in range(num_iterations):
            if self.config.staged_mode:
                stage = str(self.config.active_stage).upper().strip()
            else:
                stage = self._stage_for_iteration(iteration, stage_spans)
            train_metrics = self._train_iteration(stage=stage)
            train_loss = float(train_metrics.get("loss_total", 0.0))
            stats["train_loss"].append(train_loss)

            if iteration % max(1, int(self.config.stage_eval_interval)) == 0:
                val_metrics = self._validate_iteration(stage=stage)
                val_loss = float(val_metrics.get("loss_total", 0.0))
                last_val_metrics = val_metrics
                stats["val_loss"].append(val_loss)

                print(
                    f"Iteration {iteration}/{num_iterations} | Stage: {stage} | "
                    f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                    f"send_acc={val_metrics.get('send_acc', 0.0):.3f} "
                    f"ship_mae={val_metrics.get('ship_mae', 0.0):.3f} "
                    f"target_top1={val_metrics.get('target_top1', 0.0):.3f}"
                )

                if self.swanlab:
                    stage_prefix = f"stage_{stage}"
                    log_payload = {
                        "pretrain_iteration": iteration,
                        "stage_id": stage,
                        "train/loss_total": train_loss,
                        "val/loss_total": val_loss,
                    }
                    for k, v in train_metrics.items():
                        if k == "loss_total":
                            continue
                        log_payload[f"train/{k}"] = float(v)
                        log_payload[f"{stage_prefix}/train/{k}"] = float(v)
                    for k, v in val_metrics.items():
                        if k == "loss_total":
                            continue
                        log_payload[f"val/{k}"] = float(v)
                        log_payload[f"{stage_prefix}/val/{k}"] = float(v)
                    self.swanlab.log(log_payload)

                if self.config.staged_mode and iteration >= int(self.config.stage_min_iterations):
                    gate_passed = self._stage_gate_passed(stage, val_metrics)
                    print(
                        f"[Pretrain][Gate] stage={stage} passed={gate_passed} "
                        f"(iter={iteration})"
                    )
                    if gate_passed:
                        print(f"[Pretrain] Stage {stage} 达标，提前停止。")
                        break

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

        stats["stage_id"] = str(self.config.active_stage).upper().strip() if self.config.staged_mode else "AUTO"
        stats["gate_passed"] = gate_passed
        if last_val_metrics:
            stats["final_val_metrics"] = last_val_metrics
        return stats

    def _stage_gate_passed(self, stage: str, val_metrics: dict[str, float]) -> bool:
        stage = stage.upper().strip()
        if stage == "A":
            return float(val_metrics.get("send_acc", 0.0)) >= float(self.config.stage_a_send_acc_threshold)
        if stage == "B":
            return float(val_metrics.get("ship_mae", 999.0)) <= float(self.config.stage_b_ship_mae_threshold)
        if stage == "C":
            return float(val_metrics.get("target_top1", 0.0)) >= float(self.config.stage_c_target_top1_threshold)
        return False

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

    def _build_stage_spans(self, num_iterations: int) -> dict[str, tuple[int, int]]:
        """构建 A/B/C 阶段在总迭代中的区间。"""
        a = max(1, int(num_iterations * self.config.stage_a_ratio))
        b = max(1, int(num_iterations * self.config.stage_b_ratio))
        c = max(1, num_iterations - a - b)
        if a + b + c > num_iterations:
            c = max(1, num_iterations - a - b)
        return {
            "A": (0, a),
            "B": (a, a + b),
            "C": (a + b, num_iterations),
        }

    @staticmethod
    def _stage_for_iteration(iteration: int, spans: dict[str, tuple[int, int]]) -> str:
        if spans["A"][0] <= iteration < spans["A"][1]:
            return "A"
        if spans["B"][0] <= iteration < spans["B"][1]:
            return "B"
        return "C"

    def _train_iteration(self, stage: str) -> dict[str, float]:
        """执行一次训练迭代。"""
        self.model.train()
        agg = {
            "loss_total": 0.0,
            "loss_send": 0.0,
            "loss_ship": 0.0,
            "loss_target": 0.0,
            "send_acc": 0.0,
            "ship_mae": 0.0,
            "target_top1": 0.0,
            "multi_target_source_rate": 0.0,
        }
        num_batches = 0

        batch_size = self.config.pretrain_batch_size
        samples = self.train_ds.get_batch(batch_size, shuffle=True)

        for sample in samples:
            if not self._sample_matches_stage(sample, stage):
                continue
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

            # 计算分阶段行为克隆损失（Stage C 多目标标签实时构造）
            metrics = self._compute_behavior_clone_loss(
                target_logits, num_ships_pred,
                expert_actions,
                sample["observation"],
                sample.get("player_id", 0),
                stage=stage,
            )
            loss = metrics["loss_total"]

            if loss.requires_grad:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    max_norm=1.0,
                )
                self.optimizer.step()

            for k in agg:
                agg[k] += float(metrics.get(k, 0.0))
            num_batches += 1

        if num_batches == 0:
            return agg
        return {k: v / num_batches for k, v in agg.items()}

    def _validate_iteration(self, stage: str) -> dict[str, float]:
        """执行一次验证迭代。"""
        self.model.eval()
        agg = {
            "loss_total": 0.0,
            "loss_send": 0.0,
            "loss_ship": 0.0,
            "loss_target": 0.0,
            "send_acc": 0.0,
            "ship_mae": 0.0,
            "target_top1": 0.0,
            "multi_target_source_rate": 0.0,
        }
        num_batches = 0

        with torch.no_grad():
            batch_size = self.config.pretrain_batch_size
            samples = self.val_ds.get_batch(batch_size, shuffle=True)

            for sample in samples:
                if not self._sample_matches_stage(sample, stage):
                    continue
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
                    stage=stage,
                )

                for k in agg:
                    agg[k] += float(loss.get(k, 0.0))
                num_batches += 1

        if num_batches == 0:
            return agg
        return {k: v / num_batches for k, v in agg.items()}

    def _compute_behavior_clone_loss(
        self,
        target_logits: torch.Tensor,   # [1, N_owned, N_planets]
        num_ships_pred: torch.Tensor,   # [1, N_owned, 1] sigmoid 比例
        expert_actions: list,           # [[from_planet_id, angle, num_ships], ...]
        observation: dict,
        player_id: int,
        stage: str = "C",
    ) -> dict[str, torch.Tensor | float]:
        """计算行为克隆损失。

        将专家动作映射到模型的 target_logits 索引空间，然后计算
        target cross-entropy + ship ratio MSE。
        """
        raw_planets = observation.get("planets", [])
        if not raw_planets:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_total": zero,
                "loss_send": zero,
                "loss_ship": zero,
                "loss_target": zero,
                "send_acc": 0.0,
                "ship_mae": 0.0,
                "target_top1": 0.0,
                "multi_target_source_rate": 0.0,
            }

        # 构建星球索引映射
        all_planets_dicts = [
            {"id": int(p[0]), "owner": int(p[1]),
             "x": float(p[2]), "y": float(p[3]),
             "ships": float(p[5])}
            for p in raw_planets
        ]
        owned_planets = [p for p in all_planets_dicts if p["owner"] == player_id]
        if not owned_planets:
            zero = torch.tensor(0.0, device=self.device)
            return {
                "loss_total": zero,
                "loss_send": zero,
                "loss_ship": zero,
                "loss_target": zero,
                "send_acc": 0.0,
                "ship_mae": 0.0,
                "target_top1": 0.0,
                "multi_target_source_rate": 0.0,
            }

        # 模型输出是 gathered owned planets
        # owned_planets 的顺序应与 target_logits 的行对应
        all_id_to_idx = {p["id"]: i for i, p in enumerate(all_planets_dicts)}
        owned_id_to_row = {p["id"]: i for i, p in enumerate(owned_planets)}

        device = self.device
        n_owned = len(owned_planets)
        n_all = len(all_planets_dicts)

        send_label = torch.zeros(n_owned, dtype=torch.float32, device=device)
        ship_ratio_label = torch.zeros(n_owned, dtype=torch.float32, device=device)
        target_dist_label = torch.zeros(n_owned, n_all, dtype=torch.float32, device=device)

        grouped_actions: dict[int, list[tuple[float, float]]] = {}
        for action in expert_actions:
            from_id, angle, num_ships = action[0], action[1], action[2]
            grouped_actions.setdefault(int(from_id), []).append((float(angle), float(num_ships)))

        multi_target_sources = 0
        valid_sources = 0

        for src in owned_planets:
            src_id = int(src["id"])
            src_row = owned_id_to_row[src_id]
            actions = grouped_actions.get(src_id, [])
            if not actions:
                continue
            send_label[src_row] = 1.0
            valid_sources += 1
            if len(actions) >= 2:
                multi_target_sources += 1

            src_ships = max(float(src["ships"]), 1.0)
            total_send = 0.0
            for angle, ships in actions:
                target_pid = infer_target_planet_id(
                    observation=observation,
                    from_planet_id=src_id,
                    angle=angle,
                    num_ships=ships,
                )
                tgt_idx = all_id_to_idx.get(target_pid) if target_pid is not None else None
                if tgt_idx is None:
                    continue
                target_dist_label[src_row, tgt_idx] += max(ships, 0.0)
                total_send += max(ships, 0.0)

            if total_send > 0:
                target_dist_label[src_row] = target_dist_label[src_row] / total_send
            ship_ratio_label[src_row] = float(min(total_send / src_ships, 1.0))

        pred_ship_ratio = num_ships_pred[0, :n_owned, 0].clamp(0.0, 1.0)
        pred_send_score = pred_ship_ratio
        pred_target_logits = target_logits[0, :n_owned, :n_all]

        loss_send = F.binary_cross_entropy(pred_send_score, send_label)
        send_acc = ((pred_send_score >= 0.5) == (send_label >= 0.5)).float().mean().item()

        send_mask = (send_label > 0.5).float()
        if send_mask.sum() > 0:
            ship_diff = (pred_ship_ratio - ship_ratio_label).abs() * send_mask
            ship_mae = (ship_diff.sum() / send_mask.sum()).item()
            loss_ship = F.smooth_l1_loss(
                pred_ship_ratio * send_mask,
                ship_ratio_label * send_mask,
                reduction="sum",
            ) / send_mask.sum().clamp(min=1.0)
        else:
            ship_mae = 0.0
            loss_ship = torch.tensor(0.0, device=device)

        target_rows = (target_dist_label.sum(dim=-1) > 0).float()
        if target_rows.sum() > 0:
            logp = F.log_softmax(pred_target_logits, dim=-1)
            per_row_kl = -(target_dist_label * logp).sum(dim=-1)
            loss_target = (per_row_kl * target_rows).sum() / target_rows.sum().clamp(min=1.0)
            label_top1 = target_dist_label.argmax(dim=-1)
            pred_top1 = pred_target_logits.argmax(dim=-1)
            top1 = (((label_top1 == pred_top1).float() * target_rows).sum() / target_rows.sum().clamp(min=1.0)).item()
        else:
            loss_target = torch.tensor(0.0, device=device)
            top1 = 0.0

        if stage == "A":
            total_loss = loss_send
        elif stage == "B":
            total_loss = loss_send + loss_ship
        else:
            total_loss = loss_send + loss_ship + loss_target

        total_loss = self.config.behavior_clone_loss_coef * total_loss
        multi_rate = float(multi_target_sources / max(valid_sources, 1))
        return {
            "loss_total": total_loss,
            "loss_send": loss_send.detach().item(),
            "loss_ship": loss_ship.detach().item(),
            "loss_target": loss_target.detach().item(),
            "send_acc": send_acc,
            "ship_mae": ship_mae,
            "target_top1": top1,
            "multi_target_source_rate": multi_rate,
        }

    @staticmethod
    def _infer_num_players_from_obs(obs: dict) -> int:
        players = set()
        for p in obs.get("planets", []):
            owner = int(p[1])
            if owner >= 0:
                players.add(owner)
        return max(len(players), 2)

    def _stage_num_players_filter(self, stage: str) -> int:
        stage = stage.upper().strip()
        if stage == "A":
            return int(getattr(self.config, "stage_a_num_players", 0))
        if stage == "B":
            return int(getattr(self.config, "stage_b_num_players", 0))
        return int(getattr(self.config, "stage_c_num_players", 0))

    def _sample_matches_stage(self, sample: dict, stage: str) -> bool:
        required = self._stage_num_players_filter(stage)
        if required <= 0:
            return True
        obs = sample.get("observation", {})
        return self._infer_num_players_from_obs(obs) == required


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
