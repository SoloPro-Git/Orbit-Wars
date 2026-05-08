"""Inference predictor for Orbit Wars RL model.

Loads a trained checkpoint and generates Kaggle-format actions from observations.
"""
from __future__ import annotations

import sys
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

# Add project root to sys.path so that training.* imports work
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from training.core.config import ModelConfig
from training.core.feature_engineering import FeatureEngineer
from training.core.model import OrbitWarsModel
from training.core.action import decode_actions


class OrbitWarsPredictor:
    """Wraps a trained OrbitWarsModel for inference."""

    def __init__(
        self,
        checkpoint_path: str,
        config: ModelConfig | None = None,
        device: str = "cpu",
    ):
        """Load checkpoint and initialize model.

        Args:
            checkpoint_path: Path to a .pt file saved by torch.save.
            config: ModelConfig. If None, uses default values.
            device: Inference device, defaults to "cpu".
        """
        self.device = torch.device(device)
        self.config = config or ModelConfig()

        # Infer n_planets from checkpoint or use default
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Checkpoint may contain state_dict directly, or wrapped in a dict
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            # Try to restore config from checkpoint
            if "config" in checkpoint and config is None:
                saved_cfg = checkpoint["config"]
                if isinstance(saved_cfg, ModelConfig):
                    self.config = saved_cfg
            if "n_planets" in checkpoint:
                n_planets = checkpoint["n_planets"]
            else:
                n_planets = 32
        else:
            state_dict = checkpoint
            n_planets = 32

        # Build model and load weights
        self.model = OrbitWarsModel(self.config, n_planets=n_planets)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

        self.feature_engineer = FeatureEngineer()
        self.n_planets = n_planets

    def predict(self, obs: dict, player_id: int) -> list[list]:
        """Generate actions from an observation.

        Args:
            obs: Kaggle environment observation dict.
            player_id: Current player ID (0-3).

        Returns:
            actions: [[from_planet_id, angle, num_ships], ...]
        """
        try:
            # ---- 1. Feature extraction ----
            planet_features, fleet_features, global_features, metadata = (
                self.feature_engineer.compute(obs, player_id)
            )
        except Exception as e:
            print(f"[Predictor] Feature extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []

        # ---- 2. Numpy -> Torch tensors with batch dim ----
        planet_feat_t = torch.from_numpy(planet_features).unsqueeze(0).to(self.device)
        global_feat_t = torch.from_numpy(global_features).unsqueeze(0).to(self.device)

        # Handle empty fleet features
        if fleet_features.shape[0] > 0:
            fleet_feat_t = torch.from_numpy(fleet_features).unsqueeze(0).to(self.device)
        else:
            # Provide a [1, 0, D_FLEET] tensor so the model's empty-fleet path triggers
            fleet_feat_t = torch.zeros(1, 0, fleet_features.shape[1], device=self.device)

        # ---- 3. Build owned_mask and enemy_mask ----
        raw_planets = obs.get("planets", [])
        n_planets_actual = len(raw_planets)

        owned_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)
        enemy_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)

        for i, p in enumerate(raw_planets):
            owner = int(p[1])
            if owner == player_id:
                owned_mask[0, i] = True
            elif owner != -1:
                enemy_mask[0, i] = True

        # ---- 4. num_players ----
        owners = set()
        for p in raw_planets:
            o = int(p[1])
            if 0 <= o <= 3:
                owners.add(o)
        num_players_val = max(len(owners), 2)
        num_players = torch.tensor([num_players_val], dtype=torch.long, device=self.device)

        # ---- 5. planet_ships ----
        planet_ships = torch.tensor(
            [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=self.device
        )

        # ---- 6. Model forward ----
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
            n_model_planets = target_logits.shape[2]  # 模型输出的星球数（通常40）
            n_actual_planets = n_planets_actual

            if n_actual_planets < n_model_planets:
                # 截断到实际星球数
                target_logits = target_logits[:, :, :n_actual_planets]
            elif n_actual_planets > n_model_planets:
                # 填充（理论上不应该发生，因为max是40）
                pad_size = n_actual_planets - n_model_planets
                padding = torch.zeros(1, target_logits.shape[1], pad_size, device=self.device)
                target_logits = torch.cat([target_logits, padding], dim=2)

        # ---- 7. Convert to numpy ----
        target_logits_np = target_logits.cpu().numpy()  # [1, N_owned, N_planets]
        num_ships_np = num_ships_out.cpu().numpy()  # [1, N_owned, 1] (absolute ship counts)

        # Remove batch dim
        if target_logits_np.shape[1] == 0:
            # No owned planets -> no actions
            return []

        target_logits_np = target_logits_np[0]  # [N_owned, N_planets]
        num_ships_np = num_ships_np[0]  # [N_owned, 1]

        # ---- 8. Build owned_planets and all_planets dict lists ----
        all_planets: list[dict] = []
        owned_planets: list[dict] = []

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

        # ---- 9. Prepare num_ships_raw for decode_actions ----
        # 模型输出 num_ships_out 已经是绝对数量（sigmoid * owned_ships）
        # decode_actions 期望 [0,1] 比例值，所以需要转换回比例
        num_ships_raw_for_decode = np.zeros_like(num_ships_np)
        for i, src in enumerate(owned_planets):
            src_ships = src.get("ships", 0)
            if src_ships > 0:
                # 将绝对数量转换回比例（供 decode_actions 使用）
                num_ships_raw_for_decode[i, 0] = num_ships_np[i, 0] / src_ships
            else:
                num_ships_raw_for_decode[i, 0] = 0.0

        # ---- 10. Decode actions ----
        # 添加边界检查
        if len(owned_planets) == 0 or len(all_planets) == 0:
            return []

        if target_logits_np.shape[1] != len(all_planets):
            print(f"[Predictor] Warning: target_logits shape {target_logits_np.shape} != num_planets {len(all_planets)}")
            # 如果维度不匹配，返回空动作
            return []

        try:
            actions = decode_actions(
                target_logits=target_logits_np,
                num_ships_raw=num_ships_raw_for_decode,
                owned_planets=owned_planets,
                all_planets=all_planets,
                threshold=0.01,  # 降低阈值，让模型更容易行动
            )
        except Exception as e:
            print(f"[Predictor] Decode actions error: {e}")
            import traceback
            traceback.print_exc()
            actions = []

        return actions
