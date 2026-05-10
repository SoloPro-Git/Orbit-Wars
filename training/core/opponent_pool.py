"""对手池管理 — Kaggle 专家 + debug_eval 专家 + 历史 checkpoint 三类对手。

采样比例: KaggleExpert 0.1 / nearest_planet(debug_eval) 0.1 / 历史 checkpoint 0.8
"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, asdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np


@dataclass
class PoolEntry:
    """对手池中的一个策略 checkpoint。"""
    path: str
    elo: float = 600.0
    generation: int = 0
    timestamp: float = 0.0
    win_count: int = 0
    loss_count: int = 0
    draw_count: int = 0

    @property
    def total_games(self) -> int:
        return self.win_count + self.loss_count + self.draw_count

    @property
    def win_rate(self) -> float:
        if self.total_games == 0:
            return 0.0
        return self.win_count / self.total_games


@dataclass
class OpponentPoolConfig:
    pool_size: int = 64
    # 三类对手的采样比例
    sample_expert_ratio: float = 0.1
    sample_heuristic_ratio: float = 0.1  # 兼容旧配置；现在表示 debug_eval 专家比例
    sample_checkpoint_ratio: float = 0.8
    # checkpoint 内部再分: latest / best / random
    checkpoint_latest_ratio: float = 0.3
    checkpoint_best_ratio: float = 0.3
    checkpoint_random_ratio: float = 0.4
    add_to_pool_win_rate: float = 0.45
    diversity_threshold: float = 0.3


class OpponentPool:
    """管理三类对手来源，返回可调用的 agent 函数。

    专家策略:
      - KaggleExpertAgent (training/expert/kaggle_expert.py)
      - nearest_planet_agent (debug_eval.py)

    历史 checkpoint:
      - 保存的 model checkpoint 文件
    """

    def __init__(self, config: OpponentPoolConfig = None,
                 pool_dir: str | Path = "checkpoints/pool"):
        self.config = config or OpponentPoolConfig()
        self.pool: list[PoolEntry] = []
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)

        # 延迟初始化的专家 agent 列表（名字 -> factory/callable）
        self._kaggle_experts: dict[str, Callable] = {}
        self._debug_experts: dict[str, Callable] = {}
        # 兼容旧的外部访问/索引格式
        self._expert_agents: dict[str, Callable] = {}

    @property
    def size(self) -> int:
        return len(self.pool)

    @property
    def best(self) -> Optional[PoolEntry]:
        if not self.pool:
            return None
        return max(self.pool, key=lambda e: e.elo)

    @property
    def latest(self) -> Optional[PoolEntry]:
        if not self.pool:
            return None
        return max(self.pool, key=lambda e: e.generation)

    def register_expert(self, name: str, agent_fn: Callable):
        """注册专家策略 agent 函数。

        agent_fn 签名: (obs: dict) -> list[list]
        """
        self._kaggle_experts[name] = agent_fn
        self._expert_agents[name] = agent_fn

    def register_debug_expert(self, name: str, agent_fn: Callable):
        """注册 debug_eval 中的专家/基准策略。"""
        self._debug_experts[name] = agent_fn
        self._expert_agents[name] = agent_fn

    def register_default_experts(self):
        """注册默认的专家策略。"""
        # KaggleExpert
        try:
            from training.expert.kaggle_expert import KaggleExpertAgent

            def _make_kaggle_expert(player_id: int = 0):
                agent = KaggleExpertAgent(player_id=player_id)
                def _call(obs):
                    return agent.get_actions(obs)
                return _call

            self.register_expert("kaggle_expert", _make_kaggle_expert)
        except ImportError:
            pass

        # nearest_planet (优先使用 debug_eval.py 中的实现)
        try:
            from debug_eval import nearest_planet_agent

            self.register_debug_expert("nearest_planet", lambda pid=0: nearest_planet_agent)
        except ImportError:
            def _nearest_planet(obs: dict) -> list[list]:
                import math
                player = obs.get("player", 0)
                raw_planets = obs.get("planets", [])
                my_planets = [p for p in raw_planets if int(p[1]) == player]
                targets = [p for p in raw_planets if int(p[1]) != player]
                if not targets:
                    return []
                moves = []
                for mine in my_planets:
                    nearest = min(
                        targets,
                        key=lambda t: math.hypot(float(mine[2]) - float(t[2]), float(mine[3]) - float(t[3])),
                    )
                    ships_needed = int(float(nearest[5])) + 1
                    if float(mine[5]) >= ships_needed:
                        angle = math.atan2(float(nearest[3]) - float(mine[3]), float(nearest[2]) - float(mine[2]))
                        moves.append([int(mine[0]), angle, ships_needed])
                return moves

            self.register_debug_expert("nearest_planet", lambda pid=0: _nearest_planet)

    def add(self, checkpoint_path: str | Path, elo: float = 600.0,
            generation: int = 0) -> PoolEntry:
        """添加新策略到对手池。"""
        checkpoint_path = Path(checkpoint_path)
        path_str = str(checkpoint_path)
        for entry in self.pool:
            if entry.path == path_str:
                entry.elo = max(entry.elo, elo)
                entry.generation = max(entry.generation, generation)
                self._save_index()
                return entry

        entry = PoolEntry(
            path=path_str,
            elo=elo,
            generation=generation,
            timestamp=time.time(),
        )
        self.pool.append(entry)

        if len(self.pool) > self.config.pool_size:
            self._evict()

        self._save_index()
        return entry

    def add_checkpoint_dir(
        self,
        checkpoint_dir: str | Path,
        patterns: tuple[str, ...] = ("model_iter_*.pt", "pretrain_iter_*.pkl", "pretrained_model.pkl"),
        elo: float = 600.0,
        generation: int = 0,
    ) -> int:
        """把目录中已有 checkpoint 注册为历史对手，跳过重复和不存在文件。"""
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            return 0

        before = len(self.pool)
        for pattern in patterns:
            for ckpt_file in sorted(checkpoint_dir.glob(pattern)):
                if ckpt_file.is_file():
                    self.add(ckpt_file, elo=elo, generation=generation)
        return len(self.pool) - before

    def sample_opponent(self, player_id: int = 0) -> Callable:
        """采样一个对手 agent 函数。

        返回: (obs: dict) -> list[list]
        """
        source = self._pick_source()

        if source == "expert" and self._kaggle_experts:
            return self._instantiate_agent(random.choice(list(self._kaggle_experts.values())), player_id)

        if source == "debug_expert" and self._debug_experts:
            return self._instantiate_agent(random.choice(list(self._debug_experts.values())), player_id)

        # checkpoint: 加载模型并返回 agent
        if source == "checkpoint" and self.pool:
            entry = self._pick_checkpoint()
            if entry is not None:
                return _make_checkpoint_agent(entry.path, player_id)

        # fallback
        if self.pool:
            entry = random.choice(self.pool)
            return _make_checkpoint_agent(entry.path, player_id)

        return _random_agent

    @staticmethod
    def _instantiate_agent(factory: Callable, player_id: int) -> Callable:
        try:
            return factory(player_id)
        except TypeError:
            return factory

    def sample_opponents_for_game(self, num_players: int,
                                   training_player: int = 0) -> dict[int, Callable]:
        """为一局游戏采样所有非训练位置的对手。

        Returns:
            {player_id: agent_fn} 只包含非训练位置的对手
        """
        opponents = {}
        for pid in range(num_players):
            if pid == training_player:
                continue
            opponents[pid] = self.sample_opponent(player_id=pid)
        return opponents

    def should_add(self, win_rate: float) -> bool:
        if not self.pool:
            return True
        return win_rate >= self.config.add_to_pool_win_rate

    def update_elo(self, entry_path: str, result: float, k: float = 32.0):
        for entry in self.pool:
            if entry.path == entry_path:
                expected = 1.0 / (1.0 + 10 ** ((600 - entry.elo) / 400))
                entry.elo += k * (result - expected)
                if result > 0.5:
                    entry.win_count += 1
                elif result < 0.5:
                    entry.loss_count += 1
                else:
                    entry.draw_count += 1
                break
        self._save_index()

    def _pick_source(self) -> str:
        """按 0.1/0.1/0.8 比例选择 Kaggle/debug/checkpoint 来源。"""
        cfg = self.config
        choices: list[tuple[str, float]] = []
        if self._kaggle_experts and cfg.sample_expert_ratio > 0:
            choices.append(("expert", cfg.sample_expert_ratio))
        if self._debug_experts and cfg.sample_heuristic_ratio > 0:
            choices.append(("debug_expert", cfg.sample_heuristic_ratio))
        if self.pool and cfg.sample_checkpoint_ratio > 0:
            choices.append(("checkpoint", cfg.sample_checkpoint_ratio))

        if not choices:
            return "checkpoint"

        total = sum(weight for _, weight in choices)
        r = random.random() * total
        upto = 0.0
        for source, weight in choices:
            upto += weight
            if r <= upto:
                return source
        return choices[-1][0]

    def _pick_checkpoint(self) -> Optional[PoolEntry]:
        """从 checkpoint 池中采样一个条目。"""
        if not self.pool:
            return None

        cfg = self.config
        total = (
            cfg.checkpoint_latest_ratio
            + cfg.checkpoint_best_ratio
            + cfg.checkpoint_random_ratio
        )
        if total <= 0:
            return random.choice(self.pool)

        r = random.random() * total

        if r < cfg.checkpoint_latest_ratio and self.latest:
            return self.latest
        r -= cfg.checkpoint_latest_ratio

        if r < cfg.checkpoint_best_ratio and self.best:
            return self.best

        return random.choice(self.pool)

    def _evict(self):
        if len(self.pool) <= self.config.pool_size:
            return
        self.pool.sort(key=lambda e: e.elo, reverse=True)
        evicted = self.pool[self.config.pool_size:]
        self.pool = self.pool[:self.config.pool_size]
        for entry in evicted:
            p = Path(entry.path)
            if p.exists() and p.parent.resolve() == self.pool_dir.resolve():
                p.unlink()

    def _save_index(self):
        index_path = self.pool_dir / "pool_index.json"
        data = {
            "entries": [asdict(e) for e in self.pool],
            "kaggle_experts": list(self._kaggle_experts.keys()),
            "debug_experts": list(self._debug_experts.keys()),
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_index(self):
        index_path = self.pool_dir / "pool_index.json"
        if not index_path.exists():
            return
        with open(index_path, "r") as f:
            data = json.load(f)
        self.pool = [PoolEntry(**e) for e in data.get("entries", [])]
        # 不从文件恢复 expert_agents（需要运行时注册）

    def get_stats(self) -> dict:
        if not self.pool:
            return {
                "size": 0,
                "best_elo": 0,
                "avg_elo": 0,
                "kaggle_experts": list(self._kaggle_experts.keys()),
                "debug_experts": list(self._debug_experts.keys()),
            }
        elos = [e.elo for e in self.pool]
        return {
            "size": len(self.pool),
            "best_elo": max(elos),
            "avg_elo": float(np.mean(elos)),
            "latest_gen": max(e.generation for e in self.pool),
            "total_games": sum(e.total_games for e in self.pool),
            "kaggle_experts": list(self._kaggle_experts.keys()),
            "debug_experts": list(self._debug_experts.keys()),
        }


# ---------------------------------------------------------------------------
# 内置 agent 函数
# ---------------------------------------------------------------------------

def _random_agent(obs: dict) -> list[list]:
    """随机 agent — 不发任何动作。"""
    return []


@lru_cache(maxsize=32)
def _make_checkpoint_agent(checkpoint_path: str,
                           player_id: int = 0) -> Callable:
    """从 checkpoint 创建一个 agent 函数。"""
    import torch
    from training.core.config import ModelConfig
    from training.core.model import OrbitWarsModel
    from training.core.feature_engineering import FeatureEngineer

    device = "cpu"  # 对手用 CPU 不干扰训练 GPU
    config = ModelConfig()

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt

    n_planets = 40
    for key, val in state_dict.items():
        if "policy_head.target_mlp" in key and val.shape[0] <= 64:
            n_planets = int(val.shape[0])
            break

    model = OrbitWarsModel(config, n_planets=n_planets)
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    fe = FeatureEngineer()
    n_planets_ref = n_planets

    # 缓存：同一个 checkpoint 返回的 agent 可以重复使用
    def _agent(obs: dict) -> list[list]:
        import math
        import numpy as np

        raw_planets = obs.get("planets", [])
        n = len(raw_planets)
        if n == 0:
            return []

        pid = obs.get("player", player_id)
        planet_feat, fleet_feat, global_feat, metadata = fe.compute(obs, pid)

        planet_feat_t = torch.from_numpy(planet_feat).unsqueeze(0).to(device)
        fleet_feat_t = (
            torch.from_numpy(fleet_feat).unsqueeze(0).to(device)
            if fleet_feat.shape[0] > 0
            else torch.zeros(1, 0, 11, device=device)
        )
        global_feat_t = torch.from_numpy(global_feat).unsqueeze(0).to(device)

        owned_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
        enemy_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
        for i, p in enumerate(raw_planets):
            if int(p[1]) == pid:
                owned_mask[0, i] = True
            elif int(p[1]) != -1:
                enemy_mask[0, i] = True

        planet_ships = torch.tensor(
            [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=device
        )
        owners = {int(p[1]) for p in raw_planets if int(p[1]) >= 0}
        num_players_t = torch.tensor([max(len(owners), 2)], dtype=torch.long, device=device)

        with torch.no_grad():
            target_logits, num_ships, _, _, _ = model(
                planet_features=planet_feat_t,
                fleet_features=fleet_feat_t,
                global_features=global_feat_t,
                owned_mask=owned_mask,
                enemy_mask=enemy_mask,
                num_players=num_players_t,
                planet_ships=planet_ships,
            )

        tl = target_logits[0].cpu().numpy()
        ns = num_ships[0].cpu().numpy()

        all_planets_dicts = [
            {"id": int(p[0]), "x": float(p[2]), "y": float(p[3]),
             "ships": float(p[5]), "owner": int(p[1])}
            for p in raw_planets
        ]
        owned_p = [p for p in all_planets_dicts if p["owner"] == pid]

        # argmax 选择目标
        actions = []
        for i, src in enumerate(owned_p):
            if i >= len(tl):
                break
            tgt_idx = int(np.argmax(tl[i]))
            if tgt_idx >= len(all_planets_dicts):
                continue
            tgt = all_planets_dicts[tgt_idx]
            ratio = float(ns[i, 0]) if ns.ndim == 2 else float(ns[i])
            ships = ratio * src["ships"]
            if ships < 1 or src["ships"] <= 0:
                continue
            ships = max(1, min(int(ships), int(src["ships"])))
            angle = math.atan2(tgt["y"] - src["y"], tgt["x"] - src["x"])
            actions.append([src["id"], angle, ships])
        return actions

    return _agent
