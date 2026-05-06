"""对手池管理 - 存储、采样、评估历史策略 checkpoint。"""
from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

import numpy as np


@dataclass
class PoolEntry:
    """对手池中的一个策略 checkpoint。"""
    path: str                        # checkpoint 文件路径
    elo: float = 600.0               # Elo 评分
    generation: int = 0              # 第几代策略
    timestamp: float = 0.0           # 保存时间戳
    win_count: int = 0               # 对战胜场
    loss_count: int = 0              # 对战负场
    draw_count: int = 0              # 对战平局

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
    pool_size: int = 20
    sample_latest_ratio: float = 0.4
    sample_random_ratio: float = 0.3
    sample_best_ratio: float = 0.2
    sample_heuristic_ratio: float = 0.1
    add_to_pool_win_rate: float = 0.45
    diversity_threshold: float = 0.3


class OpponentPool:
    """管理对手策略池，支持多种采样策略。"""

    def __init__(self, config: OpponentPoolConfig = None, pool_dir: str | Path = "checkpoints/pool"):
        self.config = config or OpponentPoolConfig()
        self.pool: list[PoolEntry] = []
        self.pool_dir = Path(pool_dir)
        self.pool_dir.mkdir(parents=True, exist_ok=True)
        self._heuristic_agents: list[str] = []

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

    def add(self, checkpoint_path: str | Path, elo: float = 600.0, generation: int = 0) -> PoolEntry:
        """添加新策略到对手池。"""
        entry = PoolEntry(
            path=str(checkpoint_path),
            elo=elo,
            generation=generation,
            timestamp=time.time(),
        )
        self.pool.append(entry)

        # 池满时移除最弱且最旧的
        if len(self.pool) > self.config.pool_size:
            self._evict()

        self._save_index()
        return entry

    def add_heuristic(self, agent_path: str):
        """注册启发式 agent 路径。"""
        self._heuristic_agents.append(agent_path)

    def sample(self, count: int = 1) -> list[str]:
        """根据采样策略采样对手 checkpoint 路径。"""
        if not self.pool and not self._heuristic_agents:
            return []

        sampled = []
        for _ in range(count):
            source = self._pick_source()
            if source == "heuristic" and self._heuristic_agents:
                sampled.append(random.choice(self._heuristic_agents))
            elif source == "latest" and self.pool:
                sampled.append(self.latest.path)
            elif source == "best" and self.pool:
                sampled.append(self.best.path)
            elif source == "random" and self.pool:
                sampled.append(random.choice(self.pool).path)
            elif self.pool:
                sampled.append(random.choice(self.pool).path)
            elif self._heuristic_agents:
                sampled.append(random.choice(self._heuristic_agents))

        return sampled

    def should_add(self, win_rate: float) -> bool:
        """判断新策略是否值得加入对手池。"""
        if not self.pool:
            return True
        if win_rate >= self.config.add_to_pool_win_rate:
            return True
        return False

    def update_elo(self, entry_path: str, result: float, k: float = 32.0):
        """更新 Elo 评分。

        Args:
            entry_path: 对手 checkpoint 路径
            result: 0=对手输, 0.5=平, 1=对手赢
            k: Elo K-factor
        """
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
        """按权重随机选择对手来源。"""
        cfg = self.config
        r = random.random()

        # 启发式 agent 比例
        if r < cfg.sample_heuristic_ratio and self._heuristic_agents:
            return "heuristic"
        r -= cfg.sample_heuristic_ratio

        # 最新策略比例
        if r < cfg.sample_latest_ratio:
            return "latest"
        r -= cfg.sample_latest_ratio

        # 最强策略比例
        if r < cfg.sample_best_ratio:
            return "best"

        # 随机历史版本
        return "random"

    def _evict(self):
        """淘汰最弱的策略（保留 top 80% Elo）。"""
        if len(self.pool) <= self.config.pool_size:
            return

        # 按 Elo 排序，保留 top pool_size
        self.pool.sort(key=lambda e: e.elo, reverse=True)
        evicted = self.pool[self.config.pool_size:]
        self.pool = self.pool[: self.config.pool_size]

        # 删除被淘汰的 checkpoint 文件
        for entry in evicted:
            p = Path(entry.path)
            if p.exists():
                p.unlink()

    def _save_index(self):
        """保存池索引到 JSON。"""
        index_path = self.pool_dir / "pool_index.json"
        data = {
            "entries": [asdict(e) for e in self.pool],
            "heuristic_agents": self._heuristic_agents,
        }
        with open(index_path, "w") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def load_index(self):
        """从 JSON 加载池索引。"""
        index_path = self.pool_dir / "pool_index.json"
        if not index_path.exists():
            return

        with open(index_path, "r") as f:
            data = json.load(f)

        self.pool = [PoolEntry(**e) for e in data.get("entries", [])]
        self._heuristic_agents = data.get("heuristic_agents", [])

    def get_stats(self) -> dict:
        """返回池统计信息。"""
        if not self.pool:
            return {"size": 0, "best_elo": 0, "avg_elo": 0}

        elos = [e.elo for e in self.pool]
        return {
            "size": len(self.pool),
            "best_elo": max(elos),
            "avg_elo": np.mean(elos),
            "latest_gen": max(e.generation for e in self.pool),
            "total_games": sum(e.total_games for e in self.pool),
        }
