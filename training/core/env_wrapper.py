"""Orbit Wars RL 环境封装模块。

封装 kaggle_environments 的 orbit_wars 游戏，提供 Gym 风格的
reset / step 接口，适合强化学习训练使用。
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np
from kaggle_environments import make


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def compute_fleet_speed(ships: int, max_speed: float = 6.0) -> float:
    """根据舰队飞船数量计算移动速度。

    公式: speed = 1.0 + (maxSpeed - 1.0) * (log(ships) / log(1000)) ^ 1.5

    Args:
        ships: 舰队中的飞船数量。
        max_speed: 最大速度，默认 6.0。

    Returns:
        舰队移动速度 (单位/回合)。
    """
    if ships <= 0:
        return 0.0
    if ships == 1:
        return 1.0
    ratio = math.log(ships) / math.log(1000)
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def is_orbiting_planet(
    planet_x: float,
    planet_y: float,
    planet_radius: float,
    board_size: float = 100.0,
) -> bool:
    """判断星球是否为公转星球。

    判定条件: orbital_radius + planet_radius < 50
    其中 orbital_radius 是星球到棋盘中心 (board_size/2, board_size/2) 的距离。

    Args:
        planet_x: 星球 x 坐标。
        planet_y: 星球 y 坐标。
        planet_radius: 星球半径。
        board_size: 棋盘尺寸，默认 100。

    Returns:
        True 表示是公转星球，False 表示是静止星球。
    """
    center = board_size / 2.0
    orbital_radius = math.hypot(planet_x - center, planet_y - center)
    return (orbital_radius + planet_radius) < center


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class PlanetObs:
    """单颗星球的观测数据。"""
    id: int
    owner: int
    x: float
    y: float
    radius: float
    ships: float
    production: float


@dataclass
class FleetObs:
    """单个舰队的观测数据。"""
    id: int
    owner: int
    x: float
    y: float
    angle: float
    from_planet_id: int
    ships: float


@dataclass
class CometGroupObs:
    """彗星组观测数据。"""
    planet_ids: list[int]
    paths: list[list[list[float]]]
    path_index: int


@dataclass
class PlayerObservation:
    """某玩家的完整结构化观测。"""
    player: int
    angular_velocity: float
    turn: int
    planets: list[PlanetObs]
    initial_planets: list[PlanetObs]
    fleets: list[FleetObs]
    comets: list[CometGroupObs]
    comet_planet_ids: list[int]
    remaining_overage_time: float


# ---------------------------------------------------------------------------
# 环境封装
# ---------------------------------------------------------------------------

class OrbitWarsEnv:
    """Orbit Wars 的 Gym 风格 RL 环境封装。

    封装 kaggle_environments.make("orbit_wars")，提供逐步交互接口。
    内部使用 env.step() 接口推进游戏，而非 env.run()。

    用法::

        env = OrbitWarsEnv(num_players=4)
        obs = env.reset(seed=42)
        while True:
            actions = {0: [...], 1: [...], ...}  # player_id -> moves
            obs, rewards, dones, infos = env.step(actions)
            if all(dones.values()):
                break
    """

    # 默认游戏配置
    DEFAULT_CONFIG: dict[str, Any] = {
        "episodeSteps": 500,
        "actTimeout": 1.0,
        "shipSpeed": 6.0,
        "sunRadius": 10.0,
        "boardSize": 100.0,
        "cometSpeed": 4.0,
    }

    def __init__(
        self,
        num_players: int = 4,
        config: dict[str, Any] | None = None,
    ) -> None:
        """初始化环境。

        Args:
            num_players: 玩家数量 (2 或 4)。
            config: 游戏配置，覆盖默认值。可选键见 DEFAULT_CONFIG。
        """
        if num_players not in (2, 4):
            raise ValueError(f"num_players must be 2 or 4, got {num_players}")

        self.num_players = num_players
        self._raw_config = {**self.DEFAULT_CONFIG, **(config or {})}

        # kaggle 环境 (惰性创建 / 每次 reset 重建)
        self._env: Any = None
        self._done: bool = False
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # 公共接口
    # ------------------------------------------------------------------

    def reset(self, seed: int | None = None) -> dict[int, PlayerObservation]:
        """重置环境，返回每个玩家的初始观测。

        Args:
            seed: 随机种子，用于环境配置中的 seed 字段。

        Returns:
            dict[player_id -> PlayerObservation]
        """
        config = dict(self._raw_config)
        if seed is not None:
            config["seed"] = seed

        self._env = make("orbit_wars", configuration=config, debug=True)

        # 用 run 启动到第一步。env.run 会调用所有 agent 函数。
        # 我们用 "random" 作为占位 agent。
        agents = ["random"] * self.num_players
        self._env.run(agents)

        self._done = False
        self._step_count = 0

        # 构建初始观测
        observations: dict[int, PlayerObservation] = {}
        for pid in range(self.num_players):
            observations[pid] = self._build_observation(pid)

        return observations

    def _env_done(self) -> bool:
        """检查kaggle环境是否已经结束。

        Returns:
            True 如果环境已结束，False 否则。
        """
        if self._env is None:
            return False

        # 检查最后一步的状态
        if len(self._env.steps) == 0:
            return False

        last_step = self._env.steps[-1]
        # 如果所有玩家都不活跃，则环境已结束
        for pid in range(self.num_players):
            status = last_step[pid].get("status", "ACTIVE")
            if status == "ACTIVE":
                return False
        return True

    def step(
        self,
        actions: dict[int, list],
    ) -> tuple[
        dict[int, PlayerObservation],
        dict[int, float],
        dict[int, bool],
        dict[int, dict],
    ]:
        """执行一步。

        Args:
            actions: {player_id: [[from_planet_id, angle, num_ships], ...]}
                每个玩家 ID 映射到其移动列表。

        Returns:
            (observations, rewards, dones, infos)
            - observations: {player_id -> PlayerObservation}
            - rewards: {player_id -> float}
            - dones: {player_id -> bool}
            - infos: {player_id -> dict}
        """
        if self._env is None or self._done:
            raise RuntimeError("Environment must be reset before stepping.")

        # 将 actions 格式化为 env.step 需要的列表形式。
        # kaggle env.step 接收每个 agent 的 action (JSON 可序列化)。
        step_actions: list[list] = []
        for pid in range(self.num_players):
            action = actions.get(pid, [])
            # action 本身就是 [[from_planet_id, angle, num_ships], ...]
            step_actions.append(action)

        # 检查环境是否已经结束（避免在done后继续step）
        if self._env_done():
            # 环境已结束，返回空的观测和done=True
            return {}, {}, {pid: True for pid in range(self.num_players)}, {}

        # 调用 kaggle env.step
        self._env.step(step_actions)
        self._step_count += 1

        # 读取最新状态
        observations: dict[int, PlayerObservation] = {}
        rewards: dict[int, float] = {}
        dones: dict[int, bool] = {}
        infos: dict[int, dict] = {}

        last_step = self._env.steps[-1]
        for pid in range(self.num_players):
            state = last_step[pid]

            # reward
            rewards[pid] = float(state.get("reward", 0) or 0)

            # done 判定
            status = state.get("status", "ACTIVE")
            is_done = status != "ACTIVE"
            dones[pid] = is_done

            # info
            infos[pid] = {
                "status": status,
                "step": self._step_count,
            }

            # observation (只在仍活跃时构建)
            if not is_done:
                observations[pid] = self._build_observation(pid)
            else:
                observations[pid] = observations.get(pid, self._build_observation(pid))

        self._done = all(dones.values())

        return observations, rewards, dones, infos

    def get_observation(self, player_id: int) -> PlayerObservation:
        """获取指定玩家的当前结构化观测。

        Args:
            player_id: 玩家 ID (0-3)。

        Returns:
            PlayerObservation 结构化观测对象。
        """
        if self._env is None:
            raise RuntimeError("Environment has not been reset.")
        return self._build_observation(player_id)

    def get_raw_observation(self, player_id: int) -> dict[str, Any]:
        """获取指定玩家的原始观测字典 (kaggle 格式)。

        用于 feature_engineer 和 reward_calculator 等需要 raw dict 的模块。

        Args:
            player_id: 玩家 ID (0-3)。

        Returns:
            dict with keys: player, planets, fleets, angular_velocity, etc.
        """
        if self._env is None:
            raise RuntimeError("Environment has not been reset.")
        return self._get_raw_obs(player_id)

    @staticmethod
    def obs_to_dict(obs: PlayerObservation) -> dict[str, Any]:
        """将 PlayerObservation dataclass 转换为 raw dict 格式。

        用于将 env_wrapper 的输出转为 feature_engineer 可消费的格式。
        """
        return {
            "player": obs.player,
            "planets": [
                [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                for p in obs.planets
            ],
            "fleets": [
                [f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                for f in obs.fleets
            ],
            "angular_velocity": obs.angular_velocity,
            "step": obs.turn,
            "initial_planets": [
                [p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                for p in obs.initial_planets
            ],
            "comets": [
                {
                    "planet_ids": c.planet_ids,
                    "paths": c.paths,
                    "path_index": c.path_index,
                }
                for c in obs.comets
            ],
            "comet_planet_ids": obs.comet_planet_ids,
            "remainingOverageTime": obs.remaining_overage_time,
        }

    @property
    def done(self) -> bool:
        """游戏是否已结束。"""
        return self._done

    @property
    def step_count(self) -> int:
        """当前已执行步数。"""
        return self._step_count

    @property
    def raw_env(self) -> Any:
        """底层 kaggle_environments 环境实例。"""
        return self._env

    # ------------------------------------------------------------------
    # 内部方法
    # ------------------------------------------------------------------

    def _get_raw_obs(self, player_id: int) -> dict[str, Any]:
        """从 kaggle 环境中提取某玩家的原始观测字典。"""
        last_step = self._env.steps[-1]
        state = last_step[player_id]
        # observation 可能直接在 state 上，或嵌套在 state["observation"] 里
        obs = state.get("observation", state)
        if isinstance(obs, dict):
            return obs
        # 某些版本可能返回对象
        return vars(obs) if hasattr(obs, "__dict__") else {}

    def _build_observation(self, player_id: int) -> PlayerObservation:
        """从原始观测构建结构化 PlayerObservation。"""
        raw = self._get_raw_obs(player_id)

        # 星球
        planets: list[PlanetObs] = []
        for p in raw.get("planets", []):
            planets.append(PlanetObs(
                id=int(p[0]),
                owner=int(p[1]),
                x=float(p[2]),
                y=float(p[3]),
                radius=float(p[4]),
                ships=float(p[5]),
                production=float(p[6]),
            ))

        # 初始星球
        initial_planets: list[PlanetObs] = []
        for p in raw.get("initial_planets", []):
            initial_planets.append(PlanetObs(
                id=int(p[0]),
                owner=int(p[1]),
                x=float(p[2]),
                y=float(p[3]),
                radius=float(p[4]),
                ships=float(p[5]),
                production=float(p[6]),
            ))

        # 舰队
        fleets: list[FleetObs] = []
        for f in raw.get("fleets", []):
            fleets.append(FleetObs(
                id=int(f[0]),
                owner=int(f[1]),
                x=float(f[2]),
                y=float(f[3]),
                angle=float(f[4]),
                from_planet_id=int(f[5]),
                ships=float(f[6]),
            ))

        # 彗星组
        comets: list[CometGroupObs] = []
        for c in raw.get("comets", []):
            comets.append(CometGroupObs(
                planet_ids=[int(pid) for pid in c.get("planet_ids", [])],
                paths=c.get("paths", []),
                path_index=int(c.get("path_index", 0)),
            ))

        # 彗星 planet_ids
        comet_planet_ids = [int(pid) for pid in raw.get("comet_planet_ids", [])]

        # 回合数
        turn = int(raw.get("step", self._step_count))

        return PlayerObservation(
            player=int(raw.get("player", player_id)),
            angular_velocity=float(raw.get("angular_velocity", 0.0)),
            turn=turn,
            planets=planets,
            initial_planets=initial_planets,
            fleets=fleets,
            comets=comets,
            comet_planet_ids=comet_planet_ids,
            remaining_overage_time=float(raw.get("remainingOverageTime", 0.0)),
        )

    # ------------------------------------------------------------------
    # NumPy 便捷方法
    # ------------------------------------------------------------------

    def get_planets_array(self, player_id: int | None = None) -> np.ndarray:
        """获取星球的 NumPy 数组，方便批处理。

        Args:
            player_id: 如果指定，只返回该玩家拥有的星球。

        Returns:
            形状 (num_planets, 7) 的 float64 数组。
            列: [id, owner, x, y, radius, ships, production]
        """
        obs = self._build_observation(player_id or 0)
        planets = obs.planets
        if player_id is not None:
            planets = [p for p in planets if p.owner == player_id]
        if not planets:
            return np.empty((0, 7), dtype=np.float64)
        data = [[p.id, p.owner, p.x, p.y, p.radius, p.ships, p.production]
                for p in planets]
        return np.array(data, dtype=np.float64)

    def get_fleets_array(self, player_id: int | None = None) -> np.ndarray:
        """获取舰队的 NumPy 数组，方便批处理。

        Args:
            player_id: 如果指定，只返回该玩家拥有的舰队。

        Returns:
            形状 (num_fleets, 7) 的 float64 数组。
            列: [id, owner, x, y, angle, from_planet_id, ships]
        """
        obs = self._build_observation(player_id or 0)
        fleets = obs.fleets
        if player_id is not None:
            fleets = [f for f in fleets if f.owner == player_id]
        if not fleets:
            return np.empty((0, 7), dtype=np.float64)
        data = [[f.id, f.owner, f.x, f.y, f.angle, f.from_planet_id, f.ships]
                for f in fleets]
        return np.array(data, dtype=np.float64)

    def compute_score(self, player_id: int) -> float:
        """计算某玩家的当前总分 (星球飞船 + 舰队飞船)。

        Args:
            player_id: 玩家 ID。

        Returns:
            该玩家的飞船总数。
        """
        obs = self._build_observation(player_id)
        planet_ships = sum(p.ships for p in obs.planets if p.owner == player_id)
        fleet_ships = sum(f.ships for f in obs.fleets if f.owner == player_id)
        return planet_ships + fleet_ships

    def get_game_state_summary(self) -> dict[int, dict[str, Any]]:
        """获取所有玩家的游戏状态摘要。

        Returns:
            {player_id: {"score": float, "planets": int, "fleets": int,
                          "planet_ships": float, "fleet_ships": float}}
        """
        summary: dict[int, dict[str, Any]] = {}
        for pid in range(self.num_players):
            obs = self._build_observation(pid)
            my_planets = [p for p in obs.planets if p.owner == pid]
            my_fleets = [f for f in obs.fleets if f.owner == pid]
            planet_ships = sum(p.ships for p in my_planets)
            fleet_ships = sum(f.ships for f in my_fleets)
            summary[pid] = {
                "score": planet_ships + fleet_ships,
                "planets": len(my_planets),
                "fleets": len(my_fleets),
                "planet_ships": planet_ships,
                "fleet_ships": fleet_ships,
            }
        return summary
