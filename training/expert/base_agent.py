"""专家策略基类模块。

定义所有专家智能体的通用接口和辅助方法。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class ExpertAgent(ABC):
    """专家智能体基类。

    所有专家策略都继承此类，实现统一的 get_actions 接口。
    """

    def __init__(self, player_id: int = 0, name: str = "Expert"):
        """初始化专家智能体。

        Args:
            player_id: 玩家 ID (0-3)。
            name: 策略名称，用于标识和日志。
        """
        self.player_id = player_id
        self.name = name

    @abstractmethod
    def get_actions(
        self,
        observation: dict[str, Any],
    ) -> list[list]:
        """根据观测返回动作列表。

        Args:
            observation: Kaggle 格式的观测字典，包含：
                - player: 玩家 ID
                - planets: [[id, owner, x, y, radius, ships, production], ...]
                - fleets: [[id, owner, x, y, angle, from_planet_id, ships], ...]
                - angular_velocity: float
                - step: 当前回合数

        Returns:
            动作列表: [[from_planet_id, angle, num_ships], ...]
        """
        pass

    # ------------------------------------------------------------------
    # 辅助方法
    # ------------------------------------------------------------------

    def get_my_planets(
        self,
        planets: list[list],
    ) -> list[dict]:
        """获取我方拥有的星球列表。

        Args:
            planets: 星球数据列表。

        Returns:
            我方星球字典列表: [{id, x, y, radius, ships, production}, ...]
        """
        my_planets = []
        for p in planets:
            if int(p[1]) == self.player_id:
                my_planets.append({
                    "id": int(p[0]),
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "radius": float(p[4]),
                    "ships": float(p[5]),
                    "production": float(p[6]),
                })
        return my_planets

    def get_enemy_planets(
        self,
        planets: list[list],
    ) -> list[dict]:
        """获取敌方拥有的星球列表。

        Args:
            planets: 星球数据列表。

        Returns:
            敌方星球字典列表: [{id, x, y, radius, ships, production, owner}, ...]
        """
        enemy_planets = []
        for p in planets:
            owner = int(p[1])
            if owner != self.player_id and owner >= 0:  # 排除未占领星球
                enemy_planets.append({
                    "id": int(p[0]),
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "radius": float(p[4]),
                    "ships": float(p[5]),
                    "production": float(p[6]),
                    "owner": owner,
                })
        return enemy_planets

    def get_neutral_planets(
        self,
        planets: list[list],
    ) -> list[dict]:
        """获取未占领星球列表。

        Args:
            planets: 星球数据列表。

        Returns:
            未占领星球字典列表: [{id, x, y, radius, ships, production}, ...]
        """
        neutral_planets = []
        for p in planets:
            if int(p[1]) < 0:  # 未占领
                neutral_planets.append({
                    "id": int(p[0]),
                    "x": float(p[2]),
                    "y": float(p[3]),
                    "radius": float(p[4]),
                    "ships": float(p[5]),
                    "production": float(p[6]),
                })
        return neutral_planets

    def compute_distance(
        self,
        planet1: dict,
        planet2: dict,
    ) -> float:
        """计算两颗星球的欧氏距离。

        Args:
            planet1: 星球1字典。
            planet2: 星球2字典。

        Returns:
            距离值。
        """
        dx = planet1["x"] - planet2["x"]
        dy = planet1["y"] - planet2["y"]
        return math.hypot(dx, dy)

    def compute_angle(
        self,
        source: dict,
        target: dict,
    ) -> float:
        """计算从源星球到目标星球的发射角度。

        Args:
            source: 源星球字典。
            target: 目标星球字典。

        Returns:
            弧度角度 (-pi, pi]。
        """
        dx = target["x"] - source["x"]
        dy = target["y"] - source["y"]
        return math.atan2(dy, dx)

    def estimate_ships_on_arrival(
        self,
        target: dict,
        distance: float,
        current_ships: float,
        production_rate: float,
        travel_time: float,
    ) -> float:
        """估算舰队到达时目标星球的飞船数量。

        Args:
            target: 目标星球。
            distance: 移动距离。
            current_ships: 当前飞船数。
            production_rate: 生产速率。
            travel_time: 预计旅行时间（回合）。

        Returns:
            估算的到达时飞船数。
        """
        # 假设飞船以平均速度移动
        avg_speed = 3.0  # 根据舰队速度公式估算
        turns = travel_time if travel_time > 0 else (distance / avg_speed)

        # 目标星球生产飞船
        future_ships = current_ships + production_rate * turns
        return future_ships

    def can_capture(
        self,
        my_ships: float,
        target_ships: float,
        safety_margin: float = 1.2,
    ) -> bool:
        """判断是否能成功占领目标。

        Args:
            my_ships: 我方飞船数。
            target_ships: 目标飞船数。
            safety_margin: 安全边际，默认 1.2 (留 20% 余量)。

        Returns:
            是否能成功占领。
        """
        return my_ships >= target_ships * safety_margin

    def select_best_target(
        self,
        source: dict,
        targets: list[dict],
        score_fn: callable,
    ) -> dict | None:
        """根据评分函数选择最佳目标。

        Args:
            source: 源星球。
            targets: 候选目标列表。
            score_fn: 评分函数，接收 (source, target) 返回分数。

        Returns:
            最佳目标字典，若无目标则返回 None。
        """
        if not targets:
            return None

        best_target = None
        best_score = -float("inf")

        for target in targets:
            score = score_fn(source, target)
            if score > best_score:
                best_score = score
                best_target = target

        return best_target
