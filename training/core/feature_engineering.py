"""
Orbit Wars 特征工程模块
将原始观察 (observation) 转换为神经网络可消费的张量特征。

用法:
    fe = FeatureEngineer()
    planet_feat, fleet_feat, global_feat, meta = fe.compute(obs, player_id=0)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

# 尝试导入 Numba 优化版本，失败则使用纯Python
try:
    from core.feature_engineering_numba import (
        fleet_speed_numba,
        segment_intersects_circle_numba,
        distance_numba,
        predict_fleet_targets_numba,
    )
    USE_NUMBA = True
except ImportError:
    USE_NUMBA = False


# =====================================================================
# 常量 & 维度
# =====================================================================
NUM_OWNERS = 5        # owner 0, 1, 2, 3 + neutral(-1)
D_PLANET = 28         # 星球特征维度 (增加了运动预测特征)
D_FLEET = 11          # 舰队特征维度
D_GLOBAL = 8          # 全局特征维度
SUN_CENTER = (50.0, 50.0)


class FeatureEngineer:
    """将 Orbit Wars 原始 obs 转换为归一化特征张量。"""

    def __init__(
        self,
        board_size: float = 100.0,
        sun_radius: float = 10.0,
        max_speed: float = 6.0,
        max_turns: int = 500,
    ):
        self.board_size = board_size
        self.sun_radius = sun_radius
        self.max_speed = max_speed
        self.max_turns = max_turns

    # -----------------------------------------------------------------
    # 主入口
    # -----------------------------------------------------------------
    def compute(
        self,
        obs: dict[str, Any],
        player_id: int,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        """
        参数:
            obs: 游戏观察字典 (来自 kaggle_environments)
            player_id: 当前玩家 ID (0-3)

        返回:
            planet_features: shape [N_planets, D_PLANET]
            fleet_features:  shape [N_fleets,  D_FLEET]
            global_features: shape [D_GLOBAL]
            metadata:        包含索引等辅助信息
        """
        # ---------- 解析原始数据 ----------
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
        raw_initial_planets = obs.get("initial_planets", [])
        angular_velocity = obs.get("angular_velocity", 0.0)
        turn = obs.get("step", 0)
        comet_planet_ids = set(obs.get("comet_planet_ids", []))
        comets_data = obs.get("comets", [])
        num_players = self._infer_num_players(raw_planets)

        # ---------- 处理初始行星信息（用于预测运动） ----------
        initial_planets = {}
        if raw_initial_planets:
            for ip in raw_initial_planets:
                ip_id, ip_owner, ip_x, ip_y, ip_radius, ip_ships, ip_production = ip
                initial_planets[int(ip_id)] = {
                    "x": float(ip_x),
                    "y": float(ip_y),
                    "radius": float(ip_radius),
                }

        # ---------- 固定参考原点 ----------
        # 使用“初始母星”或中心点，避免中途丢星后坐标系跳变导致分布漂移。
        home_x, home_y = self._get_stable_reference_origin(
            raw_planets, raw_initial_planets, player_id
        )

        # ---------- 构建星球列表 ----------
        planets: list[dict] = []
        for p in raw_planets:
            pid, owner, x, y, radius, ships, production = p
            planet_id = int(pid)

            # 计算是否是公转星球
            center_x, center_y = 50.0, 50.0
            dist_to_center = math.hypot(float(x) - center_x, float(y) - center_y)
            orbital_radius = dist_to_center
            is_orbiting = (orbital_radius + float(radius)) < 50.0

            planets.append({
                "id": planet_id,
                "owner": int(owner),
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "ships": float(ships),
                "production": float(production),
                "is_comet": planet_id in comet_planet_ids,
                "is_orbiting": is_orbiting,
                "orbital_radius": orbital_radius if is_orbiting else 0.0,
                "initial_pos": initial_planets.get(planet_id, {"x": float(x), "y": float(y)}),
            })

        # ---------- 构建舰队列表 ----------
        fleets: list[dict] = []
        for f in raw_fleets:
            fid, owner, x, y, angle, from_pid, ships = f
            fleets.append({
                "id": int(fid),
                "owner": int(owner),
                "x": float(x),
                "y": float(y),
                "angle": float(angle),
                "ships": float(ships),
            })

        # ---------- 威胁预测 (按 player_id 视角) ----------
        threat_info = self._compute_threat_features(
            fleets, planets, turn, player_id, angular_velocity, comets_data
        )

        # ---------- 彗星剩余生命 ----------
        comet_remaining = self._compute_comet_remaining(comets_data)

        # ---------- 星球特征 ----------
        planet_features, planet_meta = self._build_planet_features(
            planets, fleets, player_id, home_x, home_y,
            angular_velocity, turn, threat_info, comet_remaining,
        )

        # ---------- 舰队特征 ----------
        fleet_features = self._build_fleet_features(
            fleets, player_id, home_x, home_y,
        )

        # ---------- 全局特征 ----------
        global_features = self._build_global_features(
            planets, fleets, player_id, angular_velocity, turn, num_players,
        )

        # ---------- 元数据 ----------
        metadata = {
            "owned_planet_indices": planet_meta["owned"],
            "enemy_planet_indices": planet_meta["enemy"],
            "neutral_planet_indices": planet_meta["neutral"],
            "comet_indices": planet_meta["comet"],
            "home_offset": (home_x, home_y),
            "planet_ids": planet_meta["ids"],
        }

        return planet_features, fleet_features, global_features, metadata

    # =================================================================
    # 星球特征
    # =================================================================
    def _build_planet_features(
        self,
        planets: list[dict],
        fleets: list[dict],
        player_id: int,
        home_x: float,
        home_y: float,
        angular_velocity: float,
        turn: int,
        threat_info: dict[int, dict],
        comet_remaining: dict[int, float],
    ) -> tuple[np.ndarray, dict]:
        """
        每个星球一行 [D_PLANET]:
          0-4:   owner_one_hot [5]
          5:     x (相对母星归一化)
          6:     y (相对母星归一化)
          7:     radius / 10
          8:     ships / 100
          9:     production / 5
          10:    is_comet
          11:    is_orbiting
          12:    distance_to_sun / 50
          13:    incoming_enemy_ships / 100
          14:    incoming_enemy_eta / 50
          15:    incoming_friendly_ships / 100
          16:    incoming_friendly_eta / 50
          17:    threat_level
          18:    is_contested
          19:    planet_economic_value
          20:    comet_roi (仅彗星)
          21:    future_pos_5steps_x (公转行星5步后预测位置)
          22:    future_pos_5steps_y (公转行星5步后预测位置)
          23:    future_pos_10steps_x (公转行星10步后预测位置)
          24:    future_pos_10steps_y (公转行星10步后预测位置)
          25:    future_pos_20steps_x (公转行星20步后预测位置)
          26:    future_pos_20steps_y (公转行星20步后预测位置)
          27:    orbital_radius / 50 (轨道半径)
        """
        n = len(planets)
        features = np.zeros((n, D_PLANET), dtype=np.float32)
        meta = {"owned": [], "enemy": [], "neutral": [], "comet": [], "ids": []}

        remaining_turns = max(self.max_turns - turn, 0)

        for i, p in enumerate(planets):
            pid = p["id"]
            owner = p["owner"]

            # --- 索引分类 ---
            meta["ids"].append(pid)
            if owner == player_id:
                meta["owned"].append(i)
            elif owner == -1:
                meta["neutral"].append(i)
            else:
                meta["enemy"].append(i)
            if p["is_comet"]:
                meta["comet"].append(i)

            # --- owner one-hot ---
            owner_idx = owner if 0 <= owner <= 3 else 4  # -1 -> index 4
            features[i, owner_idx] = 1.0

            # --- 位置归一化 (相对于母星偏移) ---
            rel_x = (p["x"] - home_x) / self.board_size
            rel_y = (p["y"] - home_y) / self.board_size
            features[i, 5] = rel_x
            features[i, 6] = rel_y

            # --- 半径 ---
            features[i, 7] = p["radius"] / 10.0

            # --- 驻军 ---
            features[i, 8] = math.log1p(p["ships"]) / math.log(1000.0)

            # --- 产量 ---
            features[i, 9] = p["production"] / 5.0

            # --- 是否彗星 ---
            features[i, 10] = float(p["is_comet"])

            # --- 是否公转星球 ---
            dist_to_sun = math.hypot(p["x"] - SUN_CENTER[0], p["y"] - SUN_CENTER[1])
            is_orbiting = (dist_to_sun + p["radius"] < 50.0) and not p["is_comet"]
            features[i, 11] = float(is_orbiting)

            # --- 公转行星运动预测特征 ---
            if is_orbiting and "initial_pos" in p:
                # 计算当前角度（从初始位置计算）
                init_pos = p["initial_pos"]
                current_angle = math.atan2(p["y"] - SUN_CENTER[1], p["x"] - SUN_CENTER[0])

                # 预测未来5步、10步、20步后的位置（用于长期规划）
                # 21: future_pos_5steps_x
                # 22: future_pos_5steps_y
                # 23: future_pos_10steps_x
                # 24: future_pos_10steps_y
                # 需要增加 D_PLANET 到 28 才能容纳所有特征
                for future_steps, idx_x, idx_y in [(5, 21, 22), (10, 23, 24), (20, 25, 26)]:
                    future_angle = current_angle + angular_velocity * future_steps
                    future_x = SUN_CENTER[0] + p["orbital_radius"] * math.cos(future_angle)
                    future_y = SUN_CENTER[1] + p["orbital_radius"] * math.sin(future_angle)
                    # 归一化预测位置相对于当前母星
                    features[i, idx_x] = (future_x - home_x) / self.board_size
                    features[i, idx_y] = (future_y - home_y) / self.board_size
            else:
                features[i, 21] = 0.0  # 5步预测x
                features[i, 22] = 0.0  # 5步预测y
                features[i, 23] = 0.0  # 10步预测x
                features[i, 24] = 0.0  # 10步预测y
                features[i, 25] = 0.0  # 20步预测x
                features[i, 26] = 0.0  # 20步预测y

            # --- 轨道半径（帮助模型识别公转行为）---
            if is_orbiting:
                features[i, 27] = p["orbital_radius"] / 50.0
            else:
                features[i, 27] = 0.0

            # --- 到太阳距离 ---
            features[i, 12] = dist_to_sun / 50.0

            # --- 威胁特征 ---
            ti = threat_info.get(pid, {})
            features[i, 13] = math.log1p(ti.get("incoming_enemy_ships", 0.0)) / math.log(1000.0)
            features[i, 14] = ti.get("incoming_enemy_eta", 50.0) / 50.0
            features[i, 15] = math.log1p(ti.get("incoming_friendly_ships", 0.0)) / math.log(1000.0)
            features[i, 16] = ti.get("incoming_friendly_eta", 50.0) / 50.0

            # threat_level = enemy_strength / (garrison + eta * production + eps)
            garrison = p["ships"]
            production = p["production"]
            eta = ti.get("incoming_enemy_eta", 50.0)
            enemy_str = ti.get("incoming_enemy_ships", 0.0)
            denom = garrison + eta * production + 1e-8
            features[i, 17] = min(enemy_str / denom, 10.0) / 10.0  # 裁剪到 [0,1]

            # is_contested: 多方舰队朝此星球飞
            features[i, 18] = float(ti.get("is_contested", False))

            # 星球经济价值
            features[i, 19] = (p["production"] * remaining_turns) / 2500.0

            # 彗星 ROI
            if p["is_comet"]:
                rem_life = comet_remaining.get(pid, 0.0)
                capture_cost = p["ships"] + 1.0  # 最少需要比驻军多1
                roi = (p["production"] * rem_life - capture_cost) / 500.0
                features[i, 20] = roi
            else:
                features[i, 20] = 0.0

        return features, meta

    # =================================================================
    # 舰队特征
    # =================================================================
    def _build_fleet_features(
        self,
        fleets: list[dict],
        player_id: int,
        home_x: float,
        home_y: float,
    ) -> np.ndarray:
        """
        每个舰队一行 [D_FLEET]:
          0-4:   owner_one_hot [5]
          5:     x (相对母星归一化)
          6:     y (相对母星归一化)
          7:     angle / π
          8:     ships / 100
          9:     speed / max_speed
          10:    is_own - is_enemy (1 / -1 / 0)
        """
        n = len(fleets)
        if n == 0:
            return np.zeros((0, D_FLEET), dtype=np.float32)

        features = np.zeros((n, D_FLEET), dtype=np.float32)

        for i, f in enumerate(fleets):
            owner = f["owner"]

            # owner one-hot
            owner_idx = owner if 0 <= owner <= 3 else 4
            features[i, owner_idx] = 1.0

            # 位置归一化
            features[i, 5] = (f["x"] - home_x) / self.board_size
            features[i, 6] = (f["y"] - home_y) / self.board_size

            # 角度
            features[i, 7] = f["angle"] / math.pi

            # 舰船数
            features[i, 8] = math.log1p(f["ships"]) / math.log(1000.0)

            # 速度 (根据公式计算)
            speed = self._fleet_speed(f["ships"])
            features[i, 9] = speed / self.max_speed

            # 归属标记: is_own=1, is_enemy=-1, else=0
            if owner == player_id:
                features[i, 10] = 1.0
            elif owner != -1:
                features[i, 10] = -1.0
            else:
                features[i, 10] = 0.0

        return features

    # =================================================================
    # 全局特征
    # =================================================================
    def _build_global_features(
        self,
        planets: list[dict],
        fleets: list[dict],
        player_id: int,
        angular_velocity: float,
        turn: int,
        num_players: int,
    ) -> np.ndarray:
        """
        全局特征向量 [D_GLOBAL]:
          0: angular_velocity (通常 0.025-0.05)
          1: turn / max_turns
          2: own_total_ships / 500
          3: own_total_production / 25
          4: enemy_total_ships / 500
          5: own_planet_count / 40
          6: num_players / 4
          7: remaining_turns / 500
        """
        feat = np.zeros(D_GLOBAL, dtype=np.float32)

        # angular_velocity (直接保存，范围约 0.025-0.05)
        feat[0] = angular_velocity

        # 回合进度
        feat[1] = turn / self.max_turns

        # 统计己方和敌方
        own_ships = 0.0
        own_production = 0.0
        own_planet_count = 0
        enemy_ships = 0.0

        for p in planets:
            if p["owner"] == player_id:
                own_ships += p["ships"]
                own_production += p["production"]
                own_planet_count += 1
            elif p["owner"] != -1:
                enemy_ships += p["ships"]

        # 舰队中的飞船也计入
        for f in fleets:
            if f["owner"] == player_id:
                own_ships += f["ships"]
            elif f["owner"] != -1:
                enemy_ships += f["ships"]

        feat[2] = math.log1p(own_ships) / math.log(2000.0)
        feat[3] = own_production / 25.0
        feat[4] = math.log1p(enemy_ships) / math.log(3000.0)
        feat[5] = own_planet_count / 40.0
        feat[6] = num_players / 4.0
        feat[7] = max(self.max_turns - turn, 0) / 500.0

        return feat

    # =================================================================
    # 威胁预测
    # =================================================================
    def _predict_fleet_targets(
        self,
        fleets: list[dict],
        planets: list[dict],
    ) -> dict[int, list[dict]]:
        """
        对每个舰队外推轨迹，找到最可能的碰撞星球。

        返回: {planet_id: [{"fleet_id": int, "owner": int, "ships": float, "eta": int}, ...]}
        """
        max_steps = 50  # 最多外推 50 步
        predictions: dict[int, list[dict]] = {}

        for f in fleets:
            speed = self._fleet_speed(f["ships"])
            dx = math.cos(f["angle"]) * speed
            dy = math.sin(f["angle"]) * speed

            cur_x, cur_y = f["x"], f["y"]

            for step in range(1, max_steps + 1):
                new_x = cur_x + dx
                new_y = cur_y + dy

                # 检查是否越界
                if new_x < 0 or new_x > self.board_size or new_y < 0 or new_y > self.board_size:
                    break

                # 检查是否穿过太阳
                if self._segment_intersects_circle(
                    cur_x, cur_y, new_x, new_y,
                    SUN_CENTER[0], SUN_CENTER[1], self.sun_radius,
                ):
                    break

                # 检查是否碰撞星球
                for p in planets:
                    if self._segment_intersects_circle(
                        cur_x, cur_y, new_x, new_y,
                        p["x"], p["y"], p["radius"],
                    ):
                        pid = p["id"]
                        if pid not in predictions:
                            predictions[pid] = []
                        predictions[pid].append({
                            "fleet_id": f["id"],
                            "owner": f["owner"],
                            "ships": f["ships"],
                            "eta": step,
                        })
                        # 找到第一个碰撞即停止该舰队的外推
                        break
                else:
                    # 没有碰撞星球，继续外推
                    cur_x, cur_y = new_x, new_y
                    continue

                # 碰撞了星球或太阳，停止外推
                break

        return predictions

    def _compute_threat_features(
        self,
        fleets: list[dict],
        planets: list[dict],
        turn: int,
        player_id: int,
        angular_velocity: float = 0.0,
        comets_data: list[dict] | None = None,
    ) -> dict[int, dict]:
        """
        汇总威胁信息到每个星球，按 player_id 视角区分敌我。

        - friendly (incoming_friendly_*): owner == player_id 的舰队
        - enemy    (incoming_enemy_*):    owner != player_id 的舰队
        - is_contested: 多于1个不同 owner 的舰队朝此星球飞

        返回: {planet_id: {
            "incoming_enemy_ships": float,
            "incoming_enemy_eta": float,
            "incoming_friendly_ships": float,
            "incoming_friendly_eta": float,
            "is_contested": bool,
        }}
        """
        predictions = self._predict_fleet_targets_dynamic(
            fleets, planets, turn=turn, angular_velocity=angular_velocity, comets_data=comets_data or []
        )
        result: dict[int, dict] = {}

        for pid, arrivals in predictions.items():
            friendly_ships = 0.0
            friendly_eta = 50.0
            enemy_ships = 0.0
            enemy_eta = 50.0
            owners_set: set[int] = set()

            for a in arrivals:
                o = a["owner"]
                owners_set.add(o)
                if o == player_id:
                    # 己方增援
                    friendly_ships += a["ships"]
                    if a["eta"] < friendly_eta:
                        friendly_eta = a["eta"]
                else:
                    # 敌方或中立舰队
                    enemy_ships += a["ships"]
                    if a["eta"] < enemy_eta:
                        enemy_eta = a["eta"]

            # is_contested: 多于1个不同 owner 的舰队朝此星球飞
            is_contested = len(owners_set) > 1

            result[pid] = {
                "incoming_enemy_ships": enemy_ships,
                "incoming_enemy_eta": enemy_eta,
                "incoming_friendly_ships": friendly_ships,
                "incoming_friendly_eta": friendly_eta,
                "is_contested": is_contested,
            }

        return result

    def _predict_fleet_targets_dynamic(
        self,
        fleets: list[dict],
        planets: list[dict],
        turn: int,
        angular_velocity: float,
        comets_data: list[dict],
    ) -> dict[int, list[dict]]:
        """考虑行星公转/彗星路径的动态碰撞外推。"""
        max_steps = 50
        predictions: dict[int, list[dict]] = {}
        comet_pos_by_step: dict[tuple[int, int], tuple[float, float]] = {}
        for g in comets_data:
            planet_ids = [int(pid) for pid in g.get("planet_ids", [])]
            paths = g.get("paths", [])
            path_index = int(g.get("path_index", 0))
            for j, pid in enumerate(planet_ids):
                if j >= len(paths):
                    continue
                path = paths[j]
                if not isinstance(path, (list, np.ndarray)):
                    continue
                for s in range(1, max_steps + 1):
                    idx = path_index + s
                    if idx < len(path):
                        xy = path[idx]
                        if isinstance(xy, (list, np.ndarray)) and len(xy) >= 2:
                            comet_pos_by_step[(pid, s)] = (float(xy[0]), float(xy[1]))

        for f in fleets:
            speed = self._fleet_speed(f["ships"])
            dx = math.cos(f["angle"]) * speed
            dy = math.sin(f["angle"]) * speed
            cur_x, cur_y = f["x"], f["y"]

            for step in range(1, max_steps + 1):
                new_x = cur_x + dx
                new_y = cur_y + dy
                if new_x < 0 or new_x > self.board_size or new_y < 0 or new_y > self.board_size:
                    break
                if self._segment_intersects_circle(
                    cur_x, cur_y, new_x, new_y, SUN_CENTER[0], SUN_CENTER[1], self.sun_radius
                ):
                    break

                collided = False
                for p in planets:
                    px, py = p["x"], p["y"]
                    if p["is_comet"]:
                        pos = comet_pos_by_step.get((p["id"], step))
                        if pos is not None:
                            px, py = pos
                    elif p.get("is_orbiting", False):
                        # 按当前角度往前推 step 步
                        cur_ang = math.atan2(p["y"] - SUN_CENTER[1], p["x"] - SUN_CENTER[0])
                        fut_ang = cur_ang + angular_velocity * step
                        px = SUN_CENTER[0] + p["orbital_radius"] * math.cos(fut_ang)
                        py = SUN_CENTER[1] + p["orbital_radius"] * math.sin(fut_ang)

                    if self._segment_intersects_circle(cur_x, cur_y, new_x, new_y, px, py, p["radius"]):
                        predictions.setdefault(p["id"], []).append(
                            {"fleet_id": f["id"], "owner": f["owner"], "ships": f["ships"], "eta": step}
                        )
                        collided = True
                        break

                if collided:
                    break
                cur_x, cur_y = new_x, new_y

        return predictions

    # =================================================================
    # 辅助方法
    # =================================================================
    def _fleet_speed(self, ships: float) -> float:
        """根据飞船数计算舰队速度 (对数曲线)。"""
        if USE_NUMBA:
            return fleet_speed_numba(ships, self.max_speed)
        # 纯Python回退
        if ships <= 0:
            return 1.0
        log_ratio = math.log(max(ships, 1.0)) / math.log(1000.0)
        log_ratio = max(0.0, min(1.0, log_ratio))
        speed = 1.0 + (self.max_speed - 1.0) * (log_ratio ** 1.5)
        return speed

    @staticmethod
    def _segment_intersects_circle(
        ax: float, ay: float,
        bx: float, by: float,
        cx: float, cy: float,
        cr: float,
    ) -> bool:
        """
        检查线段 (ax,ay)-(bx,by) 是否与圆 (cx,cy,cr) 相交。
        使用连续碰撞检测算法。
        """
        if USE_NUMBA:
            return segment_intersects_circle_numba(ax, ay, bx, by, cx, cy, cr)
        # 线段方向向量
        dx = bx - ax
        dy = by - ay
        # 线段起点到圆心的向量
        fx = ax - cx
        fy = ay - cy

        # 二次方程系数: |P(t) - C|^2 = r^2
        # P(t) = A + t*(B-A), t in [0,1]
        a = dx * dx + dy * dy
        b = 2.0 * (fx * dx + fy * dy)
        c = fx * fx + fy * fy - cr * cr

        # 如果线段长度为0 (点)，直接检查距离
        if a < 1e-12:
            return c <= 0

        discriminant = b * b - 4.0 * a * c
        if discriminant < 0:
            return False

        sqrt_disc = math.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2.0 * a)
        t2 = (-b + sqrt_disc) / (2.0 * a)

        # 检查是否有交点在 [0, 1] 范围内
        # t1 <= t2 因为 sqrt_disc >= 0
        if t1 > 1.0 or t2 < 0.0:
            return False

        return True

    @staticmethod
    def _get_stable_reference_origin(
        raw_planets: list, raw_initial_planets: list, player_id: int,
    ) -> tuple[float, float]:
        """
        使用稳定坐标原点，避免中途丢星导致特征突变：
        1) 初始观测中 player_id 的起始母星
        2) 回退为地图中心
        """
        for p in raw_initial_planets:
            if int(p[1]) == player_id:
                return float(p[2]), float(p[3])
        for p in raw_planets:
            if int(p[1]) == player_id:
                return float(p[2]), float(p[3])
        return SUN_CENTER[0], SUN_CENTER[1]

    @staticmethod
    def _infer_num_players(raw_planets: list) -> int:
        """从星球 owner 推断玩家数。"""
        owners = set()
        for p in raw_planets:
            o = int(p[1])
            if 0 <= o <= 3:
                owners.add(o)
        return max(len(owners), 2)

    @staticmethod
    def _compute_comet_remaining(
        comets_data: list[dict],
    ) -> dict[int, float]:
        """
        从 obs["comets"] 计算每颗彗星的剩余生命 (回合数)。

        comets 结构: [{"planet_ids": [...], "paths": [...], "path_index": int}, ...]
        paths[path_index] 是当前位置，剩余生命 = len(paths) - path_index
        """
        remaining: dict[int, float] = {}
        for group in comets_data:
            planet_ids = group.get("planet_ids", [])
            paths_list = group.get("paths", [])
            path_index = group.get("path_index", 0)

            if not paths_list:
                continue

            # paths 可以是嵌套列表: [[x,y], [x,y], ...]
            # 每个 path 是该彗星的完整轨迹
            for idx, pid in enumerate(planet_ids):
                # 取对应的 path
                if idx < len(paths_list):
                    path = paths_list[idx]
                    if isinstance(path, (list, np.ndarray)):
                        total_steps = len(path)
                    else:
                        total_steps = 0
                else:
                    total_steps = 0

                rem = max(total_steps - path_index, 0)
                remaining[int(pid)] = float(rem)

        return remaining
