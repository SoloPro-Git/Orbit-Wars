"""
Orbit Wars Simplified Environment for RL Training.

Self-contained game engine (no kaggle-environments dependency).
Simplified mechanics: no comets, distance-based collision detection.
Supports 2-player games with built-in opponent bots.
"""
import math
import random
import copy
from typing import List, Dict, Tuple, Optional

# Constants
BOARD_SIZE = 100.0
CENTER = 50.0
SUN_RADIUS = 10.0
ROTATION_RADIUS_LIMIT = 50.0
MAX_STEPS = 500
MAX_SPEED = 6.0
MIN_PLANET_GROUPS = 5
MAX_PLANET_GROUPS = 10
PLANET_CLEARANCE = 7


def dist(x1, y1, x2, y2):
    return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)


def fleet_speed(ships):
    """Speed = 1 + 5 * (log(ships)/log(1000))^1.5, capped at MAX_SPEED."""
    if ships <= 0:
        return 1.0
    s = 1.0 + (MAX_SPEED - 1.0) * (math.log(max(ships, 1)) / math.log(1000)) ** 1.5
    return min(s, MAX_SPEED)


# ─────────────────────────── Built-in Bots ───────────────────────────

class StarterBot:
    """Simple rule-based bot: attack nearest neutral/enemy, send half ships."""

    def act(self, obs):
        player = obs['player']
        planets = obs['planets']
        step = obs['step']

        my = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]
        if not my or not targets:
            return []

        moves = []
        for mp in my:
            if mp[5] <= 5:
                continue
            best = None
            best_dist = float('inf')
            for t in targets:
                d = dist(mp[2], mp[3], t[2], t[3])
                # Prefer neutral, then enemy
                priority = 0 if t[1] == -1 else 1
                score = d + priority * 20
                if score < best_dist:
                    best_dist = score
                    best = t
            if best:
                angle = math.atan2(best[3] - mp[3], best[2] - mp[2])
                ships = mp[5] // 2
                if ships >= 1:
                    moves.append([mp[0], angle, ships])
        return moves


class RandomBot:
    """Random actions: random target, random ships."""

    def __init__(self, rng=None):
        self.rng = rng or random.Random()

    def act(self, obs):
        player = obs['player']
        planets = obs['planets']
        my = [p for p in planets if p[1] == player]
        others = [p for p in planets if p[1] != player]
        if not my or not others:
            return []

        moves = []
        for mp in my:
            if self.rng.random() < 0.3 and mp[5] > 1:
                t = self.rng.choice(others)
                angle = math.atan2(t[3] - mp[3], t[2] - mp[2])
                ships = self.rng.randint(1, max(1, mp[5] - 1))
                moves.append([mp[0], angle, ships])
        return moves


class AggressiveBot:
    """Aggressive expansion: sends more ships, targets weakest enemies."""

    def act(self, obs):
        player = obs['player']
        planets = obs['planets']
        step = obs['step']

        my = [p for p in planets if p[1] == player]
        targets = [p for p in planets if p[1] != player]
        if not my or not targets:
            return []

        moves = []
        for mp in my:
            if mp[5] <= 3:
                continue
            # Score targets: prefer weak, close, high-production
            best = None
            best_score = -999
            for t in targets:
                d = dist(mp[2], mp[3], t[2], t[3])
                send = int(mp[5] * 0.7)
                if send < 1:
                    continue
                spd = fleet_speed(send)
                tt = math.ceil(d / spd)
                pred_garrison = t[5] + tt * t[6] if t[1] != -1 else t[5]
                surplus = send - pred_garrison
                score = surplus / (d + 1) + t[6] * 0.5
                if t[1] == -1:
                    score += 5  # prefer neutral
                if score > best_score:
                    best_score = score
                    best = t
            if best and best_score > 0:
                angle = math.atan2(best[3] - mp[3], best[2] - mp[2])
                ships = int(mp[5] * 0.7)
                if ships >= 1:
                    moves.append([mp[0], angle, ships])
        return moves


BOT_REGISTRY = {
    'starter': StarterBot,
    'random': RandomBot,
    'aggressive': AggressiveBot,
}


# ─────────────────────────── Environment ───────────────────────────

class OrbitWarsEnv:
    """
    Simplified Orbit Wars environment for RL.

    Observation (per player): dict with:
        - player: int
        - step: int
        - planets: list of [id, owner, x, y, radius, ships, production]
        - fleets: list of [id, owner, x, y, angle, from_planet_id, ships]

    Actions (per player): list of [planet_id, angle, num_ships]
    """

    def __init__(self, num_players=2):
        self.num_players = num_players
        self.planets: List[list] = []
        self.fleets: List[list] = []
        self.initial_planets: List[list] = []
        self.step_count = 0
        self.next_fleet_id = 0
        self.angular_velocity = 0.0
        self.done = False
        self.winner = -1  # -1 = undecided
        self._rng = random.Random()

    def reset(self, seed=None) -> List[dict]:
        """Reset environment, return initial observations."""
        if seed is not None:
            self._rng = random.Random(seed)
        self.planets = self._generate_map()
        self.initial_planets = [p[:] for p in self.planets]
        self.fleets = []
        self.step_count = 0
        self.next_fleet_id = 0
        self.done = False
        self.winner = -1

        # Set angular velocity (inner planets rotate)
        self.angular_velocity = self._rng.uniform(0.025, 0.05)
        if self._rng.random() < 0.5:
            self.angular_velocity = -self.angular_velocity

        # Assign starting planets: player 0 gets Q1, player 1 gets Q3
        # (simplified: give each player one planet near their corner)
        self._assign_starting_planets()

        return [self.get_obs(i) for i in range(self.num_players)]

    def step(self, actions_per_player: List[list]) -> Tuple[List[dict], List[float], List[bool], List[dict]]:
        """
        Execute one game step.
        actions_per_player: list of action lists, one per player.
        Returns: observations, rewards, dones, infos
        """
        if self.done:
            obs = [self.get_obs(i) for i in range(self.num_players)]
            return obs, [0.0] * self.num_players, [True] * self.num_players, [{}] * self.num_players

        # 1. Fleet launch
        for pid in range(self.num_players):
            self._process_moves(pid, actions_per_player[pid] if pid < len(actions_per_player) else [])

        # 2. Production
        for planet in self.planets:
            if planet[1] != -1:
                planet[5] += planet[6]

        # 3. Fleet movement + collision
        self._process_fleet_movement()

        # 4. Planet rotation
        self._rotate_planets()

        # 5. Step counter
        self.step_count += 1

        # 6. Check termination
        self._check_termination()

        # 7. Compute rewards
        rewards = self._compute_rewards()
        obs = [self.get_obs(i) for i in range(self.num_players)]
        dones = [self.done] * self.num_players
        infos = [{'winner': self.winner, 'step': self.step_count}] * self.num_players

        return obs, rewards, dones, infos

    def get_obs(self, player_id: int) -> dict:
        """Get observation for a specific player."""
        return {
            'player': player_id,
            'step': self.step_count,
            'planets': [p[:] for p in self.planets],
            'fleets': [f[:] for f in self.fleets],
        }

    # ──────────────── Internal Methods ────────────────

    def _generate_map(self) -> List[list]:
        """Generate symmetric map with planets."""
        planets = []
        num_groups = self._rng.randint(MIN_PLANET_GROUPS, MAX_PLANET_GROUPS)
        id_counter = 0
        attempts = 0

        while len(planets) < num_groups * 4 and attempts < 5000:
            attempts += 1
            prod = self._rng.randint(1, 5)
            r = 1 + math.log(prod)
            x = self._rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)
            y = self._rng.uniform(CENTER + 15, BOARD_SIZE - r - 5)

            orbital_radius = dist(x, y, CENTER, CENTER)
            if orbital_radius < SUN_RADIUS + r + 10:
                continue
            if orbital_radius + r >= ROTATION_RADIUS_LIMIT:
                if x + r > BOARD_SIZE or x - r < 0 or y + r > BOARD_SIZE or y - r < 0:
                    continue

            ships = self._rng.randint(5, 30)
            # 4-fold symmetry
            temp = [
                [id_counter, -1, y, x, r, ships, prod],
                [id_counter + 1, -1, BOARD_SIZE - x, y, r, ships, prod],
                [id_counter + 2, -1, x, BOARD_SIZE - y, r, ships, prod],
                [id_counter + 3, -1, BOARD_SIZE - y, BOARD_SIZE - x, r, ships, prod],
            ]

            valid = True
            for tp in temp:
                for p in planets:
                    if dist(p[2], p[3], tp[2], tp[3]) < p[4] + tp[4] + PLANET_CLEARANCE:
                        valid = False
                        break
                if not valid:
                    break

            if valid:
                planets.extend(temp)
                id_counter += 4

        return planets

    def _assign_starting_planets(self):
        """Give each player one starting planet."""
        if len(self.planets) < 4:
            return
        # 4-fold symmetry: planet 0=Q1, 1=Q2, 2=Q3, 3=Q4
        if self.num_players == 2:
            self.planets[0][1] = 0  # player 0 (Q1)
            self.planets[0][5] = 25
            self.planets[2][1] = 1  # player 1 (Q3)
            self.planets[2][5] = 25
        else:  # 4 players
            for i in range(4):
                self.planets[i][1] = i
                self.planets[i][5] = 25

    def _process_moves(self, player_id: int, actions: list):
        """Process fleet launches for a player."""
        if not actions:
            return
        for move in actions:
            if len(move) != 3:
                continue
            from_id, angle, ships = move
            ships = int(ships)
            planet = next((p for p in self.planets if p[0] == from_id), None)
            if planet and planet[1] == player_id and planet[5] >= ships and ships > 0:
                planet[5] -= ships
                sx = planet[2] + math.cos(angle) * (planet[4] + 0.1)
                sy = planet[3] + math.sin(angle) * (planet[4] + 0.1)
                self.fleets.append([
                    self.next_fleet_id, player_id, sx, sy, angle, from_id, ships
                ])
                self.next_fleet_id += 1

    def _process_fleet_movement(self):
        """Move fleets and resolve collisions."""
        fleets_to_remove = []
        combat_lists: Dict[int, list] = {p[0]: [] for p in self.planets}

        for fleet in self.fleets:
            angle = fleet[4]
            ships = fleet[6]
            spd = fleet_speed(ships)
            old_x, old_y = fleet[2], fleet[3]
            fleet[2] += math.cos(angle) * spd
            fleet[3] += math.sin(angle) * spd

            # Check collision with planets (simplified: endpoint distance)
            hit = False
            for planet in self.planets:
                d = dist(fleet[2], fleet[3], planet[2], planet[3])
                # Also check if the path crossed the planet (swept check simplified)
                d_path = self._point_to_segment_dist(
                    planet[2], planet[3], old_x, old_y, fleet[2], fleet[3])
                if d < planet[4] or d_path < planet[4]:
                    combat_lists[planet[0]].append(fleet)
                    fleets_to_remove.append(fleet)
                    hit = True
                    break

            if hit:
                continue

            # Out of bounds
            if not (0 <= fleet[2] <= BOARD_SIZE and 0 <= fleet[3] <= BOARD_SIZE):
                fleets_to_remove.append(fleet)
                continue

            # Sun collision
            d_sun = self._point_to_segment_dist(
                CENTER, CENTER, old_x, old_y, fleet[2], fleet[3])
            if d_sun < SUN_RADIUS:
                fleets_to_remove.append(fleet)

        # Resolve combat
        for pid, arriving in combat_lists.items():
            planet = next((p for p in self.planets if p[0] == pid), None)
            if not planet or not arriving:
                continue
            player_ships = {}
            for f in arriving:
                player_ships[f[1]] = player_ships.get(f[1], 0) + f[6]

            # Add defender
            if planet[1] != -1:
                player_ships[planet[1]] = player_ships.get(planet[1], 0) + planet[5]

            sorted_p = sorted(player_ships.items(), key=lambda x: x[1], reverse=True)
            top_owner, top_ships = sorted_p[0]
            if len(sorted_p) > 1:
                second_ships = sorted_p[1][1]
                survivor_ships = top_ships - second_ships
                if sorted_p[0][1] == sorted_p[1][1]:
                    survivor_ships = 0
            else:
                survivor_ships = top_ships

            if survivor_ships > 0:
                planet[1] = top_owner
                planet[5] = survivor_ships

        self.fleets = [f for f in self.fleets if f not in fleets_to_remove]

    def _rotate_planets(self):
        """Rotate inner planets around the sun."""
        step = self.step_count + 1
        initial_by_id = {p[0]: p for p in self.initial_planets}
        for planet in self.planets:
            initial_p = initial_by_id.get(planet[0])
            if initial_p is None:
                continue
            dx = initial_p[2] - CENTER
            dy = initial_p[3] - CENTER
            r = math.sqrt(dx ** 2 + dy ** 2)
            if r + planet[4] < ROTATION_RADIUS_LIMIT:
                init_angle = math.atan2(dy, dx)
                cur_angle = init_angle + self.angular_velocity * step
                planet[2] = CENTER + r * math.cos(cur_angle)
                planet[3] = CENTER + r * math.sin(cur_angle)

    def _check_termination(self):
        """Check if game is over."""
        if self.step_count >= MAX_STEPS:
            self.done = True
        alive = set()
        for p in self.planets:
            if p[1] != -1:
                alive.add(p[1])
        for f in self.fleets:
            alive.add(f[1])
        if len(alive) <= 1:
            self.done = True
            if len(alive) == 1:
                self.winner = alive.pop()

    def _compute_rewards(self) -> List[float]:
        """Compute per-player rewards."""
        rewards = [0.0] * self.num_players
        if self.done:
            # Terminal reward
            for i in range(self.num_players):
                if self.winner == i:
                    rewards[i] = 1.0
                elif self.winner != -1:
                    rewards[i] = -1.0
                else:
                    # Draw: compare total strength
                    pass
            # If draw, reward based on relative strength
            if self.winner == -1:
                strengths = []
                for i in range(self.num_players):
                    s = sum(p[5] for p in self.planets if p[1] == i)
                    s += sum(f[6] for f in self.fleets if f[1] == i)
                    strengths.append(s)
                max_s = max(strengths) if strengths else 1
                for i in range(self.num_players):
                    if strengths[i] == max_s and max_s > 0:
                        rewards[i] = 0.5
                    else:
                        rewards[i] = -0.5
        else:
            # Shaped reward: small bonus for owning planets + ships
            for i in range(self.num_players):
                my_planets = sum(1 for p in self.planets if p[1] == i)
                my_ships = sum(p[5] for p in self.planets if p[1] == i)
                total_planets = max(sum(1 for p in self.planets if p[1] != -1), 1)
                rewards[i] = (my_planets / total_planets - 0.5) * 0.01
        return rewards

    @staticmethod
    def _point_to_segment_dist(px, py, ax, ay, bx, by):
        """Distance from point (px,py) to segment (ax,ay)-(bx,by)."""
        l2 = (bx - ax) ** 2 + (by - ay) ** 2
        if l2 < 1e-12:
            return dist(px, py, ax, ay)
        t = max(0, min(1, ((px - ax) * (bx - ax) + (py - ay) * (by - ay)) / l2))
        proj_x = ax + t * (bx - ax)
        proj_y = ay + t * (by - ay)
        return dist(px, py, proj_x, proj_y)
