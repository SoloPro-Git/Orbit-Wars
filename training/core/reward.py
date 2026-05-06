"""Reward calculation module for Orbit Wars RL training.

Implements dense intermediate rewards and sparse terminal rewards based on
economic modeling of the game state. Designed to guide RL agents toward
strategically sound play in the Orbit Wars competitive environment.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Reward configuration
# ---------------------------------------------------------------------------

@dataclass
class RewardConfig:
    terminal_weight: float = 1.0
    intermediate_weight: float = 0.2
    capture_reward_weight: float = 0.3
    loss_penalty_weight: float = 0.2
    defense_reward_weight: float = 0.15
    transit_cost_weight: float = 0.1
    production_advantage_weight: float = 0.15
    comet_roi_weight: float = 0.1


# ---------------------------------------------------------------------------
# Planet / Fleet index constants (matching observation format)
# ---------------------------------------------------------------------------

# Planet: [id, owner, x, y, radius, ships, production]
_P_ID = 0
_P_OWNER = 1
_P_X = 2
_P_Y = 3
_P_RADIUS = 4
_P_SHIPS = 5
_P_PRODUCTION = 6

# Fleet: [id, owner, x, y, angle, from_planet_id, ships]
_F_ID = 0
_F_OWNER = 1
_F_X = 2
_F_Y = 3
_F_ANGLE = 4
_F_FROM = 5
_F_SHIPS = 6


# ---------------------------------------------------------------------------
# Economic value helpers
# ---------------------------------------------------------------------------

def planet_economic_value(planet: list | np.ndarray, turn: int,
                          max_turns: int = 500) -> float:
    """Compute the remaining economic value of a planet.

    value = production * remaining_turns

    This represents how many ships the planet will produce from *now* until
    the end of the game if held continuously.
    """
    production = float(planet[_P_PRODUCTION])
    remaining = max(max_turns - turn, 0)
    return production * remaining


def comet_roi(comet: list | np.ndarray, turn: int, fleets: list,
              max_turns: int = 500) -> float:
    """Estimate return-on-investment for capturing a comet.

    ROI = production * remaining_life - capture_cost

    *remaining_life* is a rough estimate: we assume the comet was spawned at
    the nearest standard spawn turn (50, 150, 250, 350, 450) and leaves the
    board after ~100 turns.  We approximate remaining_life as
    ``(next_spawn_turn + 100 - turn)`` clamped to positive.

    *capture_cost* is estimated as the number of ships currently on the comet.
    """
    production = float(comet[_P_PRODUCTION])
    ships_on_comet = float(comet[_P_SHIPS])

    # Estimate remaining lifetime
    spawn_turns = np.array([50, 150, 250, 350, 450])
    diffs = turn - spawn_turns
    valid = diffs >= 0
    if valid.any():
        last_spawn = int(spawn_turns[valid][np.argmax(diffs[valid])])
        comet_lifetime = 100  # approximate lifetime in turns
        remaining_life = max((last_spawn + comet_lifetime) - turn, 0)
    else:
        remaining_life = 0

    gross_value = production * remaining_life
    return gross_value - ships_on_comet


# ---------------------------------------------------------------------------
# Observation helpers
# ---------------------------------------------------------------------------

def _get_planets_array(obs: dict) -> np.ndarray:
    """Return planets as a 2-D numpy array, or empty array."""
    raw = obs.get("planets", [])
    if len(raw) == 0:
        return np.empty((0, 7))
    return np.asarray(raw, dtype=np.float64)


def _get_fleets_array(obs: dict) -> np.ndarray:
    """Return fleets as a 2-D numpy array, or empty array."""
    raw = obs.get("fleets", [])
    if len(raw) == 0:
        return np.empty((0, 7))
    return np.asarray(raw, dtype=np.float64)


def _get_comet_ids(obs: dict) -> set:
    """Return the set of planet ids that are comets."""
    return set(obs.get("comet_planet_ids", []))


def get_player_total_ships(obs: dict, player_id: int) -> int:
    """Total ships = ships on owned planets + ships in owned fleets."""
    planets = _get_planets_array(obs)
    fleets = _get_fleets_array(obs)

    ships_on_planets = 0
    if planets.shape[0] > 0:
        mask = planets[:, _P_OWNER] == player_id
        ships_on_planets = int(planets[mask, _P_SHIPS].sum())

    ships_in_fleets = 0
    if fleets.shape[0] > 0:
        mask = fleets[:, _F_OWNER] == player_id
        ships_in_fleets = int(fleets[mask, _F_SHIPS].sum())

    return ships_on_planets + ships_in_fleets


def get_player_rank(obs: dict, player_id: int) -> int:
    """Rank (1-based) among all players by total ships.  Ties broken arbitrarily."""
    # Determine all player ids from planets and fleets
    players = set()
    planets = _get_planets_array(obs)
    fleets = _get_fleets_array(obs)
    if planets.shape[0] > 0:
        owners = np.unique(planets[:, _P_OWNER].astype(int))
        players.update(owners.tolist())
    if fleets.shape[0] > 0:
        owners = np.unique(fleets[:, _F_OWNER].astype(int))
        players.update(owners.tolist())
    # Remove neutral (-1)
    players.discard(-1)
    if not players:
        return 1

    scores = [(pid, get_player_total_ships(obs, pid)) for pid in players]
    scores.sort(key=lambda x: x[1], reverse=True)
    for rank, (pid, _) in enumerate(scores, start=1):
        if pid == player_id:
            return rank
    return len(scores)


def get_player_production(obs: dict, player_id: int) -> int:
    """Total production rate from owned planets."""
    planets = _get_planets_array(obs)
    if planets.shape[0] == 0:
        return 0
    mask = planets[:, _P_OWNER] == player_id
    return int(planets[mask, _P_PRODUCTION].sum())


def _get_num_players(obs: dict) -> int:
    """Infer number of active players from the observation."""
    players = set()
    planets = _get_planets_array(obs)
    fleets = _get_fleets_array(obs)
    if planets.shape[0] > 0:
        players.update(planets[:, _P_OWNER].astype(int).tolist())
    if fleets.shape[0] > 0:
        players.update(fleets[:, _F_OWNER].astype(int).tolist())
    players.discard(-1)
    return max(len(players), 2)


# ---------------------------------------------------------------------------
# Reward Calculator
# ---------------------------------------------------------------------------

class RewardCalculator:
    """Stateless reward calculator for Orbit Wars RL agents.

    The calculator combines a *sparse terminal reward* (based on final ranking)
    with a *dense intermediate reward* that uses economic modelling of the game
    state to provide per-step learning signals.
    """

    def __init__(self, config: Optional[RewardConfig] = None,
                 max_turns: int = 500):
        self.config = config if config is not None else RewardConfig()
        self.max_turns = max_turns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, obs_before: dict, obs_after: dict,
                player_id: int, actions: list, done: bool) -> float:
        """Compute the per-step reward.

        Parameters
        ----------
        obs_before : dict
            Observation at the *start* of the step (before actions).
        obs_after : dict
            Observation at the *end* of the step (after actions resolved).
        player_id : int
            The player for whom the reward is computed.
        actions : list
            Actions taken this step ``[[from_planet_id, angle, ships], ...]``.
        done : bool
            Whether the episode ended after this step.

        Returns
        -------
        float
            Scalar reward value.
        """
        reward = 0.0

        # Terminal reward (only when episode ends)
        if done:
            reward += self.config.terminal_weight * self.compute_terminal(
                obs_after, player_id, _get_num_players(obs_after)
            )

        # Intermediate reward (every step)
        intermediate = self.compute_intermediate(
            obs_before, obs_after, player_id, actions
        )
        reward += self.config.intermediate_weight * intermediate

        return reward

    def compute_terminal(self, obs: dict, player_id: int,
                         num_players: int) -> float:
        """Sparse terminal reward based on final ranking.

        Ranking  | Reward
        ---------|-------
        1st      | +1.0
        2nd (4p) | +0.3
        3rd (4p) | -0.3
        Last     | -1.0
        """
        rank = get_player_rank(obs, player_id)

        # Predefined reward table for common player counts
        if num_players == 2:
            table = {1: 1.0, 2: -1.0}
        elif num_players == 3:
            table = {1: 1.0, 2: 0.0, 3: -1.0}
        elif num_players == 4:
            table = {1: 1.0, 2: 0.3, 3: -0.3, 4: -1.0}
        else:
            # General formula: linear interpolation
            if num_players <= 1:
                return 0.0
            table = {r: 1.0 - 2.0 * (r - 1) / (num_players - 1)
                     for r in range(1, num_players + 1)}

        rank = min(rank, num_players)  # safety clamp
        return table.get(rank, -1.0)

    def compute_intermediate(self, obs_before: dict, obs_after: dict,
                             player_id: int, actions: list) -> float:
        """Dense intermediate reward using economic modelling.

        Components
        ----------
        1. Capture reward   – positive value for newly captured planets
        2. Loss penalty     – negative value for failed attacks (ships lost)
        3. Defense reward   – positive value for successful defenses
        4. Transit cost     – small penalty for ships stuck in transit
        5. Production delta – reward for improving production advantage
        6. Comet ROI        – reward for capturing comets with positive ROI

        The total is normalised to approximately (-1, +1).
        """
        turn_before = obs_before.get("step", 0)
        turn_after = obs_after.get("step", turn_before + 1)
        # Use turn_after for value calculations
        turn = turn_after

        planets_before = _get_planets_array(obs_before)
        planets_after = _get_planets_array(obs_after)
        fleets_after = _get_fleets_array(obs_after)
        comet_ids = _get_comet_ids(obs_after)

        # --- 1. Capture reward ---
        capture_reward = self._compute_capture_reward(
            planets_before, planets_after, player_id, turn
        )

        # --- 2. Loss penalty ---
        loss_penalty = self._compute_loss_penalty(
            planets_before, planets_after, fleets_after, player_id, actions
        )

        # --- 3. Defense reward ---
        defense_reward = self._compute_defense_reward(
            obs_before, obs_after, player_id, turn
        )

        # --- 4. Transit cost ---
        transit_cost = self._compute_transit_cost(
            fleets_after, obs_after, player_id
        )

        # --- 5. Production advantage delta ---
        production_delta = self._compute_production_advantage(
            obs_before, obs_after, player_id, turn
        )

        # --- 6. Comet ROI ---
        comet_reward = self._compute_comet_roi(
            planets_before, planets_after, comet_ids, player_id, turn,
            obs_after.get("fleets", [])
        )

        # Weighted sum
        total = (
            self.config.capture_reward_weight * capture_reward
            + self.config.loss_penalty_weight * loss_penalty
            + self.config.defense_reward_weight * defense_reward
            + self.config.transit_cost_weight * transit_cost
            + self.config.production_advantage_weight * production_delta
            + self.config.comet_roi_weight * comet_reward
        )

        # Normalise to (-1, 1) using tanh with a scaling factor
        return float(np.tanh(total))

    # ------------------------------------------------------------------
    # Intermediate reward components (private)
    # ------------------------------------------------------------------

    def _compute_capture_reward(self, planets_before: np.ndarray,
                                planets_after: np.ndarray,
                                player_id: int, turn: int) -> float:
        """Reward for newly captured planets."""
        if planets_before.shape[0] == 0 or planets_after.shape[0] == 0:
            return 0.0

        # Build owner maps: planet_id -> owner
        owners_before = {
            int(row[_P_ID]): int(row[_P_OWNER]) for row in planets_before
        }

        reward = 0.0
        for row in planets_after:
            pid = int(row[_P_ID])
            owner_after = int(row[_P_OWNER])
            owner_before = owners_before.get(pid, -1)
            # Newly captured by us (was not ours before)
            if owner_after == player_id and owner_before != player_id:
                reward += planet_economic_value(row, turn, self.max_turns)

        return reward

    def _compute_loss_penalty(self, planets_before: np.ndarray,
                              planets_after: np.ndarray,
                              fleets_after: np.ndarray,
                              player_id: int, actions: list) -> float:
        """Penalty for ships lost in failed attacks.

        We estimate ships lost as:
        total_ships_before (planets + fleets from actions) -
        total_ships_after  (planets + fleets) where ships were sent.

        A simpler proxy: ships launched via actions that did not result in
        a capture or reinforce (i.e., they disappeared).  We estimate lost
        ships as the total ships in actions that cannot be accounted for.
        """
        if not actions:
            return 0.0

        # Total ships we launched this step
        launched = sum(a[2] for a in actions if len(a) >= 3)
        if launched <= 0:
            return 0.0

        # Our ships before (on planets)
        our_planet_ships_before = 0.0
        if planets_before.shape[0] > 0:
            mask = planets_before[:, _P_OWNER] == player_id
            our_planet_ships_before = float(planets_before[mask, _P_SHIPS].sum())

        # Our ships after (on planets + in fleets)
        our_planet_ships_after = 0.0
        if planets_after.shape[0] > 0:
            mask = planets_after[:, _P_OWNER] == player_id
            our_planet_ships_after = float(planets_after[mask, _P_SHIPS].sum())

        our_fleet_ships_after = 0.0
        if fleets_after.shape[0] > 0:
            mask = fleets_after[:, _F_OWNER] == player_id
            our_fleet_ships_after = float(fleets_after[mask, _F_SHIPS].sum())

        # Ships gained from production this step
        production_gain = 0.0
        if planets_before.shape[0] > 0:
            mask = planets_before[:, _P_OWNER] == player_id
            production_gain = float(planets_before[mask, _P_PRODUCTION].sum())

        # Net loss = ships we had - ships we have now + production
        expected = our_planet_ships_before + production_gain
        actual = our_planet_ships_after + our_fleet_ships_after
        lost = expected - actual
        lost = max(lost, 0.0)

        # Normalise by total ships to keep in range
        total = max(our_planet_ships_before, 1.0)
        return -lost / total

    def _compute_defense_reward(self, obs_before: dict, obs_after: dict,
                                player_id: int, turn: int) -> float:
        """Reward for successfully defending owned planets.

        If our planet was attacked (enemy fleets arrived) but we kept it,
        reward = planet_economic_value * (surviving_defenders / original).
        """
        planets_before = _get_planets_array(obs_before)
        planets_after = _get_planets_array(obs_after)
        fleets_before = _get_fleets_array(obs_before)

        if planets_before.shape[0] == 0 or planets_after.shape[0] == 0:
            return 0.0

        # Our planets before
        our_mask = planets_before[:, _P_OWNER] == player_id
        our_planets_before = planets_before[our_mask]

        if our_planets_before.shape[0] == 0:
            return 0.0

        # Build planet ship map for after-state
        ships_after = {}
        for row in planets_after:
            ships_after[int(row[_P_ID])] = (
                float(row[_P_SHIPS]) if int(row[_P_OWNER]) == player_id else 0.0
            )

        reward = 0.0
        for row in our_planets_before:
            pid = int(row[_P_ID])
            ships_before = float(row[_P_SHIPS])

            # Check if enemy fleets were heading to this planet
            if fleets_before.shape[0] > 0:
                enemy_mask = fleets_before[:, _F_OWNER] != player_id
                enemy_fleets = fleets_before[enemy_mask]
                # Fleets whose from_planet is this planet (they arrived)
                # More accurately, fleets that could have arrived at this planet
                # Since we don't have exact target, check if any enemy fleet
                # originated near this planet (proxy for incoming)
                near_planet = np.zeros(enemy_fleets.shape[0], dtype=bool)
                if enemy_fleets.shape[0] > 0:
                    # Simple heuristic: if enemy fleet was heading toward our planet
                    # we consider it an attack attempt
                    for i, f in enumerate(enemy_fleets):
                        fx, fy = f[_F_X], f[_F_Y]
                        px, py = row[_P_X], row[_P_Y]
                        dist = np.sqrt((fx - px) ** 2 + (fy - py) ** 2)
                        # If fleet was within reasonable attack distance
                        if dist < 15.0:
                            near_planet[i] = True

                if near_planet.any():
                    # We were attacked but still hold the planet
                    survived = ships_after.get(pid, 0.0)
                    if survived > 0 and ships_before > 0:
                        defense_ratio = min(survived / ships_before, 1.0)
                        value = planet_economic_value(row, turn, self.max_turns)
                        reward += value * defense_ratio

        return reward

    def _compute_transit_cost(self, fleets_after: np.ndarray,
                              obs_after: dict, player_id: int) -> float:
        """Penalty for ships stuck in transit (frozen assets).

        cost = -in_transit_ships * (1 - 0.99) / total_ships
             = -in_transit_ships * 0.01 / total_ships
        """
        total = get_player_total_ships(obs_after, player_id)
        if total <= 0:
            return 0.0

        in_transit = 0.0
        if fleets_after.shape[0] > 0:
            mask = fleets_after[:, _F_OWNER] == player_id
            in_transit = float(fleets_after[mask, _F_SHIPS].sum())

        return -in_transit * 0.01 / total

    def _compute_production_advantage(self, obs_before: dict,
                                      obs_after: dict,
                                      player_id: int, turn: int) -> float:
        """Reward for improving production advantage relative to opponents.

        delta = (our_production * remaining_turns) change
              - (avg_opponent_production * remaining_turns) change
        Normalised by max_turns * max_possible_production.
        """
        remaining = max(self.max_turns - turn, 1)

        our_prod_before = get_player_production(obs_before, player_id)
        our_prod_after = get_player_production(obs_after, player_id)

        # Opponent average production
        num_players_before = _get_num_players(obs_before)
        num_players_after = _get_num_players(obs_after)

        # Sum all production across all players
        planets_before = _get_planets_array(obs_before)
        planets_after = _get_planets_array(obs_after)

        total_prod_before = 0.0
        if planets_before.shape[0] > 0:
            owned_mask = planets_before[:, _P_OWNER] >= 0
            total_prod_before = float(planets_before[owned_mask, _P_PRODUCTION].sum())

        total_prod_after = 0.0
        if planets_after.shape[0] > 0:
            owned_mask = planets_after[:, _P_OWNER] >= 0
            total_prod_after = float(planets_after[owned_mask, _P_PRODUCTION].sum())

        n_opp_before = max(num_players_before - 1, 1)
        n_opp_after = max(num_players_after - 1, 1)

        opp_avg_before = max((total_prod_before - our_prod_before) / n_opp_before, 0)
        opp_avg_after = max((total_prod_after - our_prod_after) / n_opp_after, 0)

        # Advantage before and after
        advantage_before = (our_prod_before - opp_avg_before) * remaining
        advantage_after = (our_prod_after - opp_avg_after) * remaining

        delta = advantage_after - advantage_before

        # Normalise: max possible change is bounded by max production * remaining
        normaliser = max(5.0 * remaining, 1.0)  # 5 = max single planet production
        return float(delta / normaliser)

    def _compute_comet_roi(self, planets_before: np.ndarray,
                           planets_after: np.ndarray,
                           comet_ids: set, player_id: int,
                           turn: int, fleets_raw: list) -> float:
        """Reward for capturing comets with positive ROI."""
        if not comet_ids:
            return 0.0

        if planets_before.shape[0] == 0 or planets_after.shape[0] == 0:
            return 0.0

        owners_before = {
            int(row[_P_ID]): int(row[_P_OWNER]) for row in planets_before
        }

        reward = 0.0
        for row in planets_after:
            pid = int(row[_P_ID])
            if pid not in comet_ids:
                continue
            owner_after = int(row[_P_OWNER])
            owner_before = owners_before.get(pid, -1)
            if owner_after == player_id and owner_before != player_id:
                roi = comet_roi(row.tolist(), turn, fleets_raw, self.max_turns)
                if roi > 0:
                    reward += roi

        # Normalise by a typical comet value
        normaliser = 50.0  # rough normalisation
        return float(np.clip(reward / normaliser, -1.0, 1.0))
