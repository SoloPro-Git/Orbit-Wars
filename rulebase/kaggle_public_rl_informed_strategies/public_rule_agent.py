"""Stateful public rule agent distilled from the score-1049 public notebook."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .defense import max_enemy_fleet_to_target, planets_under_attack, reinforcement_plans
from .geometry import (
    angle_to,
    distance,
    find_angle_to_moving_planet,
    fleet_speed,
    sun_collision,
    travel_ticks,
)
from .scoring import closest_planets_to_target, public_custom_score
from .ship_requirements import calculate_required_ships, calculate_required_ships_moving
from .state import LocalObs, Planet, parse_observation
from .strategy_config import PUBLIC_EXACT

MIN_SHIPS_MINE_ATTACK = PUBLIC_EXACT.min_ships_mine_attack
MIN_SHIPS_TARGET_COOP_ATTACK = PUBLIC_EXACT.min_ships_target_coop_attack
COOP_PLANET_CAP = PUBLIC_EXACT.coop_planet_cap


@dataclass(slots=True)
class PublicRuleAgent:
    """A cleaner, module-friendly version of the public score-1049 rule agent.

    The strategy adds several ideas that baseline did not have:
    coordinated multi-source attacks, incoming-defense reinforcement, moving planet
    aiming, and trajectory reservations for attacks already in flight.
    """

    warmup_steps: int = PUBLIC_EXACT.warmup_steps
    min_ships_mine_attack: int = PUBLIC_EXACT.min_ships_mine_attack
    min_ships_target_coop_attack: int = PUBLIC_EXACT.min_ships_target_coop_attack
    coop_planet_cap: int = PUBLIC_EXACT.coop_planet_cap
    target_candidate_limit: int = PUBLIC_EXACT.target_candidate_limit
    skip_comet_targets: bool = PUBLIC_EXACT.skip_comet_targets
    enemy_owned_production_buffer_turns: int = PUBLIC_EXACT.enemy_owned_production_buffer_turns
    use_arrival_based_enemy_production: bool = PUBLIC_EXACT.use_arrival_based_enemy_production
    arrival_enemy_production_safety_turns: int = PUBLIC_EXACT.arrival_enemy_production_safety_turns
    arrival_enemy_production_max_turns: int = PUBLIC_EXACT.arrival_enemy_production_max_turns
    en_route_skip_owned_ratio: float = PUBLIC_EXACT.en_route_skip_owned_ratio
    enable_single_attacks: bool = PUBLIC_EXACT.enable_single_attacks
    enable_coop_attacks: bool = PUBLIC_EXACT.enable_coop_attacks
    enable_reinforcements: bool = PUBLIC_EXACT.enable_reinforcements
    enable_sun_avoidance: bool = PUBLIC_EXACT.enable_sun_avoidance
    enable_contested_target_adjustment: bool = PUBLIC_EXACT.enable_contested_target_adjustment
    contested_arrival_margin: int = PUBLIC_EXACT.contested_arrival_margin
    contested_enemy_weight: float = PUBLIC_EXACT.contested_enemy_weight
    contested_friendly_credit: float = PUBLIC_EXACT.contested_friendly_credit
    contested_skip_friendly_covered: bool = PUBLIC_EXACT.contested_skip_friendly_covered
    enable_dynamic_posture: bool = PUBLIC_EXACT.enable_dynamic_posture
    posture_aggressive_prod_deficit: float = PUBLIC_EXACT.posture_aggressive_prod_deficit
    posture_aggressive_ship_ratio: float = PUBLIC_EXACT.posture_aggressive_ship_ratio
    posture_aggressive_planet_deficit: int = PUBLIC_EXACT.posture_aggressive_planet_deficit
    posture_defensive_prod_lead: float = PUBLIC_EXACT.posture_defensive_prod_lead
    posture_defensive_ship_ratio: float = PUBLIC_EXACT.posture_defensive_ship_ratio
    posture_defensive_planet_lead: int = PUBLIC_EXACT.posture_defensive_planet_lead
    posture_late_step: int = PUBLIC_EXACT.posture_late_step
    aggressive_min_attack_delta: int = PUBLIC_EXACT.aggressive_min_attack_delta
    aggressive_target_candidate_bonus: int = PUBLIC_EXACT.aggressive_target_candidate_bonus
    defensive_min_attack_delta: int = PUBLIC_EXACT.defensive_min_attack_delta
    defensive_target_candidate_delta: int = PUBLIC_EXACT.defensive_target_candidate_delta
    defensive_reserve_turns: int = PUBLIC_EXACT.defensive_reserve_turns
    defensive_min_garrison: int = PUBLIC_EXACT.defensive_min_garrison
    defensive_high_prod_bonus: int = PUBLIC_EXACT.defensive_high_prod_bonus
    enable_value_defense: bool = PUBLIC_EXACT.enable_value_defense
    value_defense_min_production: float = PUBLIC_EXACT.value_defense_min_production
    value_defense_horizon: int = PUBLIC_EXACT.value_defense_horizon
    value_defense_buffer_turns: int = PUBLIC_EXACT.value_defense_buffer_turns
    value_defense_min_margin: int = PUBLIC_EXACT.value_defense_min_margin
    value_defense_max_send: int = PUBLIC_EXACT.value_defense_max_send
    value_defense_roi_multiplier: float = PUBLIC_EXACT.value_defense_roi_multiplier
    enable_proactive_value_defense: bool = PUBLIC_EXACT.enable_proactive_value_defense
    proactive_defense_min_production: float = PUBLIC_EXACT.proactive_defense_min_production
    proactive_defense_radius: float = PUBLIC_EXACT.proactive_defense_radius
    proactive_defense_base_margin: int = PUBLIC_EXACT.proactive_defense_base_margin
    proactive_defense_prod_turns: int = PUBLIC_EXACT.proactive_defense_prod_turns
    proactive_defense_enemy_prod_weight: float = PUBLIC_EXACT.proactive_defense_enemy_prod_weight
    proactive_defense_enemy_ship_weight: float = PUBLIC_EXACT.proactive_defense_enemy_ship_weight
    proactive_defense_enemy_launch_window: int = PUBLIC_EXACT.proactive_defense_enemy_launch_window
    proactive_defense_enemy_reserve_turns: int = PUBLIC_EXACT.proactive_defense_enemy_reserve_turns
    proactive_defense_enemy_send_fraction: float = PUBLIC_EXACT.proactive_defense_enemy_send_fraction
    proactive_defense_threat_slack: int = PUBLIC_EXACT.proactive_defense_threat_slack
    proactive_defense_max_send: int = PUBLIC_EXACT.proactive_defense_max_send
    proactive_defense_max_targets: int = PUBLIC_EXACT.proactive_defense_max_targets
    proactive_defense_max_arrival: int = PUBLIC_EXACT.proactive_defense_max_arrival
    proactive_defense_roi_multiplier: float = PUBLIC_EXACT.proactive_defense_roi_multiplier
    proactive_defense_after_attacks: bool = PUBLIC_EXACT.proactive_defense_after_attacks
    proactive_defense_require_turn_attack: bool = PUBLIC_EXACT.proactive_defense_require_turn_attack
    proactive_defense_require_recent_capture: bool = PUBLIC_EXACT.proactive_defense_require_recent_capture
    proactive_defense_recent_capture_window: int = PUBLIC_EXACT.proactive_defense_recent_capture_window
    proactive_defense_require_enemy_positive_roi: bool = PUBLIC_EXACT.proactive_defense_require_enemy_positive_roi
    proactive_defense_enemy_roi_multiplier: float = PUBLIC_EXACT.proactive_defense_enemy_roi_multiplier
    proactive_defense_enemy_min_net_value: float = PUBLIC_EXACT.proactive_defense_enemy_min_net_value
    proactive_defense_min_step: int = PUBLIC_EXACT.proactive_defense_min_step
    proactive_defense_max_step: int = PUBLIC_EXACT.proactive_defense_max_step
    proactive_defense_min_prod_diff: float = PUBLIC_EXACT.proactive_defense_min_prod_diff
    proactive_defense_min_planet_diff: int = PUBLIC_EXACT.proactive_defense_min_planet_diff
    proactive_defense_min_ship_ratio: float = PUBLIC_EXACT.proactive_defense_min_ship_ratio
    proactive_defense_source_min_after: int = PUBLIC_EXACT.proactive_defense_source_min_after
    proactive_defense_source_prod_turns_after: int = PUBLIC_EXACT.proactive_defense_source_prod_turns_after
    proactive_defense_source_front_distance: float = PUBLIC_EXACT.proactive_defense_source_front_distance
    proactive_defense_source_front_bonus: int = PUBLIC_EXACT.proactive_defense_source_front_bonus
    enable_local_source_reserve: bool = PUBLIC_EXACT.enable_local_source_reserve
    local_reserve_min_production: float = PUBLIC_EXACT.local_reserve_min_production
    local_reserve_enemy_distance: float = PUBLIC_EXACT.local_reserve_enemy_distance
    local_reserve_turns: int = PUBLIC_EXACT.local_reserve_turns
    local_reserve_min_garrison: int = PUBLIC_EXACT.local_reserve_min_garrison
    local_reserve_front_bonus: int = PUBLIC_EXACT.local_reserve_front_bonus
    enable_holdability_target_score: bool = PUBLIC_EXACT.enable_holdability_target_score
    holdability_radius: float = PUBLIC_EXACT.holdability_radius
    holdability_weight: float = PUBLIC_EXACT.holdability_weight
    holdability_enemy_prod_weight: float = PUBLIC_EXACT.holdability_enemy_prod_weight
    holdability_enemy_ship_weight: float = PUBLIC_EXACT.holdability_enemy_ship_weight
    holdability_own_prod_weight: float = PUBLIC_EXACT.holdability_own_prod_weight
    holdability_own_ship_weight: float = PUBLIC_EXACT.holdability_own_ship_weight
    enable_early_neutral_bias: bool = PUBLIC_EXACT.enable_early_neutral_bias
    early_neutral_step_limit: int = PUBLIC_EXACT.early_neutral_step_limit
    early_neutral_min_production: float = PUBLIC_EXACT.early_neutral_min_production
    early_neutral_max_ships: int = PUBLIC_EXACT.early_neutral_max_ships
    early_neutral_max_eta: int = PUBLIC_EXACT.early_neutral_max_eta
    early_neutral_bonus: float = PUBLIC_EXACT.early_neutral_bonus
    early_neutral_static_multiplier: float = PUBLIC_EXACT.early_neutral_static_multiplier
    early_neutral_safe_bonus: float = PUBLIC_EXACT.early_neutral_safe_bonus
    early_neutral_contested_penalty: float = PUBLIC_EXACT.early_neutral_contested_penalty
    early_neutral_reaction_margin: int = PUBLIC_EXACT.early_neutral_reaction_margin
    early_neutral_holdability_relief: float = PUBLIC_EXACT.early_neutral_holdability_relief
    enable_early_neutral_dynamic_max_ships: bool = PUBLIC_EXACT.enable_early_neutral_dynamic_max_ships
    early_neutral_dynamic_max_ships: int = PUBLIC_EXACT.early_neutral_dynamic_max_ships
    early_neutral_dynamic_min_production: float = PUBLIC_EXACT.early_neutral_dynamic_min_production
    early_neutral_dynamic_min_enemy_gap: int = PUBLIC_EXACT.early_neutral_dynamic_min_enemy_gap
    early_neutral_dynamic_source_min_after: int = PUBLIC_EXACT.early_neutral_dynamic_source_min_after
    enable_opening_rotating_neutral_filter: bool = PUBLIC_EXACT.enable_opening_rotating_neutral_filter
    opening_rotating_step_limit: int = PUBLIC_EXACT.opening_rotating_step_limit
    opening_rotating_max_eta: int = PUBLIC_EXACT.opening_rotating_max_eta
    opening_rotating_low_production: float = PUBLIC_EXACT.opening_rotating_low_production
    opening_rotating_penalty: float = PUBLIC_EXACT.opening_rotating_penalty
    enable_enemy_launch_punish: bool = PUBLIC_EXACT.enable_enemy_launch_punish
    enemy_launch_punish_max_fleet_age: int = PUBLIC_EXACT.enemy_launch_punish_max_fleet_age
    enemy_launch_punish_min_outgoing: int = PUBLIC_EXACT.enemy_launch_punish_min_outgoing
    enemy_launch_punish_min_production: float = PUBLIC_EXACT.enemy_launch_punish_min_production
    enemy_launch_punish_bonus_weight: float = PUBLIC_EXACT.enemy_launch_punish_bonus_weight
    enemy_launch_punish_max_targets: int = PUBLIC_EXACT.enemy_launch_punish_max_targets
    fleet_trajectories: list[dict[str, object]] = field(default_factory=list)
    reinforcement_trajectories: list[dict[str, object]] = field(default_factory=list)
    moving_planets: set[int] = field(default_factory=set)
    previous_owner_by_planet: dict[int, int] = field(default_factory=dict)
    recently_captured_steps: dict[int, int] = field(default_factory=dict)
    did_attack_this_turn: bool = False
    steps_seen: int = 0

    def __call__(self, obs, configuration=None) -> list[list[float | int]]:
        return self.act(obs)

    def act(self, obs) -> list[list[float | int]]:
        self.steps_seen += 1
        if self.steps_seen <= self.warmup_steps:
            return []

        local = parse_observation(obs)
        if self.steps_seen == self.warmup_steps + 1:
            self._fill_moving_planets(local)

        self._update_recent_captures(local)
        self._update_fleet_trajectories(local)
        self._update_reinforcement_trajectories()

        moves: list[list[float | int]] = []
        exhausted_planet_ids: set[int] = set()
        if not local.targets:
            return moves

        under_attack = planets_under_attack(
            local.mine,
            local.fleets,
            local.player,
            self.moving_planets,
            local.angular_velocity,
        )

        if self.enable_reinforcements:
            self._append_reinforcements(local, under_attack, exhausted_planet_ids, moves)
        attack_count_before = len(self.fleet_trajectories)
        self._append_attacks(local, under_attack, exhausted_planet_ids, moves)
        self.did_attack_this_turn = len(self.fleet_trajectories) > attack_count_before
        if self.enable_proactive_value_defense and self.proactive_defense_after_attacks:
            self._append_proactive_value_defense(local, under_attack, exhausted_planet_ids, moves)
        return moves

    def _fill_moving_planets(self, local: LocalObs) -> None:
        initial_by_id = {p.id: p for p in local.initial_planets}
        for planet in local.planets:
            initial = initial_by_id.get(planet.id)
            if initial is None:
                continue
            if (planet.x, planet.y) != (initial.x, initial.y):
                self.moving_planets.add(planet.id)

    def _update_fleet_trajectories(self, local: LocalObs) -> None:
        for tracked in self.fleet_trajectories[:]:
            found = any(
                fleet.from_planet_id == tracked["source_id"]
                and abs(fleet.angle - tracked["angle"]) < 1e-3
                for fleet in local.fleets
            )
            if found:
                tracked["arrive_tick"] = max(0, int(tracked["arrive_tick"]) - 1)
            else:
                self.fleet_trajectories.remove(tracked)

    def _update_reinforcement_trajectories(self) -> None:
        for tracked in self.reinforcement_trajectories[:]:
            tracked["arrive_tick"] = int(tracked["arrive_tick"]) - 1
            if tracked["arrive_tick"] <= 0:
                self.reinforcement_trajectories.remove(tracked)

    def _update_recent_captures(self, local: LocalObs) -> None:
        for planet in local.planets:
            previous_owner = self.previous_owner_by_planet.get(planet.id)
            if previous_owner is not None and previous_owner != local.player and planet.owner == local.player:
                self.recently_captured_steps[planet.id] = local.step
            self.previous_owner_by_planet[planet.id] = planet.owner

        keep_after = local.step - max(1, self.proactive_defense_recent_capture_window)
        for planet_id, step in list(self.recently_captured_steps.items()):
            if step < keep_after:
                del self.recently_captured_steps[planet_id]

    def _available_ships(
        self,
        planet: Planet,
        under_attack: dict[int, dict[str, object]],
        *,
        reserve_outgoing_reinforcements: bool = False,
    ) -> int:
        available = planet.ships
        if reserve_outgoing_reinforcements:
            available -= sum(
                int(row["total_ships"])
                for row in self.reinforcement_trajectories
                if row["source_id"] == planet.id
            )
        if planet.id in under_attack:
            available -= sum(int(row["fleet"].ships) for row in under_attack[planet.id]["fleets"])
        return max(0, available)

    def _posture(self, local: LocalObs) -> str:
        if not self.enable_dynamic_posture:
            return "balanced"

        own_prod = sum(p.production for p in local.planets if p.owner == local.player)
        enemy_prod = sum(p.production for p in local.planets if p.owner not in (-1, local.player))
        own_planets = sum(1 for p in local.planets if p.owner == local.player)
        enemy_planets = sum(1 for p in local.planets if p.owner not in (-1, local.player))
        own_ships = sum(p.ships for p in local.planets if p.owner == local.player)
        enemy_ships = sum(p.ships for p in local.planets if p.owner not in (-1, local.player))
        own_ships += sum(f.ships for f in local.fleets if f.owner == local.player)
        enemy_ships += sum(f.ships for f in local.fleets if f.owner not in (-1, local.player))

        prod_diff = own_prod - enemy_prod
        planet_diff = own_planets - enemy_planets
        ship_ratio = own_ships / max(enemy_ships, 1)

        if (
            prod_diff <= -self.posture_aggressive_prod_deficit
            or ship_ratio <= self.posture_aggressive_ship_ratio
            or planet_diff <= -self.posture_aggressive_planet_deficit
            or (local.step >= self.posture_late_step and prod_diff < 0)
        ):
            return "aggressive"
        if (
            prod_diff >= self.posture_defensive_prod_lead
            and ship_ratio >= self.posture_defensive_ship_ratio
            and planet_diff >= self.posture_defensive_planet_lead
        ):
            return "defensive"
        return "balanced"

    def _dynamic_min_attack(self, local: LocalObs) -> int:
        posture = self._posture(local)
        if posture == "aggressive":
            return max(1, self.min_ships_mine_attack + self.aggressive_min_attack_delta)
        if posture == "defensive":
            return max(1, self.min_ships_mine_attack + self.defensive_min_attack_delta)
        return self.min_ships_mine_attack

    def _dynamic_target_candidate_limit(self, local: LocalObs) -> int:
        posture = self._posture(local)
        if posture == "aggressive":
            return max(1, self.target_candidate_limit + self.aggressive_target_candidate_bonus)
        if posture == "defensive":
            return max(1, self.target_candidate_limit + self.defensive_target_candidate_delta)
        return self.target_candidate_limit

    def _available_attack_ships(
        self,
        planet: Planet,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
    ) -> int:
        available = self._available_ships(planet, under_attack)
        if self._posture(local) != "defensive":
            return available

        reserve = self.defensive_min_garrison + int(planet.production * self.defensive_reserve_turns)
        if planet.production >= 3:
            reserve += self.defensive_high_prod_bonus
        return max(0, available - reserve)

    def _local_source_reserve(self, planet: Planet, local: LocalObs) -> int:
        if not self.enable_local_source_reserve:
            return 0

        enemy_planets = [p for p in local.planets if p.owner not in (-1, local.player)]
        nearest_enemy = min((distance(planet, enemy) for enemy in enemy_planets), default=10**9)
        is_high_prod = planet.production >= self.local_reserve_min_production
        is_front = nearest_enemy <= self.local_reserve_enemy_distance
        if not is_high_prod and not is_front:
            return 0

        reserve = self.local_reserve_min_garrison + int(planet.production * self.local_reserve_turns)
        if is_front:
            front_pressure = max(0.0, self.local_reserve_enemy_distance - nearest_enemy) / max(self.local_reserve_enemy_distance, 1.0)
            reserve += int(self.local_reserve_front_bonus * front_pressure)
        return reserve

    def _available_local_attack_ships(
        self,
        planet: Planet,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
    ) -> int:
        return max(0, self._available_attack_ships(planet, local, under_attack) - self._local_source_reserve(planet, local))

    def _append_reinforcements(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        plans = reinforcement_plans(local.mine, under_attack, self.reinforcement_trajectories)
        planet_by_id = {p.id: p for p in local.mine}

        for target_id, plan in plans.items():
            target = planet_by_id.get(target_id)
            if target is None:
                continue
            already_reinforced = any(row["target"].id == target.id and row["arrive_tick"] >= 0 for row in self.reinforcement_trajectories)
            if already_reinforced:
                continue

            ships_needed = int(plan["ships_needed"])
            needed_by_tick = int(plan["needed_by_tick"])
            for source, _ in closest_planets_to_target(local.mine, target):
                if source.id == target.id or source.id in exhausted_planet_ids:
                    continue
                ships_to_send = max(self.min_ships_mine_attack, ships_needed)
                if self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True) < ships_to_send:
                    continue

                angle, arrive_tick = self._angle_and_arrival(source, target, ships_to_send, local)
                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue

                moves.append([source.id, angle, ships_to_send])
                exhausted_planet_ids.add(source.id)
                self.reinforcement_trajectories.append(
                    {
                        "source_id": source.id,
                        "target": target,
                        "angle": angle,
                        "total_ships": ships_to_send,
                        "arrive_tick": arrive_tick,
                    }
                )
                break
        if self.enable_value_defense:
            self._append_value_defense(local, under_attack, exhausted_planet_ids, moves)
        if self.enable_proactive_value_defense and not self.proactive_defense_after_attacks:
            self._append_proactive_value_defense(local, under_attack, exhausted_planet_ids, moves)

    def _append_value_defense(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        planet_by_id = {p.id: p for p in local.mine}
        for target_id, row in sorted(
            under_attack.items(),
            key=lambda item: planet_by_id.get(item[0], Planet(-1, -1, 0, 0, 0, 0, 0)).production,
            reverse=True,
        ):
            target = planet_by_id.get(target_id)
            if target is None or target.production < self.value_defense_min_production:
                continue
            if any(tracked["target"].id == target.id and tracked["arrive_tick"] >= 0 for tracked in self.reinforcement_trajectories):
                continue

            pressure = self._project_defense_pressure(target, row, local)
            if pressure is None:
                continue
            ships_needed, needed_by_tick = pressure
            ships_needed = min(ships_needed, self.value_defense_max_send)
            if ships_needed < self.min_ships_mine_attack:
                continue

            planet_value = target.production * max(0, 500 - local.step)
            if planet_value < ships_needed * self.value_defense_roi_multiplier:
                continue

            for source, _ in closest_planets_to_target(local.mine, target):
                if source.id == target.id or source.id in exhausted_planet_ids:
                    continue
                available = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
                if available < ships_needed:
                    continue
                angle, arrive_tick = self._angle_and_arrival(source, target, ships_needed, local)
                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue
                moves.append([source.id, angle, ships_needed])
                exhausted_planet_ids.add(source.id)
                self.reinforcement_trajectories.append(
                    {
                        "source_id": source.id,
                        "target": target,
                        "angle": angle,
                        "total_ships": ships_needed,
                        "arrive_tick": arrive_tick,
                    }
                )
                break

    def _append_proactive_value_defense(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if self.proactive_defense_require_turn_attack and not self.did_attack_this_turn:
            return
        if not self._proactive_context_allowed(local):
            return

        reinforced = 0
        targets = sorted(
            (planet for planet in local.mine if planet.production >= self.proactive_defense_min_production),
            key=lambda planet: (planet.production * max(0, 500 - local.step), planet.production),
            reverse=True,
        )
        for target in targets:
            if reinforced >= self.proactive_defense_max_targets:
                break
            if not self._proactive_target_allowed(target, local):
                continue
            if target.id in under_attack:
                continue
            if any(tracked["target"].id == target.id and tracked["arrive_tick"] >= 0 for tracked in self.reinforcement_trajectories):
                continue

            need = self._proactive_defense_need(target, local)
            if need is None:
                continue
            ships_needed, needed_by_tick = need
            ships_needed = min(ships_needed, self.proactive_defense_max_send)
            if ships_needed < self.min_ships_mine_attack:
                continue

            planet_value = target.production * max(0, 500 - local.step)
            if planet_value < ships_needed * self.proactive_defense_roi_multiplier:
                continue

            for source, _ in closest_planets_to_target(local.mine, target):
                if source.id == target.id or source.id in exhausted_planet_ids:
                    continue
                available = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
                if available < ships_needed:
                    continue
                if available - ships_needed < self._proactive_source_reserve(source, local):
                    continue
                angle, arrive_tick = self._angle_and_arrival(source, target, ships_needed, local)
                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue
                moves.append([source.id, angle, ships_needed])
                exhausted_planet_ids.add(source.id)
                reinforced += 1
                self.reinforcement_trajectories.append(
                    {
                        "source_id": source.id,
                        "target": target,
                        "angle": angle,
                        "total_ships": ships_needed,
                        "arrive_tick": arrive_tick,
                    }
                )
                break

    def _proactive_target_allowed(self, target: Planet, local: LocalObs) -> bool:
        if not self.proactive_defense_require_recent_capture:
            return True
        captured_step = self.recently_captured_steps.get(target.id)
        if captured_step is None:
            return False
        return local.step - captured_step <= self.proactive_defense_recent_capture_window

    def _proactive_source_reserve(self, source: Planet, local: LocalObs) -> int:
        reserve = self.proactive_defense_source_min_after + int(source.production * self.proactive_defense_source_prod_turns_after)
        if self.proactive_defense_source_front_bonus <= 0:
            return max(0, reserve)

        enemy_planets = [p for p in local.planets if p.owner not in (-1, local.player)]
        nearest_enemy = min((distance(source, enemy) for enemy in enemy_planets), default=10**9)
        if nearest_enemy <= self.proactive_defense_source_front_distance:
            pressure = max(0.0, self.proactive_defense_source_front_distance - nearest_enemy)
            pressure /= max(self.proactive_defense_source_front_distance, 1.0)
            reserve += int(self.proactive_defense_source_front_bonus * pressure)
        return max(0, reserve)

    def _proactive_context_allowed(self, local: LocalObs) -> bool:
        if local.step < self.proactive_defense_min_step or local.step > self.proactive_defense_max_step:
            return False

        own_prod = sum(p.production for p in local.planets if p.owner == local.player)
        enemy_prod = sum(p.production for p in local.planets if p.owner not in (-1, local.player))
        own_planets = sum(1 for p in local.planets if p.owner == local.player)
        enemy_planets = sum(1 for p in local.planets if p.owner not in (-1, local.player))
        own_ships = sum(p.ships for p in local.planets if p.owner == local.player)
        enemy_ships = sum(p.ships for p in local.planets if p.owner not in (-1, local.player))
        own_ships += sum(f.ships for f in local.fleets if f.owner == local.player)
        enemy_ships += sum(f.ships for f in local.fleets if f.owner not in (-1, local.player))

        if own_prod - enemy_prod < self.proactive_defense_min_prod_diff:
            return False
        if own_planets - enemy_planets < self.proactive_defense_min_planet_diff:
            return False
        if own_ships / max(enemy_ships, 1) < self.proactive_defense_min_ship_ratio:
            return False
        return True

    def _proactive_defense_need(self, target: Planet, local: LocalObs) -> tuple[int, int] | None:
        strongest_need = 0
        urgent_tick = self.proactive_defense_max_arrival
        for planet in local.planets:
            if planet.owner in (-1, local.player):
                continue
            dist = distance(planet, target)
            if dist > self.proactive_defense_radius:
                continue

            sendable = self._project_enemy_sendable(planet)
            if sendable < self.min_ships_mine_attack:
                continue
            arrival = travel_ticks(planet, target, sendable)
            if arrival > self.proactive_defense_max_arrival:
                continue

            projected_defense = self._project_owned_planet_ships(target, arrival)
            projected_defense += self._incoming_reinforcement_to_target(target, arrival)
            threat_gap = sendable - projected_defense
            if threat_gap < -self.proactive_defense_threat_slack:
                continue
            enemy_cost = projected_defense + 1
            if not self._enemy_recapture_roi_allowed(target, local, arrival, enemy_cost):
                continue

            desired_margin = self.proactive_defense_base_margin + int(target.production * self.proactive_defense_prod_turns)
            need = sendable + desired_margin - projected_defense
            if need > strongest_need:
                strongest_need = need
                urgent_tick = arrival

        if strongest_need <= 0:
            return None
        return max(self.min_ships_mine_attack, strongest_need), max(1, urgent_tick)

    def _project_enemy_sendable(self, planet: Planet) -> int:
        projected = int(planet.ships * self.proactive_defense_enemy_send_fraction)
        projected += int(planet.production * self.proactive_defense_enemy_launch_window)
        reserve = int(planet.production * self.proactive_defense_enemy_reserve_turns)
        return max(0, projected - reserve)

    def _enemy_recapture_roi_allowed(self, target: Planet, local: LocalObs, arrival: int, enemy_cost: int) -> bool:
        if not self.proactive_defense_require_enemy_positive_roi:
            return True
        remaining_value = target.production * max(0, 500 - local.step - arrival)
        if remaining_value < enemy_cost * self.proactive_defense_enemy_roi_multiplier:
            return False
        return remaining_value - enemy_cost >= self.proactive_defense_enemy_min_net_value

    def _project_owned_planet_ships(self, target: Planet, ticks: int) -> int:
        return int(target.ships + max(0, ticks) * target.production)

    def _incoming_reinforcement_to_target(self, target: Planet, by_tick: int) -> int:
        return sum(
            int(row["total_ships"])
            for row in self.reinforcement_trajectories
            if row["target"].id == target.id and 0 <= int(row["arrive_tick"]) <= by_tick
        )

    def _project_defense_pressure(
        self,
        target: Planet,
        attack_row: dict[str, object],
        local: LocalObs,
    ) -> tuple[int, int] | None:
        attacking = sorted(
            [row for row in attack_row["fleets"] if int(row["arrive_tick"]) <= self.value_defense_horizon],
            key=lambda row: row["arrive_tick"],
        )
        if not attacking:
            return None

        incoming = sorted(
            [row for row in self.reinforcement_trajectories if row["target"].id == target.id],
            key=lambda row: row["arrive_tick"],
        )
        desired_margin = self.value_defense_min_margin + int(target.production * self.value_defense_buffer_turns)
        available = target.ships
        previous_tick = 0
        reinf_idx = 0
        lowest_margin = 10**9
        low_tick = int(attacking[0]["arrive_tick"])
        for attack in attacking:
            arrive_tick = int(attack["arrive_tick"])
            available += int((arrive_tick - previous_tick) * target.production)
            while reinf_idx < len(incoming) and incoming[reinf_idx]["arrive_tick"] <= arrive_tick:
                available += int(incoming[reinf_idx]["total_ships"])
                reinf_idx += 1
            available -= int(attack["fleet"].ships)
            previous_tick = arrive_tick
            if available < lowest_margin:
                lowest_margin = available
                low_tick = arrive_tick

        if lowest_margin >= desired_margin:
            return None
        ships_needed = int(desired_margin - lowest_margin)
        return max(self.min_ships_mine_attack, ships_needed), low_tick

    def _append_attacks(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        launch_pressure = self._enemy_launch_pressure(local)
        for source in sorted(local.mine, key=lambda p: p.ships, reverse=True):
            if source.id in exhausted_planet_ids:
                continue
            if self._available_local_attack_ships(source, local, under_attack) < self._dynamic_min_attack(local):
                continue

            candidate_targets = [
                target
                for target in local.targets
                if not self.skip_comet_targets or target.id not in local.comet_planet_ids
            ]
            candidate_targets.sort(key=lambda target: self._target_score(source, target, local) + launch_pressure.get(target.id, 0.0), reverse=True)

            for target in candidate_targets[: self._dynamic_target_candidate_limit(local)]:
                if self.enable_single_attacks and self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break
                if self.enable_coop_attacks and self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break

    def _enemy_launch_pressure(self, local: LocalObs) -> dict[int, float]:
        if not self.enable_enemy_launch_punish:
            return {}

        planet_by_id = {planet.id: planet for planet in local.planets}
        outgoing: dict[int, int] = {}
        for fleet in local.fleets:
            if fleet.owner in (-1, local.player) or fleet.from_planet_id < 0:
                continue
            source = planet_by_id.get(fleet.from_planet_id)
            if source is None or source.owner in (-1, local.player):
                continue
            age = int(math.hypot(source.x - fleet.x, source.y - fleet.y) / fleet_speed(fleet.ships))
            if age > self.enemy_launch_punish_max_fleet_age:
                continue
            outgoing[source.id] = outgoing.get(source.id, 0) + int(fleet.ships)

        scored = []
        for planet_id, ships in outgoing.items():
            target = planet_by_id.get(planet_id)
            if target is None or target.production < self.enemy_launch_punish_min_production:
                continue
            if ships < self.enemy_launch_punish_min_outgoing:
                continue
            score = self.enemy_launch_punish_bonus_weight * ships + 10.0 * target.production - 0.5 * target.ships
            scored.append((score, planet_id))

        scored.sort(reverse=True)
        return {planet_id: score for score, planet_id in scored[: self.enemy_launch_punish_max_targets]}

    def _base_ships_needed(self, target: Planet, local: LocalObs, source: Planet | None = None) -> int | None:
        en_route = sum(
            int(row["total_ships"])
            for row in self.fleet_trajectories
            if row["target"].id == target.id
        )

        needed_now = target.ships + 1
        if target.owner != -1:
            needed_now += self._enemy_production_buffer(source, target, local, needed_now)

        enemy_competing = max_enemy_fleet_to_target(
            target,
            local.fleets,
            local.player,
            self.moving_planets,
            local.angular_velocity,
        )
        if enemy_competing > 0:
            needed_with_race = enemy_competing + target.ships + 1
            if target.owner != -1:
                needed_with_race += self._enemy_production_buffer(source, target, local, needed_with_race)
            needed_now = max(needed_now, needed_with_race)

        if len(local.mine) < len(local.planets) * self.en_route_skip_owned_ratio and en_route >= needed_now:
            return None
        return max(self._dynamic_min_attack(local), needed_now - en_route)

    def _enemy_production_buffer(
        self,
        source: Planet | None,
        target: Planet,
        local: LocalObs,
        ships: int,
    ) -> int:
        if not self.use_arrival_based_enemy_production or source is None:
            return int(self.enemy_owned_production_buffer_turns * target.production)
        arrival = self._estimate_arrival_for_requirement(source, target, ships, local)
        arrival = min(arrival + self.arrival_enemy_production_safety_turns, self.arrival_enemy_production_max_turns)
        return int(arrival * target.production)

    def _estimate_arrival_for_requirement(
        self,
        source: Planet,
        target: Planet,
        ships: int,
        local: LocalObs,
    ) -> int:
        if target.id in self.moving_planets:
            _, arrival = find_angle_to_moving_planet(source, target, max(1, ships), local.angular_velocity)
            if arrival is not None:
                return int(arrival)
        return travel_ticks(source, target, max(1, ships))

    def _try_single_attack(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> bool:
        base_ships = self._base_ships_needed(target, local, source=source)
        if base_ships is None:
            return False

        available = self._available_local_attack_ships(source, local, under_attack)
        if available < base_ships:
            return False

        total_ships = base_ships
        angle: float | None
        arrive_tick: int | None

        if target.id in self.moving_planets:
            angle, arrive_tick = self._moving_attack_plan(
                source,
                target,
                total_ships,
                available,
                local,
                include_target_production=not self.use_arrival_based_enemy_production,
            )
        else:
            angle, arrive_tick, total_ships = self._static_attack_plan(
                source,
                target,
                total_ships,
                available,
                include_target_production=not self.use_arrival_based_enemy_production,
            )

        if angle is None or arrive_tick is None:
            return False

        adjusted_ships = self._adjust_for_contested_target(target, local, total_ships, arrive_tick)
        if adjusted_ships is None:
            return False
        if adjusted_ships > total_ships:
            if adjusted_ships > available:
                return False
            total_ships = adjusted_ships
            if target.id in self.moving_planets:
                angle, arrive_tick = self._moving_attack_plan(
                    source,
                    target,
                    total_ships,
                    available,
                    local,
                    include_target_production=not self.use_arrival_based_enemy_production,
                )
            else:
                angle, arrive_tick, total_ships = self._static_attack_plan(
                    source,
                    target,
                    total_ships,
                    available,
                    include_target_production=not self.use_arrival_based_enemy_production,
                )
            if angle is None or arrive_tick is None:
                return False

        if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
            return False

        moves.append([source.id, angle, total_ships])
        exhausted_planet_ids.add(source.id)
        self._track_attack(source, target, angle, total_ships, arrive_tick)
        return True

    def _try_coop_attack(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> bool:
        base_ships = self._base_ships_needed(target, local, source=source)
        if base_ships is None or target.ships < self.min_ships_target_coop_attack or len(local.mine) <= 1:
            return False

        source_available = self._available_local_attack_ships(source, local, under_attack)
        if source_available >= base_ships:
            return False

        attacking_planets = [{"planet": source, "ships": source_available}]
        accum = source_available
        for planet, _ in closest_planets_to_target(local.mine, target):
            if planet.id == source.id or planet.id in exhausted_planet_ids:
                continue
            available = self._available_local_attack_ships(planet, local, under_attack)
            if available < self._dynamic_min_attack(local):
                continue

            attacking_planets.append({"planet": planet, "ships": available})
            accum += available
            if len(attacking_planets) > self.coop_planet_cap:
                break
            if accum < base_ships:
                continue

            required = base_ships
            if target.owner != -1:
                if target.id in self.moving_planets:
                    required = calculate_required_ships_moving(attacking_planets, target, base_ships, local.angular_velocity)
                else:
                    required = calculate_required_ships(attacking_planets, target, base_ships)
            if accum < required:
                continue

            planned = self._build_coop_plan(attacking_planets, target, required, local)
            if not planned:
                continue
            for move_source, angle, ships, arrive_tick in planned:
                moves.append([move_source.id, angle, ships])
                exhausted_planet_ids.add(move_source.id)
                self._track_attack(move_source, target, angle, ships, arrive_tick)
            return True

        return False

    def _target_score(self, source: Planet, target: Planet, local: LocalObs) -> float:
        score = public_custom_score(source, target)
        score += self._early_neutral_score_adjustment(source, target, local)
        if not self.enable_holdability_target_score:
            return score

        enemy_prod = 0.0
        enemy_ships = 0.0
        own_prod = 0.0
        own_ships = 0.0
        for planet in local.planets:
            if planet.id == target.id:
                continue
            dist = distance(planet, target)
            if dist > self.holdability_radius:
                continue
            proximity = 1.0 - dist / max(self.holdability_radius, 1.0)
            if planet.owner == local.player:
                own_prod += planet.production * proximity
                own_ships += planet.ships * proximity
            elif planet.owner != -1:
                enemy_prod += planet.production * proximity
                enemy_ships += planet.ships * proximity

        risk = (
            enemy_prod * self.holdability_enemy_prod_weight
            + enemy_ships * self.holdability_enemy_ship_weight
            - own_prod * self.holdability_own_prod_weight
            - own_ships * self.holdability_own_ship_weight
        )
        if risk <= 0:
            return score
        if self._early_neutral_allowed(source, target, local) and self.early_neutral_holdability_relief < 1.0:
            risk *= max(0.0, self.early_neutral_holdability_relief)
        return score - risk * self.holdability_weight

    def _early_neutral_allowed(self, source: Planet, target: Planet, local: LocalObs) -> bool:
        if target.owner != -1 or local.step > self.early_neutral_step_limit:
            return False
        if target.production < self.early_neutral_min_production:
            return False
        needed = target.ships + 1
        if target.ships > self.early_neutral_max_ships:
            if not self._early_neutral_dynamic_cap_allowed(source, target, needed, local):
                return False
        eta = self._estimate_arrival_for_requirement(source, target, target.ships + 1, local)
        return eta <= self.early_neutral_max_eta

    def _early_neutral_dynamic_cap_allowed(
        self,
        source: Planet,
        target: Planet,
        needed: int,
        local: LocalObs,
    ) -> bool:
        if not self.enable_early_neutral_dynamic_max_ships:
            return False
        if target.ships > self.early_neutral_dynamic_max_ships:
            return False
        if target.production < self.early_neutral_dynamic_min_production:
            return False
        if source.ships - needed < self.early_neutral_dynamic_source_min_after:
            return False

        eta = self._estimate_arrival_for_requirement(source, target, needed, local)
        if eta > self.early_neutral_max_eta:
            return False
        enemy_eta = self._nearest_enemy_eta(target, local)
        return enemy_eta - eta >= self.early_neutral_dynamic_min_enemy_gap

    def _early_neutral_score_adjustment(self, source: Planet, target: Planet, local: LocalObs) -> float:
        if target.owner != -1:
            return 0.0

        needed = max(1, target.ships + 1)
        eta = self._estimate_arrival_for_requirement(source, target, needed, local)
        adjustment = 0.0

        if self.enable_opening_rotating_neutral_filter and local.step <= self.opening_rotating_step_limit:
            if target.id in self.moving_planets and (eta > self.opening_rotating_max_eta or target.production <= self.opening_rotating_low_production):
                adjustment -= self.opening_rotating_penalty

        if not self.enable_early_neutral_bias or not self._early_neutral_allowed(source, target, local):
            return adjustment

        bonus = self.early_neutral_bonus * target.production
        if target.id not in self.moving_planets:
            bonus *= self.early_neutral_static_multiplier

        enemy_eta = self._nearest_enemy_eta(target, local)
        if enemy_eta < 10**8:
            gap = enemy_eta - eta
            if gap >= self.early_neutral_reaction_margin:
                bonus += self.early_neutral_safe_bonus
            elif abs(gap) <= self.early_neutral_reaction_margin:
                bonus -= self.early_neutral_contested_penalty

        return adjustment + bonus

    def _nearest_enemy_eta(self, target: Planet, local: LocalObs) -> int:
        best = 10**9
        for planet in local.planets:
            if planet.owner in (-1, local.player):
                continue
            ships = max(1, min(max(1, planet.ships), target.ships + 1))
            best = min(best, travel_ticks(planet, target, ships))
        return best

    def _adjust_for_contested_target(
        self,
        target: Planet,
        local: LocalObs,
        total_ships: int,
        arrive_tick: int,
    ) -> int | None:
        if not self.enable_contested_target_adjustment:
            return total_ships

        horizon = max(1, arrive_tick + self.contested_arrival_margin)
        friendly, enemy = self._incoming_to_target_by_horizon(target, local, horizon)
        if (
            self.contested_skip_friendly_covered
            and friendly >= target.ships + 1
            and enemy <= friendly
        ):
            return None

        extra = int(max(0.0, enemy * self.contested_enemy_weight - friendly * self.contested_friendly_credit))
        return total_ships + extra

    def _incoming_to_target_by_horizon(
        self,
        target: Planet,
        local: LocalObs,
        horizon: int,
    ) -> tuple[int, int]:
        friendly = 0
        enemy = 0
        for fleet in local.fleets:
            speed = fleet_speed(fleet.ships)
            if speed <= 0:
                continue
            dx = target.x - fleet.x
            dy = target.y - fleet.y
            vx = math.cos(fleet.angle)
            vy = math.sin(fleet.angle)
            proj = dx * vx + dy * vy
            if proj <= 0:
                continue
            eta = int(math.ceil(proj / speed))
            if eta > horizon:
                continue
            perp = abs(dx * vy - dy * vx)
            if perp > target.radius + 1.5:
                continue
            if fleet.owner == local.player:
                friendly += fleet.ships
            else:
                enemy += fleet.ships
        return friendly, enemy

    def _build_coop_plan(
        self,
        attacking_planets: list[dict[str, object]],
        target: Planet,
        required_ships: int,
        local: LocalObs,
    ) -> list[tuple[Planet, float, int, int]] | None:
        remainder = required_ships
        planned: list[tuple[Planet, float, int, int]] = []
        for attack in attacking_planets:
            source = attack["planet"]
            ships = min(int(attack["ships"]), remainder)
            if ships > 0:
                ships = min(int(attack["ships"]), max(ships, self.min_ships_mine_attack))
            if ships <= 0:
                continue

            angle, arrive_tick = self._angle_and_arrival(source, target, ships, local)
            if angle is None or arrive_tick is None:
                continue
            if self.enable_sun_avoidance and sun_collision(source, ships, angle):
                continue

            planned.append((source, angle, ships, arrive_tick))
            remainder -= ships
            if remainder <= 0:
                return planned
        return None

    def _static_attack_plan(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        available: int,
        *,
        include_target_production: bool = True,
    ) -> tuple[float | None, int | None, int]:
        angle = angle_to(source, target)
        arrive_tick = travel_ticks(source, target, total_ships)
        if target.owner == -1 or not include_target_production:
            return angle, arrive_tick, total_ships

        for _ in range(3):
            turns_to_arrive = travel_ticks(source, target, total_ships)
            new_total = int(total_ships + turns_to_arrive * target.production)
            if new_total > available:
                return None, None, total_ships
            arrive_tick = turns_to_arrive
            if new_total == total_ships:
                break
            total_ships = new_total
        return angle, arrive_tick, total_ships

    def _moving_attack_plan(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        available: int,
        local: LocalObs,
        *,
        include_target_production: bool = True,
    ) -> tuple[float | None, int | None]:
        for _ in range(3):
            angle, arrive_tick = find_angle_to_moving_planet(source, target, total_ships, local.angular_velocity)
            if angle is None or arrive_tick is None:
                return None, None
            new_total = int(total_ships + arrive_tick * target.production) if target.owner != -1 and include_target_production else total_ships
            if new_total > available:
                return None, None
            if new_total == total_ships:
                return angle, arrive_tick
            total_ships = new_total
        return find_angle_to_moving_planet(source, target, total_ships, local.angular_velocity)

    def _angle_and_arrival(
        self,
        source: Planet,
        target: Planet,
        ships: int,
        local: LocalObs,
    ) -> tuple[float | None, int | None]:
        if target.id in self.moving_planets:
            return find_angle_to_moving_planet(source, target, ships, local.angular_velocity)
        return angle_to(source, target), int(math.floor(distance(source, target) / fleet_speed(ships)))

    def _track_attack(self, source: Planet, target: Planet, angle: float, ships: int, arrive_tick: int) -> None:
        self.fleet_trajectories.append(
            {
                "source_id": source.id,
                "target": target,
                "angle": angle,
                "total_ships": ships,
                "arrive_tick": arrive_tick,
            }
        )
