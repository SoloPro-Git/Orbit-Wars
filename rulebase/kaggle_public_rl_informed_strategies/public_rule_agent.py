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
    en_route_skip_owned_ratio: float = PUBLIC_EXACT.en_route_skip_owned_ratio
    enable_single_attacks: bool = PUBLIC_EXACT.enable_single_attacks
    enable_coop_attacks: bool = PUBLIC_EXACT.enable_coop_attacks
    enable_reinforcements: bool = PUBLIC_EXACT.enable_reinforcements
    enable_sun_avoidance: bool = PUBLIC_EXACT.enable_sun_avoidance
    fleet_trajectories: list[dict[str, object]] = field(default_factory=list)
    reinforcement_trajectories: list[dict[str, object]] = field(default_factory=list)
    moving_planets: set[int] = field(default_factory=set)
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
        self._append_attacks(local, under_attack, exhausted_planet_ids, moves)
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

    def _append_attacks(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        for source in sorted(local.mine, key=lambda p: p.ships, reverse=True):
            if source.id in exhausted_planet_ids:
                continue
            if self._available_ships(source, under_attack) < self.min_ships_mine_attack:
                continue

            candidate_targets = [
                target
                for target in local.targets
                if not self.skip_comet_targets or target.id not in local.comet_planet_ids
            ]
            candidate_targets.sort(key=lambda target: public_custom_score(source, target), reverse=True)

            for target in candidate_targets[: self.target_candidate_limit]:
                if self.enable_single_attacks and self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break
                if self.enable_coop_attacks and self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break

    def _base_ships_needed(self, target: Planet, local: LocalObs) -> int | None:
        en_route = sum(
            int(row["total_ships"])
            for row in self.fleet_trajectories
            if row["target"].id == target.id
        )

        needed_now = target.ships + 1
        if target.owner != -1:
            needed_now += int(self.enemy_owned_production_buffer_turns * target.production)

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
                needed_with_race += int(self.enemy_owned_production_buffer_turns * target.production)
            needed_now = max(needed_now, needed_with_race)

        if len(local.mine) < len(local.planets) * self.en_route_skip_owned_ratio and en_route >= needed_now:
            return None
        return max(self.min_ships_mine_attack, needed_now - en_route)

    def _try_single_attack(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> bool:
        base_ships = self._base_ships_needed(target, local)
        if base_ships is None:
            return False

        available = self._available_ships(source, under_attack)
        if available < base_ships:
            return False

        total_ships = base_ships
        angle: float | None
        arrive_tick: int | None

        if target.id in self.moving_planets:
            angle, arrive_tick = self._moving_attack_plan(source, target, total_ships, available, local)
        else:
            angle, arrive_tick, total_ships = self._static_attack_plan(source, target, total_ships, available)

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
        base_ships = self._base_ships_needed(target, local)
        if base_ships is None or target.ships < self.min_ships_target_coop_attack or len(local.mine) <= 1:
            return False

        source_available = self._available_ships(source, under_attack)
        if source_available >= base_ships:
            return False

        attacking_planets = [{"planet": source, "ships": source_available}]
        accum = source_available
        for planet, _ in closest_planets_to_target(local.mine, target):
            if planet.id == source.id or planet.id in exhausted_planet_ids:
                continue
            available = self._available_ships(planet, under_attack)
            if available < self.min_ships_mine_attack:
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
    ) -> tuple[float | None, int | None, int]:
        angle = angle_to(source, target)
        arrive_tick = travel_ticks(source, target, total_ships)
        if target.owner == -1:
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
    ) -> tuple[float | None, int | None]:
        for _ in range(3):
            angle, arrive_tick = find_angle_to_moving_planet(source, target, total_ships, local.angular_velocity)
            if angle is None or arrive_tick is None:
                return None, None
            new_total = int(total_ships + arrive_tick * target.production) if target.owner != -1 else total_ships
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
