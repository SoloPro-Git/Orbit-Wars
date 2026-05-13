"""Public rule agent with extra signals borrowed from the local RL pipeline."""

from __future__ import annotations

from dataclasses import dataclass

from .geometry import angle_to, distance, find_angle_to_moving_planet, fleet_speed, sun_collision
from .public_rule_agent import PublicRuleAgent
from .rl_signals import comet_remaining_from_obs, remaining_steps, strategic_target_score
from .scoring import closest_planets_to_target, public_custom_score
from .state import LocalObs, Planet, parse_observation
from .strategy_config import PUBLIC_EXACT


@dataclass(slots=True)
class RLInformedPublicRuleAgent(PublicRuleAgent):
    """A copy of the public strategy with RL-pipeline-inspired rule signals.

    Added on top of PublicRuleAgent:
    - target scoring includes remaining-turn economic value
    - comet targets use real remaining life when available
    - late-game attacks are skipped if arrival is too late to matter
    - rear planets can reinforce the closest front planet after attacks
    """

    use_custom_attack_loop: bool = PUBLIC_EXACT.use_custom_attack_loop
    use_rl_target_score: bool = PUBLIC_EXACT.use_rl_target_score
    rl_score_weight: float = PUBLIC_EXACT.rl_score_weight
    use_late_filter: bool = PUBLIC_EXACT.use_late_filter
    use_comet_roi_filter: bool = PUBLIC_EXACT.use_comet_roi_filter
    include_comet_targets: bool = PUBLIC_EXACT.include_comet_targets
    custom_candidate_limit: int = PUBLIC_EXACT.custom_candidate_limit
    enable_front_support: bool = PUBLIC_EXACT.enable_front_support
    support_distance_factor: float = PUBLIC_EXACT.support_distance_factor
    support_min_available: int = PUBLIC_EXACT.support_min_available
    support_fraction: float = PUBLIC_EXACT.support_fraction
    support_min_send: int = PUBLIC_EXACT.support_min_send
    support_max_arrival: int = PUBLIC_EXACT.support_max_arrival
    _raw_obs: object | None = None

    def act(self, obs) -> list[list[float | int]]:
        self._raw_obs = obs
        return PublicRuleAgent.act(self, obs)

    def _append_attacks(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if not self.use_custom_attack_loop:
            PublicRuleAgent._append_attacks(self, local, under_attack, exhausted_planet_ids, moves)
            if self.enable_front_support:
                self._append_front_support(local, under_attack, exhausted_planet_ids, moves)
            return

        for source in sorted(local.mine, key=lambda p: p.ships, reverse=True):
            if source.id in exhausted_planet_ids:
                continue
            if self._available_ships(source, under_attack) < self.min_ships_mine_attack:
                continue

            scored_targets: list[tuple[float, Planet]] = []
            for target in local.targets:
                if target.id in local.comet_planet_ids and not self.include_comet_targets:
                    continue
                base_ships = self._base_ships_needed(target, local)
                if base_ships is None:
                    continue
                angle, arrival = self._angle_and_arrival(source, target, base_ships, local)
                if angle is None or arrival is None:
                    continue
                if self.use_late_filter and arrival > max(5, remaining_steps(local) - 5):
                    continue
                comet_life = None
                if target.id in local.comet_planet_ids:
                    comet_life = comet_remaining_from_obs(getattr(self, "_raw_obs", {}), target.id)
                    if self.use_comet_roi_filter and comet_life <= arrival:
                        continue
                public_score = public_custom_score(source, target)
                if self.use_rl_target_score:
                    strategic_score = strategic_target_score(
                        source,
                        target,
                        local,
                        public_score,
                        base_ships,
                        arrival,
                        comet_life=comet_life,
                    )
                    score = public_score + self.rl_score_weight * (strategic_score - public_score)
                else:
                    score = public_score
                scored_targets.append((score, target))

            scored_targets.sort(key=lambda row: row[0], reverse=True)
            for _, target in scored_targets[: self.custom_candidate_limit]:
                if self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break
                if self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break

        if getattr(self, "enable_front_support", False):
            self._append_front_support(local, under_attack, exhausted_planet_ids, moves)

    def _append_front_support(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if remaining_steps(local) < 60 or len(local.mine) <= 1:
            return
        enemy_planets = [p for p in local.planets if p.owner not in (-1, local.player)]
        if not enemy_planets:
            return

        front_dist = {
            planet.id: min(distance(planet, enemy) for enemy in enemy_planets)
            for planet in local.mine
        }
        front = min(local.mine, key=lambda p: front_dist[p.id])

        for rear in sorted(local.mine, key=lambda p: front_dist[p.id], reverse=True):
            if rear.id == front.id or rear.id in exhausted_planet_ids:
                continue
            if front_dist[rear.id] < front_dist[front.id] * self.support_distance_factor:
                continue
            available = self._available_ships(rear, under_attack)
            if available < self.support_min_available:
                continue
            ships = int(available * self.support_fraction)
            if ships < self.support_min_send:
                continue
            angle, arrival = self._support_angle(rear, front, ships, local)
            if angle is None or arrival is None or arrival > self.support_max_arrival:
                continue
            if self.enable_sun_avoidance and sun_collision(rear, ships, angle):
                continue
            moves.append([rear.id, float(angle), int(ships)])
            exhausted_planet_ids.add(rear.id)

    def _support_angle(
        self,
        source: Planet,
        target: Planet,
        ships: int,
        local: LocalObs,
    ) -> tuple[float | None, int | None]:
        if target.id in self.moving_planets:
            return find_angle_to_moving_planet(source, target, ships, local.angular_velocity)
        return angle_to(source, target), int(distance(source, target) / fleet_speed(ships))
