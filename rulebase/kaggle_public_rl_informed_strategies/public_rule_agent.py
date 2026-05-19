"""Stateful public rule agent distilled from the score-1049 public notebook."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

from .defense import max_enemy_fleet_to_target, planets_under_attack, reinforcement_plans
from .geometry import (
    angle_to,
    collides_segment_circle,
    distance,
    find_angle_to_moving_planet,
    fleet_speed,
    planet_trajectory,
    sun_collision,
    travel_ticks,
)
from .rl_signals import comet_remaining_from_obs
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
    enable_path_first_hit_check: bool = PUBLIC_EXACT.enable_path_first_hit_check
    enable_path_first_hit_redirect: bool = PUBLIC_EXACT.enable_path_first_hit_redirect
    path_first_hit_min_active_players: int = PUBLIC_EXACT.path_first_hit_min_active_players
    path_first_hit_padding: float = PUBLIC_EXACT.path_first_hit_padding
    path_block_wait_penalty: float = PUBLIC_EXACT.path_block_wait_penalty
    enable_comet_evacuation: bool = PUBLIC_EXACT.enable_comet_evacuation
    comet_evacuation_remaining_turns: int = PUBLIC_EXACT.comet_evacuation_remaining_turns
    comet_evacuation_min_ships: int = PUBLIC_EXACT.comet_evacuation_min_ships
    comet_evacuation_front_distance: float = PUBLIC_EXACT.comet_evacuation_front_distance
    comet_evacuation_own_prod_weight: float = PUBLIC_EXACT.comet_evacuation_own_prod_weight
    comet_evacuation_front_bonus: float = PUBLIC_EXACT.comet_evacuation_front_bonus
    comet_evacuation_target_roi: float = PUBLIC_EXACT.comet_evacuation_target_roi
    comet_evacuation_target_prod_weight: float = PUBLIC_EXACT.comet_evacuation_target_prod_weight
    comet_evacuation_target_enemy_bonus: float = PUBLIC_EXACT.comet_evacuation_target_enemy_bonus
    enable_endgame_fleet_dump: bool = PUBLIC_EXACT.enable_endgame_fleet_dump
    endgame_dump_min_step: int = PUBLIC_EXACT.endgame_dump_min_step
    endgame_dump_min_ships: int = PUBLIC_EXACT.endgame_dump_min_ships
    endgame_dump_keep_source_ships: int = PUBLIC_EXACT.endgame_dump_keep_source_ships
    endgame_dump_angle_samples: int = PUBLIC_EXACT.endgame_dump_angle_samples
    endgame_dump_min_active_players: int = PUBLIC_EXACT.endgame_dump_min_active_players
    enable_arrival_based_under_attack_availability: bool = PUBLIC_EXACT.enable_arrival_based_under_attack_availability
    under_attack_availability_min_step: int = PUBLIC_EXACT.under_attack_availability_min_step
    under_attack_availability_margin: int = PUBLIC_EXACT.under_attack_availability_margin
    under_attack_availability_horizon: int = PUBLIC_EXACT.under_attack_availability_horizon
    enable_capture_hold_margin_gate: bool = PUBLIC_EXACT.enable_capture_hold_margin_gate
    capture_hold_min_active_players: int = PUBLIC_EXACT.capture_hold_min_active_players
    capture_hold_min_production: float = PUBLIC_EXACT.capture_hold_min_production
    capture_hold_enemy_radius: float = PUBLIC_EXACT.capture_hold_enemy_radius
    capture_hold_enemy_send_fraction: float = PUBLIC_EXACT.capture_hold_enemy_send_fraction
    capture_hold_enemy_launch_window: int = PUBLIC_EXACT.capture_hold_enemy_launch_window
    capture_hold_enemy_reserve_turns: int = PUBLIC_EXACT.capture_hold_enemy_reserve_turns
    capture_hold_enemy_max_arrival: int = PUBLIC_EXACT.capture_hold_enemy_max_arrival
    capture_hold_margin: int = PUBLIC_EXACT.capture_hold_margin
    capture_hold_allow_extra_send: bool = PUBLIC_EXACT.capture_hold_allow_extra_send
    capture_hold_use_post_capture_window: bool = PUBLIC_EXACT.capture_hold_use_post_capture_window
    enable_contested_target_adjustment: bool = PUBLIC_EXACT.enable_contested_target_adjustment
    contested_arrival_margin: int = PUBLIC_EXACT.contested_arrival_margin
    contested_enemy_weight: float = PUBLIC_EXACT.contested_enemy_weight
    contested_friendly_credit: float = PUBLIC_EXACT.contested_friendly_credit
    contested_skip_friendly_covered: bool = PUBLIC_EXACT.contested_skip_friendly_covered
    enable_contested_stop_loss: bool = PUBLIC_EXACT.enable_contested_stop_loss
    contested_stop_loss_min_active_players: int = PUBLIC_EXACT.contested_stop_loss_min_active_players
    contested_stop_loss_max_active_players: int = PUBLIC_EXACT.contested_stop_loss_max_active_players
    contested_stop_loss_window: int = PUBLIC_EXACT.contested_stop_loss_window
    contested_stop_loss_flip_threshold: int = PUBLIC_EXACT.contested_stop_loss_flip_threshold
    contested_stop_loss_low_prod_max: float = PUBLIC_EXACT.contested_stop_loss_low_prod_max
    contested_stop_loss_high_prod_exception_min: float = PUBLIC_EXACT.contested_stop_loss_high_prod_exception_min
    contested_stop_loss_penalty: float = PUBLIC_EXACT.contested_stop_loss_penalty
    contested_stop_loss_min_hold: int = PUBLIC_EXACT.contested_stop_loss_min_hold
    contested_stop_loss_prod_hold_turns: int = PUBLIC_EXACT.contested_stop_loss_prod_hold_turns
    enable_dynamic_posture: bool = PUBLIC_EXACT.enable_dynamic_posture
    posture_max_active_players: int = PUBLIC_EXACT.posture_max_active_players
    posture_defensive_min_step: int = PUBLIC_EXACT.posture_defensive_min_step
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
    value_defense_multiplayer_min_active_players: int = PUBLIC_EXACT.value_defense_multiplayer_min_active_players
    value_defense_multiplayer_horizon: int = PUBLIC_EXACT.value_defense_multiplayer_horizon
    value_defense_multiplayer_max_send: int = PUBLIC_EXACT.value_defense_multiplayer_max_send
    value_defense_multiplayer_min_margin: int = PUBLIC_EXACT.value_defense_multiplayer_min_margin
    enable_enemy_wave_preserve_prod: bool = PUBLIC_EXACT.enable_enemy_wave_preserve_prod
    enemy_wave_preserve_min_active_players: int = PUBLIC_EXACT.enemy_wave_preserve_min_active_players
    enemy_wave_preserve_max_active_players: int = PUBLIC_EXACT.enemy_wave_preserve_max_active_players
    enemy_wave_preserve_min_step: int = PUBLIC_EXACT.enemy_wave_preserve_min_step
    enemy_wave_preserve_max_step: int = PUBLIC_EXACT.enemy_wave_preserve_max_step
    enemy_wave_preserve_min_production: float = PUBLIC_EXACT.enemy_wave_preserve_min_production
    enemy_wave_preserve_horizon: int = PUBLIC_EXACT.enemy_wave_preserve_horizon
    enemy_wave_preserve_min_enemy_post_capture: int = PUBLIC_EXACT.enemy_wave_preserve_min_enemy_post_capture
    enemy_wave_preserve_margin: int = PUBLIC_EXACT.enemy_wave_preserve_margin
    enemy_wave_preserve_min_send: int = PUBLIC_EXACT.enemy_wave_preserve_min_send
    enemy_wave_preserve_max_send: int = PUBLIC_EXACT.enemy_wave_preserve_max_send
    enemy_wave_preserve_max_sources: int = PUBLIC_EXACT.enemy_wave_preserve_max_sources
    enemy_wave_preserve_max_targets: int = PUBLIC_EXACT.enemy_wave_preserve_max_targets
    enemy_wave_preserve_source_min_after: int = PUBLIC_EXACT.enemy_wave_preserve_source_min_after
    enemy_wave_preserve_source_prod_turns_after: int = PUBLIC_EXACT.enemy_wave_preserve_source_prod_turns_after
    enable_proactive_value_defense: bool = PUBLIC_EXACT.enable_proactive_value_defense
    proactive_defense_min_active_players: int = PUBLIC_EXACT.proactive_defense_min_active_players
    proactive_defense_max_active_players: int = PUBLIC_EXACT.proactive_defense_max_active_players
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
    local_reserve_min_step: int = PUBLIC_EXACT.local_reserve_min_step
    local_reserve_max_step: int = PUBLIC_EXACT.local_reserve_max_step
    local_reserve_min_production: float = PUBLIC_EXACT.local_reserve_min_production
    local_reserve_enemy_distance: float = PUBLIC_EXACT.local_reserve_enemy_distance
    local_reserve_turns: int = PUBLIC_EXACT.local_reserve_turns
    local_reserve_min_garrison: int = PUBLIC_EXACT.local_reserve_min_garrison
    local_reserve_front_bonus: int = PUBLIC_EXACT.local_reserve_front_bonus
    enable_recent_capture_source_reserve: bool = PUBLIC_EXACT.enable_recent_capture_source_reserve
    recent_capture_source_reserve_min_active_players: int = PUBLIC_EXACT.recent_capture_source_reserve_min_active_players
    recent_capture_source_reserve_min_step: int = PUBLIC_EXACT.recent_capture_source_reserve_min_step
    recent_capture_source_reserve_max_step: int = PUBLIC_EXACT.recent_capture_source_reserve_max_step
    recent_capture_source_reserve_window: int = PUBLIC_EXACT.recent_capture_source_reserve_window
    recent_capture_source_reserve_min_production: float = PUBLIC_EXACT.recent_capture_source_reserve_min_production
    recent_capture_source_reserve_min_after: int = PUBLIC_EXACT.recent_capture_source_reserve_min_after
    recent_capture_source_reserve_prod_turns_after: int = PUBLIC_EXACT.recent_capture_source_reserve_prod_turns_after
    recent_capture_source_reserve_enemy_radius: float = PUBLIC_EXACT.recent_capture_source_reserve_enemy_radius
    recent_capture_source_reserve_front_bonus: int = PUBLIC_EXACT.recent_capture_source_reserve_front_bonus
    enable_source_threat_reserve: bool = PUBLIC_EXACT.enable_source_threat_reserve
    source_threat_min_active_players: int = PUBLIC_EXACT.source_threat_min_active_players
    source_threat_min_step: int = PUBLIC_EXACT.source_threat_min_step
    source_threat_max_step: int = PUBLIC_EXACT.source_threat_max_step
    source_threat_min_production: float = PUBLIC_EXACT.source_threat_min_production
    source_threat_radius: float = PUBLIC_EXACT.source_threat_radius
    source_threat_enemy_send_fraction: float = PUBLIC_EXACT.source_threat_enemy_send_fraction
    source_threat_enemy_launch_window: int = PUBLIC_EXACT.source_threat_enemy_launch_window
    source_threat_enemy_reserve_turns: int = PUBLIC_EXACT.source_threat_enemy_reserve_turns
    source_threat_max_arrival: int = PUBLIC_EXACT.source_threat_max_arrival
    source_threat_margin: int = PUBLIC_EXACT.source_threat_margin
    source_threat_roi_multiplier: float = PUBLIC_EXACT.source_threat_roi_multiplier
    source_threat_min_net_value: float = PUBLIC_EXACT.source_threat_min_net_value
    enable_source_threat_send_filter: bool = PUBLIC_EXACT.enable_source_threat_send_filter
    source_threat_send_min_active_players: int = PUBLIC_EXACT.source_threat_send_min_active_players
    source_threat_send_max_active_players: int = PUBLIC_EXACT.source_threat_send_max_active_players
    source_threat_send_min_step: int = PUBLIC_EXACT.source_threat_send_min_step
    source_threat_send_max_step: int = PUBLIC_EXACT.source_threat_send_max_step
    source_threat_send_min_production: float = PUBLIC_EXACT.source_threat_send_min_production
    source_threat_send_radius: float = PUBLIC_EXACT.source_threat_send_radius
    source_threat_send_enemy_fraction: float = PUBLIC_EXACT.source_threat_send_enemy_fraction
    source_threat_send_enemy_launch_window: int = PUBLIC_EXACT.source_threat_send_enemy_launch_window
    source_threat_send_enemy_reserve_turns: int = PUBLIC_EXACT.source_threat_send_enemy_reserve_turns
    source_threat_send_max_arrival: int = PUBLIC_EXACT.source_threat_send_max_arrival
    source_threat_send_margin: int = PUBLIC_EXACT.source_threat_send_margin
    source_threat_send_roi_multiplier: float = PUBLIC_EXACT.source_threat_send_roi_multiplier
    source_threat_send_min_net_value: float = PUBLIC_EXACT.source_threat_send_min_net_value
    source_threat_send_trade_ratio: float = PUBLIC_EXACT.source_threat_send_trade_ratio
    enable_source_threat_target_penalty: bool = PUBLIC_EXACT.enable_source_threat_target_penalty
    source_threat_target_penalty_weight: float = PUBLIC_EXACT.source_threat_target_penalty_weight
    enable_local_source_defense_gate: bool = PUBLIC_EXACT.enable_local_source_defense_gate
    local_source_defense_gate_min_active_players: int = PUBLIC_EXACT.local_source_defense_gate_min_active_players
    local_source_defense_gate_min_step: int = PUBLIC_EXACT.local_source_defense_gate_min_step
    local_source_defense_gate_max_step: int = PUBLIC_EXACT.local_source_defense_gate_max_step
    local_source_defense_gate_min_production: float = PUBLIC_EXACT.local_source_defense_gate_min_production
    local_source_defense_gate_front_distance: float = PUBLIC_EXACT.local_source_defense_gate_front_distance
    local_source_defense_gate_enemy_fraction: float = PUBLIC_EXACT.local_source_defense_gate_enemy_fraction
    local_source_defense_gate_enemy_launch_window: int = PUBLIC_EXACT.local_source_defense_gate_enemy_launch_window
    local_source_defense_gate_enemy_reserve_turns: int = PUBLIC_EXACT.local_source_defense_gate_enemy_reserve_turns
    local_source_defense_gate_max_arrival: int = PUBLIC_EXACT.local_source_defense_gate_max_arrival
    local_source_defense_gate_margin: int = PUBLIC_EXACT.local_source_defense_gate_margin
    local_source_defense_gate_use_arrival_production: bool = PUBLIC_EXACT.local_source_defense_gate_use_arrival_production
    enable_holdability_target_score: bool = PUBLIC_EXACT.enable_holdability_target_score
    holdability_radius: float = PUBLIC_EXACT.holdability_radius
    holdability_weight: float = PUBLIC_EXACT.holdability_weight
    holdability_enemy_prod_weight: float = PUBLIC_EXACT.holdability_enemy_prod_weight
    holdability_enemy_ship_weight: float = PUBLIC_EXACT.holdability_enemy_ship_weight
    holdability_own_prod_weight: float = PUBLIC_EXACT.holdability_own_prod_weight
    holdability_own_ship_weight: float = PUBLIC_EXACT.holdability_own_ship_weight
    enable_early_neutral_bias: bool = PUBLIC_EXACT.enable_early_neutral_bias
    early_neutral_min_active_players: int = PUBLIC_EXACT.early_neutral_min_active_players
    early_neutral_max_active_players: int = PUBLIC_EXACT.early_neutral_max_active_players
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
    enable_early_neutral_multiplayer_override: bool = PUBLIC_EXACT.enable_early_neutral_multiplayer_override
    early_neutral_multiplayer_min_active_players: int = PUBLIC_EXACT.early_neutral_multiplayer_min_active_players
    early_neutral_multiplayer_max_active_players: int = PUBLIC_EXACT.early_neutral_multiplayer_max_active_players
    early_neutral_multiplayer_step_limit: int = PUBLIC_EXACT.early_neutral_multiplayer_step_limit
    early_neutral_multiplayer_min_production: float = PUBLIC_EXACT.early_neutral_multiplayer_min_production
    early_neutral_multiplayer_bonus: float = PUBLIC_EXACT.early_neutral_multiplayer_bonus
    early_neutral_multiplayer_safe_bonus: float = PUBLIC_EXACT.early_neutral_multiplayer_safe_bonus
    early_neutral_multiplayer_contested_penalty: float = PUBLIC_EXACT.early_neutral_multiplayer_contested_penalty
    enable_early_neutral_dynamic_max_ships: bool = PUBLIC_EXACT.enable_early_neutral_dynamic_max_ships
    early_neutral_dynamic_max_ships: int = PUBLIC_EXACT.early_neutral_dynamic_max_ships
    early_neutral_dynamic_min_production: float = PUBLIC_EXACT.early_neutral_dynamic_min_production
    early_neutral_dynamic_min_enemy_gap: int = PUBLIC_EXACT.early_neutral_dynamic_min_enemy_gap
    early_neutral_dynamic_source_min_after: int = PUBLIC_EXACT.early_neutral_dynamic_source_min_after
    early_neutral_dynamic_check_source_safety: bool = PUBLIC_EXACT.early_neutral_dynamic_check_source_safety
    early_neutral_dynamic_source_threat_radius: float = PUBLIC_EXACT.early_neutral_dynamic_source_threat_radius
    early_neutral_dynamic_source_safety_margin: int = PUBLIC_EXACT.early_neutral_dynamic_source_safety_margin
    early_neutral_dynamic_check_target_hold: bool = PUBLIC_EXACT.early_neutral_dynamic_check_target_hold
    early_neutral_dynamic_target_hold_margin: int = PUBLIC_EXACT.early_neutral_dynamic_target_hold_margin
    enable_opening_neutral_territory_score: bool = PUBLIC_EXACT.enable_opening_neutral_territory_score
    opening_territory_min_active_players: int = PUBLIC_EXACT.opening_territory_min_active_players
    opening_territory_step_limit: int = PUBLIC_EXACT.opening_territory_step_limit
    opening_territory_enemy_closer_margin: float = PUBLIC_EXACT.opening_territory_enemy_closer_margin
    opening_territory_penalty: float = PUBLIC_EXACT.opening_territory_penalty
    opening_territory_prod_scale: float = PUBLIC_EXACT.opening_territory_prod_scale
    opening_territory_allow_if_safe_gap: int = PUBLIC_EXACT.opening_territory_allow_if_safe_gap
    enable_opening_neutral_hold_margin: bool = PUBLIC_EXACT.enable_opening_neutral_hold_margin
    opening_hold_min_active_players: int = PUBLIC_EXACT.opening_hold_min_active_players
    opening_hold_max_active_players: int = PUBLIC_EXACT.opening_hold_max_active_players
    opening_hold_step_limit: int = PUBLIC_EXACT.opening_hold_step_limit
    opening_hold_min_production: float = PUBLIC_EXACT.opening_hold_min_production
    opening_hold_base_margin: int = PUBLIC_EXACT.opening_hold_base_margin
    opening_hold_prod_turns: int = PUBLIC_EXACT.opening_hold_prod_turns
    opening_hold_contested_extra: int = PUBLIC_EXACT.opening_hold_contested_extra
    opening_hold_allow_extra_send: bool = PUBLIC_EXACT.opening_hold_allow_extra_send
    enable_opening_rotating_neutral_filter: bool = PUBLIC_EXACT.enable_opening_rotating_neutral_filter
    opening_rotating_step_limit: int = PUBLIC_EXACT.opening_rotating_step_limit
    opening_rotating_max_eta: int = PUBLIC_EXACT.opening_rotating_max_eta
    opening_rotating_low_production: float = PUBLIC_EXACT.opening_rotating_low_production
    opening_rotating_penalty: float = PUBLIC_EXACT.opening_rotating_penalty
    enable_opening_high_prod_trickle: bool = PUBLIC_EXACT.enable_opening_high_prod_trickle
    opening_trickle_min_active_players: int = PUBLIC_EXACT.opening_trickle_min_active_players
    opening_trickle_max_active_players: int = PUBLIC_EXACT.opening_trickle_max_active_players
    opening_trickle_step_limit: int = PUBLIC_EXACT.opening_trickle_step_limit
    opening_trickle_source_min_production: float = PUBLIC_EXACT.opening_trickle_source_min_production
    opening_trickle_target_min_production: float = PUBLIC_EXACT.opening_trickle_target_min_production
    opening_trickle_max_target_ships: int = PUBLIC_EXACT.opening_trickle_max_target_ships
    opening_trickle_min_ships: int = PUBLIC_EXACT.opening_trickle_min_ships
    enable_enemy_launch_punish: bool = PUBLIC_EXACT.enable_enemy_launch_punish
    enemy_launch_punish_max_fleet_age: int = PUBLIC_EXACT.enemy_launch_punish_max_fleet_age
    enemy_launch_punish_min_outgoing: int = PUBLIC_EXACT.enemy_launch_punish_min_outgoing
    enemy_launch_punish_min_production: float = PUBLIC_EXACT.enemy_launch_punish_min_production
    enemy_launch_punish_bonus_weight: float = PUBLIC_EXACT.enemy_launch_punish_bonus_weight
    enemy_launch_punish_max_targets: int = PUBLIC_EXACT.enemy_launch_punish_max_targets
    enemy_launch_punish_min_step: int = PUBLIC_EXACT.enemy_launch_punish_min_step
    enemy_launch_punish_max_step: int = PUBLIC_EXACT.enemy_launch_punish_max_step
    enemy_launch_punish_min_prod_diff: float = PUBLIC_EXACT.enemy_launch_punish_min_prod_diff
    enemy_launch_punish_min_planet_diff: int = PUBLIC_EXACT.enemy_launch_punish_min_planet_diff
    enemy_launch_punish_min_ship_ratio: float = PUBLIC_EXACT.enemy_launch_punish_min_ship_ratio
    enable_recent_loss_recapture_bias: bool = PUBLIC_EXACT.enable_recent_loss_recapture_bias
    recent_loss_recapture_min_production: float = PUBLIC_EXACT.recent_loss_recapture_min_production
    recent_loss_recapture_window: int = PUBLIC_EXACT.recent_loss_recapture_window
    recent_loss_recapture_min_step: int = PUBLIC_EXACT.recent_loss_recapture_min_step
    recent_loss_recapture_max_step: int = PUBLIC_EXACT.recent_loss_recapture_max_step
    recent_loss_recapture_bonus: float = PUBLIC_EXACT.recent_loss_recapture_bonus
    recent_loss_recapture_prod_weight: float = PUBLIC_EXACT.recent_loss_recapture_prod_weight
    enable_dynamic_front_base_recapture: bool = PUBLIC_EXACT.enable_dynamic_front_base_recapture
    dynamic_recapture_min_production: float = PUBLIC_EXACT.dynamic_recapture_min_production
    dynamic_recapture_enable_2p: bool = PUBLIC_EXACT.dynamic_recapture_enable_2p
    dynamic_recapture_enable_4p_collapse: bool = PUBLIC_EXACT.dynamic_recapture_enable_4p_collapse
    dynamic_recapture_collapse_loss_count: int = PUBLIC_EXACT.dynamic_recapture_collapse_loss_count
    dynamic_recapture_collapse_window: int = PUBLIC_EXACT.dynamic_recapture_collapse_window
    dynamic_recapture_enable_front_base: bool = PUBLIC_EXACT.dynamic_recapture_enable_front_base
    dynamic_recapture_front_own_radius: float = PUBLIC_EXACT.dynamic_recapture_front_own_radius
    dynamic_recapture_front_min_own_neighbors: int = PUBLIC_EXACT.dynamic_recapture_front_min_own_neighbors
    dynamic_recapture_anchor_window: int = PUBLIC_EXACT.dynamic_recapture_anchor_window
    dynamic_recapture_anchor_min_production: float = PUBLIC_EXACT.dynamic_recapture_anchor_min_production
    dynamic_recapture_anchor_radius: float = PUBLIC_EXACT.dynamic_recapture_anchor_radius
    enable_recent_loss_recapture_hold_gate: bool = PUBLIC_EXACT.enable_recent_loss_recapture_hold_gate
    recent_loss_recapture_hold_min_active_players: int = PUBLIC_EXACT.recent_loss_recapture_hold_min_active_players
    recent_loss_recapture_hold_min_step: int = PUBLIC_EXACT.recent_loss_recapture_hold_min_step
    recent_loss_recapture_hold_max_step: int = PUBLIC_EXACT.recent_loss_recapture_hold_max_step
    recent_loss_recapture_hold_window: int = PUBLIC_EXACT.recent_loss_recapture_hold_window
    recent_loss_recapture_hold_min_production: float = PUBLIC_EXACT.recent_loss_recapture_hold_min_production
    recent_loss_recapture_hold_enemy_radius: float = PUBLIC_EXACT.recent_loss_recapture_hold_enemy_radius
    recent_loss_recapture_hold_margin: int = PUBLIC_EXACT.recent_loss_recapture_hold_margin
    recent_loss_recapture_hold_allow_extra_send: bool = PUBLIC_EXACT.recent_loss_recapture_hold_allow_extra_send
    enable_third_party_tail_capture: bool = PUBLIC_EXACT.enable_third_party_tail_capture
    third_party_tail_min_active_players: int = PUBLIC_EXACT.third_party_tail_min_active_players
    third_party_tail_min_step: int = PUBLIC_EXACT.third_party_tail_min_step
    third_party_tail_max_step: int = PUBLIC_EXACT.third_party_tail_max_step
    third_party_tail_min_production: float = PUBLIC_EXACT.third_party_tail_min_production
    third_party_tail_max_enemy_arrival: int = PUBLIC_EXACT.third_party_tail_max_enemy_arrival
    third_party_tail_min_delay: int = PUBLIC_EXACT.third_party_tail_min_delay
    third_party_tail_max_delay: int = PUBLIC_EXACT.third_party_tail_max_delay
    third_party_tail_margin: int = PUBLIC_EXACT.third_party_tail_margin
    third_party_tail_min_send: int = PUBLIC_EXACT.third_party_tail_min_send
    third_party_tail_max_ships: int = PUBLIC_EXACT.third_party_tail_max_ships
    third_party_tail_source_min_after: int = PUBLIC_EXACT.third_party_tail_source_min_after
    third_party_tail_bonus: float = PUBLIC_EXACT.third_party_tail_bonus
    third_party_tail_prod_weight: float = PUBLIC_EXACT.third_party_tail_prod_weight
    third_party_tail_savings_weight: float = PUBLIC_EXACT.third_party_tail_savings_weight
    third_party_tail_min_savings: int = PUBLIC_EXACT.third_party_tail_min_savings
    third_party_tail_min_savings_ratio: float = PUBLIC_EXACT.third_party_tail_min_savings_ratio
    third_party_tail_min_post_capture_ships: int = PUBLIC_EXACT.third_party_tail_min_post_capture_ships
    third_party_tail_overpay_min_post_capture: int = PUBLIC_EXACT.third_party_tail_overpay_min_post_capture
    third_party_tail_neutral_max_arrival: int = PUBLIC_EXACT.third_party_tail_neutral_max_arrival
    third_party_tail_neutral_min_post_capture_ships: int = PUBLIC_EXACT.third_party_tail_neutral_min_post_capture_ships
    third_party_tail_neutral_min_enemy_post_capture: int = PUBLIC_EXACT.third_party_tail_neutral_min_enemy_post_capture
    third_party_tail_roi_multiplier: float = PUBLIC_EXACT.third_party_tail_roi_multiplier
    third_party_tail_min_net_value: float = PUBLIC_EXACT.third_party_tail_min_net_value
    enable_third_party_tail_candidate_injection: bool = PUBLIC_EXACT.enable_third_party_tail_candidate_injection
    third_party_tail_candidate_limit: int = PUBLIC_EXACT.third_party_tail_candidate_limit
    third_party_tail_candidate_min_score: float = PUBLIC_EXACT.third_party_tail_candidate_min_score
    third_party_tail_candidate_keep_front: int = PUBLIC_EXACT.third_party_tail_candidate_keep_front
    third_party_tail_only_neutral_targets: bool = PUBLIC_EXACT.third_party_tail_only_neutral_targets
    enable_third_party_tail_hold_filter: bool = PUBLIC_EXACT.enable_third_party_tail_hold_filter
    third_party_tail_hold_enemy_radius: float = PUBLIC_EXACT.third_party_tail_hold_enemy_radius
    third_party_tail_hold_enemy_send_fraction: float = PUBLIC_EXACT.third_party_tail_hold_enemy_send_fraction
    third_party_tail_hold_enemy_launch_window: int = PUBLIC_EXACT.third_party_tail_hold_enemy_launch_window
    third_party_tail_hold_enemy_reserve_turns: int = PUBLIC_EXACT.third_party_tail_hold_enemy_reserve_turns
    third_party_tail_hold_enemy_max_arrival: int = PUBLIC_EXACT.third_party_tail_hold_enemy_max_arrival
    third_party_tail_hold_margin: int = PUBLIC_EXACT.third_party_tail_hold_margin
    enable_third_party_anti_tail_hold_gate: bool = PUBLIC_EXACT.enable_third_party_anti_tail_hold_gate
    anti_tail_hold_min_active_players: int = PUBLIC_EXACT.anti_tail_hold_min_active_players
    anti_tail_hold_min_production: float = PUBLIC_EXACT.anti_tail_hold_min_production
    anti_tail_hold_enemy_radius: float = PUBLIC_EXACT.anti_tail_hold_enemy_radius
    anti_tail_hold_enemy_send_fraction: float = PUBLIC_EXACT.anti_tail_hold_enemy_send_fraction
    anti_tail_hold_enemy_launch_window: int = PUBLIC_EXACT.anti_tail_hold_enemy_launch_window
    anti_tail_hold_enemy_reserve_turns: int = PUBLIC_EXACT.anti_tail_hold_enemy_reserve_turns
    anti_tail_hold_enemy_max_arrival: int = PUBLIC_EXACT.anti_tail_hold_enemy_max_arrival
    anti_tail_hold_margin: int = PUBLIC_EXACT.anti_tail_hold_margin
    enable_third_party_tail_watchlist: bool = PUBLIC_EXACT.enable_third_party_tail_watchlist
    third_party_tail_watchlist_horizon: int = PUBLIC_EXACT.third_party_tail_watchlist_horizon
    third_party_tail_watchlist_post_window: int = PUBLIC_EXACT.third_party_tail_watchlist_post_window
    third_party_tail_watchlist_max_entries: int = PUBLIC_EXACT.third_party_tail_watchlist_max_entries
    third_party_tail_watchlist_score_bonus: float = PUBLIC_EXACT.third_party_tail_watchlist_score_bonus
    third_party_tail_watchlist_min_recheck_age: int = PUBLIC_EXACT.third_party_tail_watchlist_min_recheck_age
    enable_global_attack_priority: bool = PUBLIC_EXACT.enable_global_attack_priority
    global_attack_roi_weight: float = PUBLIC_EXACT.global_attack_roi_weight
    global_attack_arrival_penalty: float = PUBLIC_EXACT.global_attack_arrival_penalty
    global_attack_max_failed_pairs: int = PUBLIC_EXACT.global_attack_max_failed_pairs
    enable_no_attack_fallback: bool = PUBLIC_EXACT.enable_no_attack_fallback
    no_attack_fallback_min_active_players: int = PUBLIC_EXACT.no_attack_fallback_min_active_players
    no_attack_fallback_max_active_players: int = PUBLIC_EXACT.no_attack_fallback_max_active_players
    no_attack_fallback_min_step: int = PUBLIC_EXACT.no_attack_fallback_min_step
    no_attack_fallback_max_step: int = PUBLIC_EXACT.no_attack_fallback_max_step
    no_attack_fallback_candidate_limit: int = PUBLIC_EXACT.no_attack_fallback_candidate_limit
    no_attack_fallback_min_attack_delta: int = PUBLIC_EXACT.no_attack_fallback_min_attack_delta
    enable_opening_tempo_neutral_fallback: bool = PUBLIC_EXACT.enable_opening_tempo_neutral_fallback
    opening_tempo_min_active_players: int = PUBLIC_EXACT.opening_tempo_min_active_players
    opening_tempo_max_active_players: int = PUBLIC_EXACT.opening_tempo_max_active_players
    opening_tempo_step_limit: int = PUBLIC_EXACT.opening_tempo_step_limit
    opening_tempo_min_production: float = PUBLIC_EXACT.opening_tempo_min_production
    opening_tempo_max_target_ships: int = PUBLIC_EXACT.opening_tempo_max_target_ships
    opening_tempo_max_eta: int = PUBLIC_EXACT.opening_tempo_max_eta
    opening_tempo_source_min_production: float = PUBLIC_EXACT.opening_tempo_source_min_production
    opening_tempo_source_min_after: int = PUBLIC_EXACT.opening_tempo_source_min_after
    opening_tempo_candidate_limit: int = PUBLIC_EXACT.opening_tempo_candidate_limit
    enable_recent_capture_chain_attack: bool = PUBLIC_EXACT.enable_recent_capture_chain_attack
    chain_attack_min_active_players: int = PUBLIC_EXACT.chain_attack_min_active_players
    chain_attack_max_active_players: int = PUBLIC_EXACT.chain_attack_max_active_players
    chain_attack_min_step: int = PUBLIC_EXACT.chain_attack_min_step
    chain_attack_max_step: int = PUBLIC_EXACT.chain_attack_max_step
    chain_attack_source_window: int = PUBLIC_EXACT.chain_attack_source_window
    chain_attack_source_min_production: float = PUBLIC_EXACT.chain_attack_source_min_production
    chain_attack_target_min_production: float = PUBLIC_EXACT.chain_attack_target_min_production
    chain_attack_max_eta: int = PUBLIC_EXACT.chain_attack_max_eta
    chain_attack_source_order_bonus: float = PUBLIC_EXACT.chain_attack_source_order_bonus
    chain_attack_enemy_bonus: float = PUBLIC_EXACT.chain_attack_enemy_bonus
    chain_attack_neutral_bonus: float = PUBLIC_EXACT.chain_attack_neutral_bonus
    enable_mobile_relay_attack: bool = PUBLIC_EXACT.enable_mobile_relay_attack
    mobile_relay_min_active_players: int = PUBLIC_EXACT.mobile_relay_min_active_players
    mobile_relay_max_active_players: int = PUBLIC_EXACT.mobile_relay_max_active_players
    mobile_relay_min_step: int = PUBLIC_EXACT.mobile_relay_min_step
    mobile_relay_max_step: int = PUBLIC_EXACT.mobile_relay_max_step
    mobile_relay_max_ships: int = PUBLIC_EXACT.mobile_relay_max_ships
    mobile_relay_min_production: float = PUBLIC_EXACT.mobile_relay_min_production
    mobile_relay_goal_min_production: float = PUBLIC_EXACT.mobile_relay_goal_min_production
    mobile_relay_direct_min_eta: int = PUBLIC_EXACT.mobile_relay_direct_min_eta
    mobile_relay_max_first_eta: int = PUBLIC_EXACT.mobile_relay_max_first_eta
    mobile_relay_max_second_eta: int = PUBLIC_EXACT.mobile_relay_max_second_eta
    mobile_relay_min_eta_savings: int = PUBLIC_EXACT.mobile_relay_min_eta_savings
    mobile_relay_bonus: float = PUBLIC_EXACT.mobile_relay_bonus
    mobile_relay_comet_min_remaining: int = PUBLIC_EXACT.mobile_relay_comet_min_remaining
    mobile_relay_recent_source_window: int = PUBLIC_EXACT.mobile_relay_recent_source_window
    mobile_relay_source_order_bonus: float = PUBLIC_EXACT.mobile_relay_source_order_bonus
    mobile_relay_source_target_bonus: float = PUBLIC_EXACT.mobile_relay_source_target_bonus
    enable_recent_high_prod_hub_support: bool = PUBLIC_EXACT.enable_recent_high_prod_hub_support
    hub_support_min_active_players: int = PUBLIC_EXACT.hub_support_min_active_players
    hub_support_max_active_players: int = PUBLIC_EXACT.hub_support_max_active_players
    hub_support_min_step: int = PUBLIC_EXACT.hub_support_min_step
    hub_support_max_step: int = PUBLIC_EXACT.hub_support_max_step
    hub_support_recent_capture_window: int = PUBLIC_EXACT.hub_support_recent_capture_window
    hub_support_min_production: float = PUBLIC_EXACT.hub_support_min_production
    hub_support_enemy_radius: float = PUBLIC_EXACT.hub_support_enemy_radius
    hub_support_base_margin: int = PUBLIC_EXACT.hub_support_base_margin
    hub_support_prod_turns: int = PUBLIC_EXACT.hub_support_prod_turns
    hub_support_front_bonus: int = PUBLIC_EXACT.hub_support_front_bonus
    hub_support_max_eta: int = PUBLIC_EXACT.hub_support_max_eta
    hub_support_min_send: int = PUBLIC_EXACT.hub_support_min_send
    hub_support_max_send: int = PUBLIC_EXACT.hub_support_max_send
    hub_support_source_min_after: int = PUBLIC_EXACT.hub_support_source_min_after
    hub_support_source_prod_turns_after: int = PUBLIC_EXACT.hub_support_source_prod_turns_after
    hub_support_max_targets: int = PUBLIC_EXACT.hub_support_max_targets
    enable_multiplayer_diplomacy_score: bool = PUBLIC_EXACT.enable_multiplayer_diplomacy_score
    multiplayer_min_active_players: int = PUBLIC_EXACT.multiplayer_min_active_players
    multiplayer_far_enemy_distance: float = PUBLIC_EXACT.multiplayer_far_enemy_distance
    multiplayer_far_enemy_penalty: float = PUBLIC_EXACT.multiplayer_far_enemy_penalty
    multiplayer_local_enemy_bonus: float = PUBLIC_EXACT.multiplayer_local_enemy_bonus
    multiplayer_leader_prod_bonus: float = PUBLIC_EXACT.multiplayer_leader_prod_bonus
    multiplayer_neutral_bonus: float = PUBLIC_EXACT.multiplayer_neutral_bonus
    enable_home_anchor_source_reserve: bool = PUBLIC_EXACT.enable_home_anchor_source_reserve
    home_anchor_min_active_players: int = PUBLIC_EXACT.home_anchor_min_active_players
    home_anchor_min_production: float = PUBLIC_EXACT.home_anchor_min_production
    home_anchor_step_min: int = PUBLIC_EXACT.home_anchor_step_min
    home_anchor_step_max: int = PUBLIC_EXACT.home_anchor_step_max
    home_anchor_home_radius: float = PUBLIC_EXACT.home_anchor_home_radius
    home_anchor_min_after: int = PUBLIC_EXACT.home_anchor_min_after
    home_anchor_prod_turns_after: int = PUBLIC_EXACT.home_anchor_prod_turns_after
    home_anchor_front_threat_bonus: int = PUBLIC_EXACT.home_anchor_front_threat_bonus
    enable_midgame_border_source_reserve: bool = PUBLIC_EXACT.enable_midgame_border_source_reserve
    midgame_border_min_active_players: int = PUBLIC_EXACT.midgame_border_min_active_players
    midgame_border_step_min: int = PUBLIC_EXACT.midgame_border_step_min
    midgame_border_step_max: int = PUBLIC_EXACT.midgame_border_step_max
    midgame_border_min_production: float = PUBLIC_EXACT.midgame_border_min_production
    midgame_border_enemy_radius: float = PUBLIC_EXACT.midgame_border_enemy_radius
    midgame_border_min_after: int = PUBLIC_EXACT.midgame_border_min_after
    midgame_border_prod_turns_after: int = PUBLIC_EXACT.midgame_border_prod_turns_after
    midgame_border_threat_margin: int = PUBLIC_EXACT.midgame_border_threat_margin
    fleet_trajectories: list[dict[str, object]] = field(default_factory=list)
    reinforcement_trajectories: list[dict[str, object]] = field(default_factory=list)
    moving_planets: set[int] = field(default_factory=set)
    home_anchor_positions: list[tuple[float, float]] = field(default_factory=list)
    previous_owner_by_planet: dict[int, int] = field(default_factory=dict)
    recently_captured_steps: dict[int, int] = field(default_factory=dict)
    planet_flip_steps: dict[int, list[int]] = field(default_factory=dict)
    recently_lost_steps: dict[int, int] = field(default_factory=dict)
    third_party_tail_watchlist: dict[tuple[int, int], dict[str, object]] = field(default_factory=dict)
    comet_remaining_by_planet: dict[int, int] = field(default_factory=dict)
    did_attack_this_turn: bool = False
    steps_seen: int = 0

    def __call__(self, obs, configuration=None) -> list[list[float | int]]:
        return self.act(obs)

    def act(self, obs) -> list[list[float | int]]:
        self.steps_seen += 1
        if self.steps_seen <= self.warmup_steps:
            return []

        local = parse_observation(obs)
        self.comet_remaining_by_planet = {
            planet_id: comet_remaining_from_obs(obs, planet_id)
            for planet_id in local.comet_planet_ids
        }
        if self.steps_seen == self.warmup_steps + 1:
            self._fill_moving_planets(local)

        self._update_recent_captures(local)
        self._update_fleet_trajectories(local)
        self._update_reinforcement_trajectories()
        self._update_third_party_tail_watchlist(local)

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
        if self.enable_enemy_wave_preserve_prod:
            self._append_enemy_wave_preserve_prod(local, under_attack, exhausted_planet_ids, moves)
        if self.enable_recent_high_prod_hub_support:
            self._append_recent_high_prod_hub_support(local, under_attack, exhausted_planet_ids, moves)
        if self.enable_comet_evacuation:
            self._append_comet_evacuation(local, under_attack, exhausted_planet_ids, moves)
        attack_count_before = len(self.fleet_trajectories)
        self._append_attacks(local, under_attack, exhausted_planet_ids, moves)
        self.did_attack_this_turn = len(self.fleet_trajectories) > attack_count_before
        if not self.did_attack_this_turn:
            self._append_opening_tempo_neutral_fallback(local, under_attack, exhausted_planet_ids, moves)
            self.did_attack_this_turn = len(self.fleet_trajectories) > attack_count_before
        if not self.did_attack_this_turn:
            self._append_no_attack_fallback(local, under_attack, exhausted_planet_ids, moves)
            self.did_attack_this_turn = len(self.fleet_trajectories) > attack_count_before
        if self.enable_proactive_value_defense and self.proactive_defense_after_attacks:
            self._append_proactive_value_defense(local, under_attack, exhausted_planet_ids, moves)
        if self.enable_endgame_fleet_dump:
            self._append_endgame_fleet_dump(local, under_attack, exhausted_planet_ids, moves)
        return moves

    def _fill_moving_planets(self, local: LocalObs) -> None:
        if not self.home_anchor_positions:
            self.home_anchor_positions = [(planet.x, planet.y) for planet in local.mine]
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
            if previous_owner is not None and previous_owner != planet.owner:
                flips = self.planet_flip_steps.setdefault(planet.id, [])
                flips.append(local.step)
                keep_flip_after = local.step - max(1, self.contested_stop_loss_window)
                while flips and flips[0] < keep_flip_after:
                    flips.pop(0)
            if previous_owner is not None and previous_owner != local.player and planet.owner == local.player:
                self.recently_captured_steps[planet.id] = local.step
                self.recently_lost_steps.pop(planet.id, None)
            elif (
                previous_owner == local.player
                and planet.owner != local.player
                and planet.production >= self._recent_loss_tracking_min_production(local)
            ):
                self.recently_lost_steps[planet.id] = local.step
            self.previous_owner_by_planet[planet.id] = planet.owner

        keep_after = local.step - max(1, self.proactive_defense_recent_capture_window)
        for planet_id, step in list(self.recently_captured_steps.items()):
            if step < keep_after:
                del self.recently_captured_steps[planet_id]
        keep_lost_after = local.step - max(1, self.recent_loss_recapture_window)
        for planet_id, step in list(self.recently_lost_steps.items()):
            if step < keep_lost_after:
                del self.recently_lost_steps[planet_id]
        keep_flip_after = local.step - max(1, self.contested_stop_loss_window)
        for planet_id, steps in list(self.planet_flip_steps.items()):
            kept = [step for step in steps if step >= keep_flip_after]
            if kept:
                self.planet_flip_steps[planet_id] = kept
            else:
                del self.planet_flip_steps[planet_id]

    def _update_third_party_tail_watchlist(self, local: LocalObs) -> None:
        if not self.enable_third_party_tail_watchlist:
            self.third_party_tail_watchlist.clear()
            return

        planets_by_id = {planet.id: planet for planet in local.planets}
        for key, entry in list(self.third_party_tail_watchlist.items()):
            target = planets_by_id.get(int(entry["target_id"]))
            if target is None:
                del self.third_party_tail_watchlist[key]
                continue
            expires_step = int(entry["expected_step"]) + self.third_party_tail_watchlist_post_window
            if local.step > expires_step or target.owner == local.player:
                del self.third_party_tail_watchlist[key]
                continue
            if target.owner not in (-1, int(entry["expected_owner"])):
                del self.third_party_tail_watchlist[key]

        for fleet in local.fleets:
            if fleet.owner in (-1, local.player) or fleet.ships <= 0:
                continue
            hit = self._third_party_fleet_first_capture_target(fleet, local)
            if hit is None:
                continue
            target, arrival, enemy_post_capture, direct_need = hit
            if arrival > self.third_party_tail_watchlist_horizon:
                continue
            if target.owner == local.player:
                continue
            if self.third_party_tail_only_neutral_targets and target.owner != -1:
                continue
            if target.production < self.third_party_tail_min_production:
                continue
            if target.owner == -1 and enemy_post_capture < self.third_party_tail_neutral_min_enemy_post_capture:
                continue

            key = (int(fleet.id), int(target.id))
            current = self.third_party_tail_watchlist.get(key)
            observed_step = int(current["observed_step"]) if current is not None else local.step
            self.third_party_tail_watchlist[key] = {
                "fleet_id": int(fleet.id),
                "target_id": int(target.id),
                "expected_owner": int(fleet.owner),
                "observed_step": observed_step,
                "last_seen_step": local.step,
                "expected_step": local.step + int(arrival),
                "enemy_post_capture": int(enemy_post_capture),
                "direct_need": int(direct_need),
                "target_owner": int(target.owner),
                "target_production": float(target.production),
            }

        if len(self.third_party_tail_watchlist) > self.third_party_tail_watchlist_max_entries:
            rows = sorted(
                self.third_party_tail_watchlist.items(),
                key=lambda item: (
                    float(item[1]["target_production"]),
                    int(item[1]["direct_need"]) - int(item[1]["enemy_post_capture"]),
                    -abs(local.step - int(item[1]["expected_step"])),
                ),
                reverse=True,
            )
            self.third_party_tail_watchlist = dict(rows[: self.third_party_tail_watchlist_max_entries])

    def _third_party_fleet_first_capture_target(
        self,
        fleet: object,
        local: LocalObs,
    ) -> tuple[Planet, int, int, int] | None:
        best: tuple[int, Planet] | None = None
        for target in local.planets:
            if target.owner == fleet.owner:
                continue
            arrival = self._fleet_arrival_to_target(fleet, target, local)
            if arrival is None:
                continue
            if best is None or arrival < best[0]:
                best = (arrival, target)
        if best is None:
            return None

        arrival, target = best
        target_defense = int(target.ships)
        if target.owner != -1:
            target_defense += int(target.production * arrival)
        if int(fleet.ships) <= target_defense:
            return None
        return target, int(arrival), int(fleet.ships) - target_defense, target_defense + 1

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
            if self._arrival_based_under_attack_availability_enabled(attack_row=under_attack[planet.id]):
                available -= self._incoming_reserve_required(planet, under_attack[planet.id])
            else:
                available -= sum(int(row["fleet"].ships) for row in under_attack[planet.id]["fleets"])
        return max(0, available)

    def _arrival_based_under_attack_availability_enabled(self, attack_row: dict[str, object]) -> bool:
        if not self.enable_arrival_based_under_attack_availability:
            return False
        return self.steps_seen - 1 >= self.under_attack_availability_min_step

    def _incoming_reserve_required(self, planet: Planet, attack_row: dict[str, object]) -> int:
        projected = int(planet.ships)
        previous_tick = 0
        lowest_margin = projected
        for row in sorted(attack_row["fleets"], key=lambda item: item["arrive_tick"]):
            arrive_tick = int(row["arrive_tick"])
            if arrive_tick > self.under_attack_availability_horizon:
                continue
            projected += int((arrive_tick - previous_tick) * planet.production)
            projected -= int(row["fleet"].ships)
            lowest_margin = min(lowest_margin, projected)
            previous_tick = arrive_tick
        return max(0, self.under_attack_availability_margin - lowest_margin)

    def _posture(self, local: LocalObs) -> str:
        if not self.enable_dynamic_posture:
            return "balanced"

        active_players = {
            planet.owner
            for planet in local.planets
            if planet.owner != -1
        } | {
            fleet.owner
            for fleet in local.fleets
            if fleet.owner != -1
        }
        if len(active_players) > self.posture_max_active_players:
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
            local.step >= self.posture_defensive_min_step
            and
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

    def _source_min_attack(self, source: Planet, local: LocalObs) -> int:
        active_players = self._active_player_count(local)
        if (
            self.enable_opening_high_prod_trickle
            and local.step <= self.opening_trickle_step_limit
            and source.production >= self.opening_trickle_source_min_production
            and active_players >= self.opening_trickle_min_active_players
            and active_players <= self.opening_trickle_max_active_players
        ):
            return max(1, self.opening_trickle_min_ships)
        return self._dynamic_min_attack(local)

    def _target_min_attack(self, source: Planet | None, target: Planet, local: LocalObs) -> int:
        if (
            source is not None
            and self.enable_opening_high_prod_trickle
            and local.step <= self.opening_trickle_step_limit
            and target.owner == -1
            and source.production >= self.opening_trickle_source_min_production
            and target.production >= self.opening_trickle_target_min_production
            and target.ships <= self.opening_trickle_max_target_ships
            and self.opening_trickle_min_active_players <= self._active_player_count(local) <= self.opening_trickle_max_active_players
        ):
            return max(1, self.opening_trickle_min_ships)
        return self._dynamic_min_attack(local)

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
        reserve = self._source_threat_reserve(planet, local)
        reserve = max(reserve, self._home_anchor_source_reserve(planet, local))
        reserve = max(reserve, self._midgame_border_source_reserve(planet, local))
        reserve = max(reserve, self._recent_capture_source_reserve(planet, local))
        if not self.enable_local_source_reserve:
            return reserve
        if local.step < self.local_reserve_min_step or local.step > self.local_reserve_max_step:
            return reserve

        enemy_planets = [p for p in local.planets if p.owner not in (-1, local.player)]
        nearest_enemy = min((distance(planet, enemy) for enemy in enemy_planets), default=10**9)
        is_high_prod = planet.production >= self.local_reserve_min_production
        is_front = nearest_enemy <= self.local_reserve_enemy_distance
        if not is_high_prod and not is_front:
            return reserve

        local_reserve = self.local_reserve_min_garrison + int(planet.production * self.local_reserve_turns)
        if is_front:
            front_pressure = max(0.0, self.local_reserve_enemy_distance - nearest_enemy) / max(self.local_reserve_enemy_distance, 1.0)
            local_reserve += int(self.local_reserve_front_bonus * front_pressure)
        return max(reserve, local_reserve)

    def _recent_capture_source_reserve(self, planet: Planet, local: LocalObs) -> int:
        if not self.enable_recent_capture_source_reserve:
            return 0
        if local.step < self.recent_capture_source_reserve_min_step or local.step > self.recent_capture_source_reserve_max_step:
            return 0
        if self.recent_capture_source_reserve_min_active_players > 0 and self._active_player_count(local) < self.recent_capture_source_reserve_min_active_players:
            return 0
        if planet.production < self.recent_capture_source_reserve_min_production:
            return 0

        captured_step = self.recently_captured_steps.get(planet.id)
        if captured_step is None or local.step - captured_step > self.recent_capture_source_reserve_window:
            return 0

        reserve = self.recent_capture_source_reserve_min_after + int(
            planet.production * self.recent_capture_source_reserve_prod_turns_after
        )
        if self.recent_capture_source_reserve_front_bonus > 0:
            enemy_planets = [enemy for enemy in local.planets if enemy.owner not in (-1, local.player)]
            nearest_enemy = min((distance(planet, enemy) for enemy in enemy_planets), default=10**9)
            if nearest_enemy <= self.recent_capture_source_reserve_enemy_radius:
                pressure = 1.0 - nearest_enemy / max(self.recent_capture_source_reserve_enemy_radius, 1.0)
                reserve += int(self.recent_capture_source_reserve_front_bonus * max(0.0, pressure))
        return max(0, reserve)

    def _home_anchor_source_reserve(self, planet: Planet, local: LocalObs) -> int:
        if not self.enable_home_anchor_source_reserve:
            return 0
        if local.step < self.home_anchor_step_min or local.step > self.home_anchor_step_max:
            return 0
        if self.home_anchor_min_active_players > 0 and self._active_player_count(local) < self.home_anchor_min_active_players:
            return 0
        if planet.production < self.home_anchor_min_production:
            return 0
        if not self._near_home_anchor(planet, self.home_anchor_home_radius):
            return 0

        reserve = self.home_anchor_min_after + int(planet.production * self.home_anchor_prod_turns_after)
        enemy_planets = [enemy for enemy in local.planets if enemy.owner not in (-1, local.player)]
        nearest_enemy = min((distance(planet, enemy) for enemy in enemy_planets), default=10**9)
        if nearest_enemy <= self.midgame_border_enemy_radius:
            pressure = 1.0 - nearest_enemy / max(self.midgame_border_enemy_radius, 1.0)
            reserve += int(self.home_anchor_front_threat_bonus * max(0.0, pressure))
        return max(0, reserve)

    def _midgame_border_source_reserve(self, planet: Planet, local: LocalObs) -> int:
        if not self.enable_midgame_border_source_reserve:
            return 0
        if local.step < self.midgame_border_step_min or local.step > self.midgame_border_step_max:
            return 0
        if self.midgame_border_min_active_players > 0 and self._active_player_count(local) < self.midgame_border_min_active_players:
            return 0
        if planet.production < self.midgame_border_min_production:
            return 0

        reserve = 0
        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            if distance(enemy, planet) > self.midgame_border_enemy_radius:
                continue
            sendable = self._project_enemy_sendable_for_source_filter(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            arrival = travel_ticks(enemy, planet, sendable)
            projected_need = sendable + self.midgame_border_threat_margin - int(planet.production * arrival)
            reserve = max(reserve, projected_need)

        baseline = self.midgame_border_min_after + int(planet.production * self.midgame_border_prod_turns_after)
        return max(0, reserve, baseline if reserve > 0 else 0)

    def _near_home_anchor(self, planet: Planet, radius: float) -> bool:
        if not self.home_anchor_positions:
            return False
        return any(math.hypot(planet.x - x, planet.y - y) <= radius for x, y in self.home_anchor_positions)

    def _source_threat_reserve(self, planet: Planet, local: LocalObs) -> int:
        if not self.enable_source_threat_reserve:
            return 0
        if local.step < self.source_threat_min_step or local.step > self.source_threat_max_step:
            return 0
        if planet.production < self.source_threat_min_production:
            return 0
        if self.source_threat_min_active_players > 0:
            active_players = {
                p.owner
                for p in local.planets
                if p.owner != -1
            } | {
                f.owner
                for f in local.fleets
                if f.owner != -1
            }
            if len(active_players) < self.source_threat_min_active_players:
                return 0

        reserve = 0
        remaining_steps = max(0, 500 - local.step)
        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            dist = distance(enemy, planet)
            if dist > self.source_threat_radius:
                continue
            sendable = int(enemy.ships * self.source_threat_enemy_send_fraction)
            sendable += int(enemy.production * self.source_threat_enemy_launch_window)
            sendable -= int(enemy.production * self.source_threat_enemy_reserve_turns)
            if sendable < self.min_ships_mine_attack:
                continue
            arrival = travel_ticks(enemy, planet, sendable)
            if arrival > self.source_threat_max_arrival:
                continue

            enemy_cost = sendable + 1
            capture_value = planet.production * max(0, remaining_steps - arrival)
            if capture_value < enemy_cost * self.source_threat_roi_multiplier:
                continue
            if capture_value - enemy_cost < self.source_threat_min_net_value:
                continue

            needed_now = sendable + self.source_threat_margin - int(planet.production * arrival)
            reserve = max(reserve, needed_now)
        return max(0, reserve)

    def _source_exposed_after_send(
        self,
        source: Planet,
        target: Planet,
        sent_ships: int,
        arrive_tick: int,
        local: LocalObs,
    ) -> bool:
        if self.enable_local_source_defense_gate and self._local_source_defense_gate_blocks(source, sent_ships, local):
            return True
        if self.enable_source_threat_send_filter:
            return self._source_threat_after_send_gap(source, target, sent_ships, arrive_tick, local) > 0.0
        return False

    def _local_source_defense_gate_blocks(self, source: Planet, sent_ships: int, local: LocalObs) -> bool:
        if local.step < self.local_source_defense_gate_min_step or local.step > self.local_source_defense_gate_max_step:
            return False
        if (
            self.local_source_defense_gate_min_active_players > 0
            and self._active_player_count(local) < self.local_source_defense_gate_min_active_players
        ):
            return False

        enemy_planets = [planet for planet in local.planets if planet.owner not in (-1, local.player)]
        if not enemy_planets:
            return False
        nearest_enemy = min(distance(source, enemy) for enemy in enemy_planets)
        is_high_prod = source.production >= self.local_source_defense_gate_min_production
        is_front = nearest_enemy <= self.local_source_defense_gate_front_distance
        if not is_high_prod and not is_front:
            return False

        source_after_send = max(0, int(source.ships) - int(sent_ships))
        for enemy in enemy_planets:
            sendable = self._project_enemy_sendable_for_local_gate(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            arrival = travel_ticks(enemy, source, sendable)
            if arrival > self.local_source_defense_gate_max_arrival:
                continue
            projected_source = source_after_send
            if self.local_source_defense_gate_use_arrival_production:
                projected_source += int(source.production * arrival)
            if projected_source < sendable + self.local_source_defense_gate_margin:
                return True
        return False

    def _project_enemy_sendable_for_local_gate(self, enemy: Planet) -> int:
        sendable = int(enemy.ships * self.local_source_defense_gate_enemy_fraction)
        sendable += int(enemy.production * self.local_source_defense_gate_enemy_launch_window)
        sendable -= int(enemy.production * self.local_source_defense_gate_enemy_reserve_turns)
        return max(0, sendable)

    def _capture_hold_adjusted_ships(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        arrive_tick: int,
        available: int,
        local: LocalObs,
    ) -> int | None:
        if not self.enable_capture_hold_margin_gate:
            return total_ships
        if target.owner == local.player or target.production < self.capture_hold_min_production:
            return total_ships
        if (
            self.capture_hold_min_active_players > 0
            and self._active_player_count(local) < self.capture_hold_min_active_players
        ):
            return total_ships

        post_capture = self._post_capture_ships(source, target, total_ships, local)

        extra_needed = 0
        for enemy in local.planets:
            if enemy.owner in (-1, local.player) or enemy.id == target.id:
                continue
            if distance(enemy, target) > self.capture_hold_enemy_radius:
                continue
            sendable = self._project_enemy_sendable_for_capture_hold(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            enemy_arrival = travel_ticks(enemy, target, sendable)
            if enemy_arrival > self.capture_hold_enemy_max_arrival:
                continue
            production_ticks = self._capture_hold_production_ticks(arrive_tick, enemy_arrival)
            projected_defense = post_capture + int(target.production * production_ticks)
            required_defense = sendable + self.capture_hold_margin
            if projected_defense < required_defense:
                extra_needed = max(extra_needed, required_defense - projected_defense)

        if extra_needed <= 0:
            return total_ships
        adjusted = total_ships + extra_needed
        if adjusted <= available and self.capture_hold_allow_extra_send:
            return adjusted
        return None

    def _project_enemy_sendable_for_capture_hold(self, enemy: Planet) -> int:
        sendable = int(enemy.ships * self.capture_hold_enemy_send_fraction)
        sendable += int(enemy.production * self.capture_hold_enemy_launch_window)
        sendable -= int(enemy.production * self.capture_hold_enemy_reserve_turns)
        return max(0, sendable)

    def _post_capture_ships(self, source: Planet, target: Planet, total_ships: int, local: LocalObs) -> int:
        post_capture = int(total_ships - target.ships)
        if target.owner != -1:
            post_capture -= self._enemy_production_buffer(source, target, local, total_ships)
        return max(0, post_capture)

    def _capture_hold_production_ticks(self, arrive_tick: int, enemy_arrival: int) -> int:
        if not self.capture_hold_use_post_capture_window:
            return enemy_arrival
        return max(0, enemy_arrival - arrive_tick)

    def _recent_loss_recapture_hold_adjusted_ships(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        arrive_tick: int,
        available: int,
        local: LocalObs,
    ) -> int | None:
        if not self.enable_recent_loss_recapture_hold_gate:
            return total_ships
        if target.owner in (-1, local.player) or target.production < self.recent_loss_recapture_hold_min_production:
            return total_ships
        if local.step < self.recent_loss_recapture_hold_min_step or local.step > self.recent_loss_recapture_hold_max_step:
            return total_ships
        if (
            self.recent_loss_recapture_hold_min_active_players > 0
            and self._active_player_count(local) < self.recent_loss_recapture_hold_min_active_players
        ):
            return total_ships
        lost_step = self.recently_lost_steps.get(target.id)
        if lost_step is None or local.step - lost_step > self.recent_loss_recapture_hold_window:
            return total_ships

        post_capture = self._post_capture_ships(source, target, total_ships, local)
        extra_needed = 0
        for enemy in local.planets:
            if enemy.owner in (-1, local.player) or enemy.id == target.id:
                continue
            if distance(enemy, target) > self.recent_loss_recapture_hold_enemy_radius:
                continue
            sendable = self._project_enemy_sendable_for_capture_hold(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            enemy_arrival = travel_ticks(enemy, target, sendable)
            if enemy_arrival > self.capture_hold_enemy_max_arrival:
                continue
            production_ticks = self._capture_hold_production_ticks(arrive_tick, enemy_arrival)
            projected_defense = post_capture + int(target.production * production_ticks)
            required_defense = sendable + self.recent_loss_recapture_hold_margin
            if projected_defense < required_defense:
                extra_needed = max(extra_needed, required_defense - projected_defense)

        if extra_needed <= 0:
            return total_ships
        adjusted = total_ships + extra_needed
        if adjusted <= available and self.recent_loss_recapture_hold_allow_extra_send:
            return adjusted
        return None

    def _opening_hold_adjusted_ships(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        arrive_tick: int,
        available: int,
        local: LocalObs,
    ) -> int | None:
        if not self.enable_opening_neutral_hold_margin:
            return total_ships
        if target.owner != -1 or local.step > self.opening_hold_step_limit:
            return total_ships
        if target.production < self.opening_hold_min_production:
            return total_ships
        active_players = self._active_player_count(local)
        if self.opening_hold_min_active_players > 0 and active_players < self.opening_hold_min_active_players:
            return total_ships
        if active_players > self.opening_hold_max_active_players:
            return total_ships

        post_capture = max(0, int(total_ships - target.ships))
        required = self.opening_hold_base_margin + int(target.production * self.opening_hold_prod_turns)
        enemy_eta = self._nearest_enemy_eta(target, local)
        if enemy_eta < 10**8 and enemy_eta - arrive_tick <= self.early_neutral_reaction_margin:
            required += self.opening_hold_contested_extra

        if post_capture >= required:
            return total_ships
        adjusted = total_ships + (required - post_capture)
        if adjusted <= available and self.opening_hold_allow_extra_send:
            return adjusted
        return None

    def _source_threat_target_penalty(self, source: Planet, target: Planet, local: LocalObs) -> float:
        if not self.enable_source_threat_target_penalty:
            return 0.0
        ships = self._base_ships_needed(target, local, source=source)
        if ships is None:
            return 0.0
        arrive_tick = self._estimate_arrival_for_requirement(source, target, ships, local)
        gap = self._source_threat_after_send_gap(source, target, ships, arrive_tick, local)
        return gap * self.source_threat_target_penalty_weight

    def _source_threat_after_send_gap(
        self,
        source: Planet,
        target: Planet,
        sent_ships: int,
        arrive_tick: int,
        local: LocalObs,
    ) -> float:
        if not self.enable_source_threat_send_filter and not self.enable_source_threat_target_penalty:
            return 0.0
        if local.step < self.source_threat_send_min_step or local.step > self.source_threat_send_max_step:
            return 0.0
        if source.production < self.source_threat_send_min_production:
            return 0.0
        active_players = self._active_player_count(local)
        if self.source_threat_send_min_active_players > 0 and active_players < self.source_threat_send_min_active_players:
            return 0.0
        if self.source_threat_send_max_active_players > 0 and active_players > self.source_threat_send_max_active_players:
            return 0.0

        source_remaining = max(0, int(source.ships) - int(sent_ships))
        enemy_best_value = 0.0
        remaining_steps = max(0, 500 - local.step)
        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            if distance(enemy, source) > self.source_threat_send_radius:
                continue
            sendable = self._project_enemy_sendable_for_source_filter(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            enemy_arrival = travel_ticks(enemy, source, sendable)
            if enemy_arrival > self.source_threat_send_max_arrival:
                continue

            projected_source = source_remaining + int(source.production * enemy_arrival)
            if projected_source >= sendable + self.source_threat_send_margin:
                continue

            enemy_cost = sendable + 1
            enemy_value = source.production * max(0, remaining_steps - enemy_arrival)
            if enemy_value < enemy_cost * self.source_threat_send_roi_multiplier:
                continue
            enemy_net = enemy_value - enemy_cost
            if enemy_net < self.source_threat_send_min_net_value:
                continue
            enemy_best_value = max(enemy_best_value, enemy_net)

        if enemy_best_value <= 0:
            return 0.0

        attack_value = target.production * max(0, 500 - local.step - arrive_tick)
        if target.owner not in (-1, local.player):
            attack_value *= 1.5
        attack_net = attack_value - sent_ships
        return max(0.0, enemy_best_value - attack_net * self.source_threat_send_trade_ratio)

    def _active_player_count(self, local: LocalObs) -> int:
        active_players = {
            p.owner
            for p in local.planets
            if p.owner != -1
        } | {
            f.owner
            for f in local.fleets
            if f.owner != -1
        }
        return len(active_players)

    def _project_enemy_sendable_for_source_filter(self, enemy: Planet) -> int:
        sendable = int(enemy.ships * self.source_threat_send_enemy_fraction)
        sendable += int(enemy.production * self.source_threat_send_enemy_launch_window)
        sendable -= int(enemy.production * self.source_threat_send_enemy_reserve_turns)
        return max(0, sendable)

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

    def _append_enemy_wave_preserve_prod(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if local.step < self.enemy_wave_preserve_min_step or local.step > self.enemy_wave_preserve_max_step:
            return
        active_players = self._active_player_count(local)
        if self.enemy_wave_preserve_min_active_players > 0 and active_players < self.enemy_wave_preserve_min_active_players:
            return
        if self.enemy_wave_preserve_max_active_players > 0 and active_players > self.enemy_wave_preserve_max_active_players:
            return

        planet_by_id = {planet.id: planet for planet in local.mine}
        rows: list[tuple[float, Planet, int, int]] = []
        for target_id, attack_row in under_attack.items():
            target = planet_by_id.get(target_id)
            if target is None or target.production < self.enemy_wave_preserve_min_production:
                continue
            need = self._enemy_wave_preserve_need(target, attack_row)
            if need is None:
                continue
            ships_needed, needed_by_tick, enemy_post_capture = need
            score = enemy_post_capture + target.production * 12.0 - max(0, needed_by_tick) * 0.2
            rows.append((score, target, ships_needed, needed_by_tick))

        rows.sort(key=lambda row: row[0], reverse=True)
        preserved = 0
        for _, target, ships_needed, needed_by_tick in rows:
            if preserved >= self.enemy_wave_preserve_max_targets:
                break
            remaining = ships_needed
            used_sources = 0
            for source, _ in closest_planets_to_target(local.mine, target):
                if used_sources >= self.enemy_wave_preserve_max_sources:
                    break
                if remaining < self.enemy_wave_preserve_min_send:
                    break
                if source.id == target.id or source.id in exhausted_planet_ids:
                    continue
                reserve = self.enemy_wave_preserve_source_min_after + int(
                    source.production * self.enemy_wave_preserve_source_prod_turns_after
                )
                available = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
                ships_to_send = min(
                    remaining,
                    self.enemy_wave_preserve_max_send,
                    max(0, available - reserve),
                )
                if ships_to_send < self.enemy_wave_preserve_min_send:
                    continue
                angle, arrive_tick = self._angle_and_arrival(source, target, ships_to_send, local)
                if angle is None or arrive_tick is None or arrive_tick > needed_by_tick:
                    continue
                if self.enable_sun_avoidance and sun_collision(source, ships_to_send, angle):
                    continue
                if not self._path_hits_target(source, target, ships_to_send, angle, arrive_tick, local):
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
                remaining -= ships_to_send
                used_sources += 1
            if used_sources > 0:
                preserved += 1

    def _enemy_wave_preserve_need(
        self,
        target: Planet,
        attack_row: dict[str, object],
    ) -> tuple[int, int, int] | None:
        attacking = sorted(
            [row for row in attack_row["fleets"] if int(row["arrive_tick"]) <= self.enemy_wave_preserve_horizon],
            key=lambda row: row["arrive_tick"],
        )
        if not attacking:
            return None

        incoming = sorted(
            [row for row in self.reinforcement_trajectories if row["target"].id == target.id],
            key=lambda row: row["arrive_tick"],
        )
        available = int(target.ships)
        previous_tick = 0
        reinf_idx = 0
        lowest_margin = 10**9
        low_tick = int(attacking[0]["arrive_tick"])
        for attack in attacking:
            arrive_tick = int(attack["arrive_tick"])
            available += int((arrive_tick - previous_tick) * target.production)
            while reinf_idx < len(incoming) and int(incoming[reinf_idx]["arrive_tick"]) <= arrive_tick:
                available += int(incoming[reinf_idx]["total_ships"])
                reinf_idx += 1
            available -= int(attack["fleet"].ships)
            previous_tick = arrive_tick
            if available < lowest_margin:
                lowest_margin = available
                low_tick = arrive_tick

        enemy_post_capture = max(0, -int(lowest_margin))
        if enemy_post_capture < self.enemy_wave_preserve_min_enemy_post_capture:
            return None
        ships_needed = enemy_post_capture + self.enemy_wave_preserve_margin
        return max(self.enemy_wave_preserve_min_send, ships_needed), max(1, low_tick), enemy_post_capture

    def _append_recent_high_prod_hub_support(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if local.step < self.hub_support_min_step or local.step > self.hub_support_max_step:
            return
        active_players = self._active_player_count(local)
        if self.hub_support_min_active_players > 0 and active_players < self.hub_support_min_active_players:
            return
        if self.hub_support_max_active_players > 0 and active_players > self.hub_support_max_active_players:
            return

        supported = 0
        enemy_planets = [p for p in local.planets if p.owner not in (-1, local.player)]
        target_rows: list[tuple[float, Planet, int]] = []
        for target in local.mine:
            captured_step = self.recently_captured_steps.get(target.id)
            if captured_step is None:
                continue
            age = local.step - captured_step
            if age < 0 or age > self.hub_support_recent_capture_window:
                continue
            if target.production < self.hub_support_min_production:
                continue
            if target.id in under_attack:
                continue
            if any(row["target"].id == target.id and row["arrive_tick"] >= 0 for row in self.reinforcement_trajectories):
                continue

            nearest_enemy = min((distance(target, enemy) for enemy in enemy_planets), default=999.0)
            if nearest_enemy > self.hub_support_enemy_radius:
                continue
            incoming_support = sum(
                int(row["total_ships"])
                for row in self.reinforcement_trajectories
                if row["target"].id == target.id and int(row["arrive_tick"]) <= self.hub_support_max_eta
            )
            desired = self.hub_support_base_margin + int(target.production * self.hub_support_prod_turns)
            if nearest_enemy <= self.hub_support_enemy_radius:
                desired += self.hub_support_front_bonus
            deficit = desired - int(target.ships) - incoming_support
            if deficit < self.hub_support_min_send:
                continue
            score = target.production * 12.0 + max(0.0, self.hub_support_enemy_radius - nearest_enemy) * 0.2
            score += max(0, self.hub_support_recent_capture_window - age) * 0.25
            score += deficit * 0.5
            target_rows.append((score, target, deficit))

        target_rows.sort(key=lambda row: row[0], reverse=True)
        for _, target, deficit in target_rows:
            if supported >= self.hub_support_max_targets:
                break
            for source, _ in closest_planets_to_target(local.mine, target):
                if source.id == target.id or source.id in exhausted_planet_ids:
                    continue
                captured_step = self.recently_captured_steps.get(source.id)
                if captured_step is not None and local.step - captured_step <= self.hub_support_recent_capture_window:
                    continue
                reserve = self.hub_support_source_min_after + int(
                    source.production * self.hub_support_source_prod_turns_after
                )
                available = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
                ships_to_send = min(deficit, self.hub_support_max_send, max(0, available - reserve))
                if ships_to_send < self.hub_support_min_send:
                    continue
                angle, arrive_tick = self._angle_and_arrival(source, target, ships_to_send, local)
                if angle is None or arrive_tick is None or arrive_tick > self.hub_support_max_eta:
                    continue
                if self.enable_sun_avoidance and sun_collision(source, ships_to_send, angle):
                    continue
                if not self._path_hits_target(source, target, ships_to_send, angle, arrive_tick, local):
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
                supported += 1
                break

    def _append_comet_evacuation(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        expiring_comets = [
            planet
            for planet in local.mine
            if planet.id in local.comet_planet_ids
            and planet.id not in exhausted_planet_ids
            and 0 < self.comet_remaining_by_planet.get(planet.id, 10**9) <= self.comet_evacuation_remaining_turns
        ]
        expiring_comets.sort(key=lambda planet: self.comet_remaining_by_planet.get(planet.id, 10**9))

        for source in expiring_comets:
            ships = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
            if ships < self.comet_evacuation_min_ships:
                continue

            target = self._best_comet_evacuation_target(source, ships, local)
            if target is None:
                continue
            angle, arrive_tick = self._angle_and_arrival(source, target, ships, local)
            if angle is None or arrive_tick is None:
                continue
            if self.enable_sun_avoidance and sun_collision(source, ships, angle):
                continue
            if not self._path_hits_target(source, target, ships, angle, arrive_tick, local):
                continue

            moves.append([source.id, angle, ships])
            exhausted_planet_ids.add(source.id)
            if target.owner == local.player:
                self.reinforcement_trajectories.append(
                    {
                        "source_id": source.id,
                        "target": target,
                        "angle": angle,
                        "total_ships": ships,
                        "arrive_tick": arrive_tick,
                    }
                )
            else:
                self._track_attack(source, target, angle, ships, arrive_tick)

    def _best_comet_evacuation_target(self, source: Planet, ships: int, local: LocalObs) -> Planet | None:
        scored: list[tuple[float, Planet]] = []
        for target in local.planets:
            if target.id == source.id:
                continue
            angle, arrive_tick = self._angle_and_arrival(source, target, ships, local)
            if angle is None or arrive_tick is None:
                continue

            if target.owner == local.player:
                score = self._comet_evacuation_own_target_score(source, target, local, arrive_tick)
                scored.append((score, target))
                continue

            needed = self._base_ships_needed(target, local, source=source)
            if needed is None or ships < needed:
                continue
            remaining_value = target.production * max(0, 500 - local.step - arrive_tick)
            if target.owner not in (-1, local.player):
                remaining_value *= 1.5
            if remaining_value < ships * self.comet_evacuation_target_roi:
                continue
            score = (
                self._target_score(source, target, local)
                + target.production * self.comet_evacuation_target_prod_weight
                + (self.comet_evacuation_target_enemy_bonus if target.owner not in (-1, local.player) else 0.0)
                - arrive_tick
            )
            scored.append((score, target))

        if not scored:
            return None
        scored.sort(key=lambda row: row[0], reverse=True)
        return scored[0][1]

    def _append_endgame_fleet_dump(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if local.step < self.endgame_dump_min_step:
            return
        if self.endgame_dump_min_active_players > 0 and self._active_player_count(local) < self.endgame_dump_min_active_players:
            return

        remaining_ticks = max(1, 500 - local.step)
        for source in sorted(local.mine, key=lambda planet: planet.ships, reverse=True):
            if source.id in exhausted_planet_ids:
                continue
            ships = self._available_ships(source, under_attack, reserve_outgoing_reinforcements=True)
            ships -= max(0, self.endgame_dump_keep_source_ships)
            if ships < self.endgame_dump_min_ships:
                continue
            angle = self._safe_endgame_dump_angle(source, ships, remaining_ticks, local)
            if angle is None:
                continue
            moves.append([source.id, angle, ships])
            exhausted_planet_ids.add(source.id)

    def _safe_endgame_dump_angle(
        self,
        source: Planet,
        ships: int,
        remaining_ticks: int,
        local: LocalObs,
    ) -> float | None:
        samples = max(8, self.endgame_dump_angle_samples)
        start_angle = 0.0
        enemies = [planet for planet in local.planets if planet.owner not in (-1, local.player)]
        if enemies:
            nearest = min(enemies, key=lambda planet: distance(source, planet))
            start_angle = angle_to(nearest, source)

        for idx in range(samples):
            angle = start_angle + (2.0 * math.pi * idx / samples)
            if self.enable_sun_avoidance and sun_collision(source, ships, angle, ticks=remaining_ticks + 1):
                continue
            if self._first_planet_hit(source, angle, ships, remaining_ticks, local) is None:
                return angle
        return None

    def _comet_evacuation_own_target_score(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        arrive_tick: int,
    ) -> float:
        enemy_planets = [planet for planet in local.planets if planet.owner not in (-1, local.player)]
        nearest_enemy = min((distance(target, enemy) for enemy in enemy_planets), default=10**9)
        front_bonus = 0.0
        if nearest_enemy <= self.comet_evacuation_front_distance:
            pressure = 1.0 - nearest_enemy / max(self.comet_evacuation_front_distance, 1.0)
            front_bonus = self.comet_evacuation_front_bonus * pressure
        return target.production * self.comet_evacuation_own_prod_weight + front_bonus - arrive_tick
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
        horizon, max_send, min_margin = self._effective_value_defense_knobs(local)
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

            pressure = self._project_defense_pressure(target, row, local, horizon, min_margin)
            if pressure is None:
                continue
            ships_needed, needed_by_tick = pressure
            ships_needed = min(ships_needed, max_send)
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

    def _effective_value_defense_knobs(self, local: LocalObs) -> tuple[int, int, int]:
        horizon = self.value_defense_horizon
        max_send = self.value_defense_max_send
        min_margin = self.value_defense_min_margin
        if self.value_defense_multiplayer_min_active_players > 0:
            active_players = {
                planet.owner
                for planet in local.planets
                if planet.owner != -1
            } | {
                fleet.owner
                for fleet in local.fleets
                if fleet.owner != -1
            }
            if len(active_players) >= self.value_defense_multiplayer_min_active_players:
                horizon = self.value_defense_multiplayer_horizon or horizon
                max_send = self.value_defense_multiplayer_max_send or max_send
                min_margin = self.value_defense_multiplayer_min_margin or min_margin
        return horizon, max_send, min_margin

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
        active_players = self._active_player_count(local)
        if active_players < self.proactive_defense_min_active_players:
            return False
        if active_players > self.proactive_defense_max_active_players:
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
        horizon: int,
        min_margin: int,
    ) -> tuple[int, int] | None:
        attacking = sorted(
            [row for row in attack_row["fleets"] if int(row["arrive_tick"]) <= horizon],
            key=lambda row: row["arrive_tick"],
        )
        if not attacking:
            return None

        incoming = sorted(
            [row for row in self.reinforcement_trajectories if row["target"].id == target.id],
            key=lambda row: row["arrive_tick"],
        )
        desired_margin = min_margin + int(target.production * self.value_defense_buffer_turns)
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
        if self.enable_global_attack_priority:
            self._append_global_priority_attacks(local, under_attack, exhausted_planet_ids, moves, launch_pressure)
            return

        for source in sorted(
            local.mine,
            key=lambda p: (
                self._recent_capture_chain_source_priority(p, local)
                + self._mobile_relay_source_priority(p, local),
                p.ships,
            ),
            reverse=True,
        ):
            if source.id in exhausted_planet_ids:
                continue
            if self._available_local_attack_ships(source, local, under_attack) < self._source_min_attack(source, local):
                continue

            candidate_targets = [
                target
                for target in local.targets
                if (
                    not self.skip_comet_targets
                    or target.id not in local.comet_planet_ids
                    or self._mobile_relay_candidate_allowed(source, target, local)
                )
            ]
            candidate_targets.sort(
                key=lambda target: (
                    self._target_score(source, target, local)
                    + self._recent_capture_chain_target_bonus(source, target, local)
                    + self._mobile_relay_score(source, target, local)
                    + self._mobile_relay_source_target_bonus(source, target, local)
                    + launch_pressure.get(target.id, 0.0)
                    - self._source_threat_target_penalty(source, target, local)
                ),
                reverse=True,
            )
            candidate_targets = self._inject_tail_capture_candidates(source, candidate_targets, local)

            for target in candidate_targets[: self._dynamic_target_candidate_limit(local)]:
                if self.enable_single_attacks and self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break
                if self.enable_coop_attacks and self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    break

    def _append_global_priority_attacks(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
        launch_pressure: dict[int, float],
    ) -> None:
        failed_pairs: set[tuple[int, int]] = set()
        while len(failed_pairs) < self.global_attack_max_failed_pairs:
            best: tuple[float, Planet, Planet] | None = None
            for source in local.mine:
                if source.id in exhausted_planet_ids:
                    continue
                if self._available_local_attack_ships(source, local, under_attack) < self._source_min_attack(source, local):
                    continue
                for target in local.targets:
                    if self.skip_comet_targets and target.id in local.comet_planet_ids:
                        if not self._mobile_relay_candidate_allowed(source, target, local):
                            continue
                    if (source.id, target.id) in failed_pairs:
                        continue
                    score = self._global_attack_score(source, target, local, launch_pressure)
                    score += self._recent_capture_chain_target_bonus(source, target, local)
                    score += self._mobile_relay_score(source, target, local)
                    score += self._mobile_relay_source_target_bonus(source, target, local)
                    score -= self._source_threat_target_penalty(source, target, local)
                    if best is None or score > best[0]:
                        best = (score, source, target)

            if best is None:
                return

            _, source, target = best
            before_moves = len(moves)
            if self.enable_single_attacks and self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                continue
            if self.enable_coop_attacks and self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                continue

            if len(moves) == before_moves:
                failed_pairs.add((source.id, target.id))

    def _append_no_attack_fallback(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if not self.enable_no_attack_fallback:
            return
        if local.step < self.no_attack_fallback_min_step or local.step > self.no_attack_fallback_max_step:
            return
        active_players = self._active_player_count(local)
        if active_players < self.no_attack_fallback_min_active_players:
            return
        if self.no_attack_fallback_max_active_players > 0 and active_players > self.no_attack_fallback_max_active_players:
            return

        launch_pressure = self._enemy_launch_pressure(local)
        original_min_attack = self.min_ships_mine_attack
        self.min_ships_mine_attack = max(1, original_min_attack + self.no_attack_fallback_min_attack_delta)
        try:
            for source in sorted(
                local.mine,
                key=lambda p: (
                    self._recent_capture_chain_source_priority(p, local)
                    + self._mobile_relay_source_priority(p, local),
                    p.ships,
                ),
                reverse=True,
            ):
                if source.id in exhausted_planet_ids:
                    continue
                if self._available_local_attack_ships(source, local, under_attack) < self._source_min_attack(source, local):
                    continue

                candidate_targets = [
                    target
                    for target in local.targets
                    if (
                        not self.skip_comet_targets
                        or target.id not in local.comet_planet_ids
                        or self._mobile_relay_candidate_allowed(source, target, local)
                    )
                ]
                candidate_targets.sort(
                    key=lambda target: (
                        self._target_score(source, target, local)
                        + self._recent_capture_chain_target_bonus(source, target, local)
                        + self._mobile_relay_score(source, target, local)
                        + self._mobile_relay_source_target_bonus(source, target, local)
                        + launch_pressure.get(target.id, 0.0)
                        - self._source_threat_target_penalty(source, target, local)
                    ),
                    reverse=True,
                )
                candidate_targets = self._inject_tail_capture_candidates(source, candidate_targets, local)

                for target in candidate_targets[: self.no_attack_fallback_candidate_limit]:
                    if self.enable_single_attacks and self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                        return
                    if self.enable_coop_attacks and self._try_coop_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                        return
        finally:
            self.min_ships_mine_attack = original_min_attack

    def _append_opening_tempo_neutral_fallback(
        self,
        local: LocalObs,
        under_attack: dict[int, dict[str, object]],
        exhausted_planet_ids: set[int],
        moves: list[list[float | int]],
    ) -> None:
        if not self.enable_opening_tempo_neutral_fallback:
            return
        if local.step > self.opening_tempo_step_limit:
            return
        active_players = self._active_player_count(local)
        if active_players < self.opening_tempo_min_active_players:
            return
        if self.opening_tempo_max_active_players > 0 and active_players > self.opening_tempo_max_active_players:
            return

        candidates = [
            target
            for target in local.targets
            if target.owner == -1
            and target.production >= self.opening_tempo_min_production
            and target.ships <= self.opening_tempo_max_target_ships
            and (not self.skip_comet_targets or target.id not in local.comet_planet_ids)
        ]
        if not candidates:
            return

        for source in sorted(local.mine, key=lambda p: (p.production, p.ships), reverse=True):
            if source.id in exhausted_planet_ids:
                continue
            if source.production < self.opening_tempo_source_min_production:
                continue
            available = self._available_local_attack_ships(source, local, under_attack)
            if available <= self.opening_tempo_source_min_after:
                continue

            scored: list[tuple[float, Planet]] = []
            for target in candidates:
                ships_needed = self._base_ships_needed(target, local, source=source)
                if ships_needed is None:
                    continue
                if available - ships_needed < self.opening_tempo_source_min_after:
                    continue
                arrival = self._estimate_arrival_for_requirement(source, target, ships_needed, local)
                if arrival > self.opening_tempo_max_eta:
                    continue
                score = (target.production * 40.0) / max(1.0, target.ships + 0.35 * arrival)
                score += self._target_score(source, target, local) * 0.01
                scored.append((score, target))

            scored.sort(key=lambda item: item[0], reverse=True)
            for _, target in scored[: self.opening_tempo_candidate_limit]:
                if self._try_single_attack(source, target, local, under_attack, exhausted_planet_ids, moves):
                    return

    def _inject_tail_capture_candidates(
        self,
        source: Planet,
        candidate_targets: list[Planet],
        local: LocalObs,
    ) -> list[Planet]:
        if not self.enable_third_party_tail_candidate_injection:
            return candidate_targets
        if self.third_party_tail_candidate_limit <= 0:
            return candidate_targets

        tail_candidates: list[tuple[float, Planet]] = []
        for target in candidate_targets:
            plan = self._third_party_tail_capture_plan(source, target, local)
            if plan is None:
                continue
            score = float(plan["score"])
            if score < self.third_party_tail_candidate_min_score:
                continue
            tail_candidates.append((score, target))
        if not tail_candidates:
            return candidate_targets

        tail_candidates.sort(key=lambda item: item[0], reverse=True)
        injected = [target for _, target in tail_candidates[: self.third_party_tail_candidate_limit]]
        injected_ids = {target.id for target in injected}
        keep_front = max(0, self.third_party_tail_candidate_keep_front)
        front = candidate_targets[:keep_front]
        front_ids = {target.id for target in front}
        injected = [target for target in injected if target.id not in front_ids]
        injected_ids = {target.id for target in injected}
        rest = [
            target
            for target in candidate_targets[keep_front:]
            if target.id not in injected_ids and target.id not in front_ids
        ]
        return front + injected + rest

    def _recent_capture_chain_active(self, local: LocalObs) -> bool:
        if not self.enable_recent_capture_chain_attack:
            return False
        if local.step < self.chain_attack_min_step or local.step > self.chain_attack_max_step:
            return False
        active_players = self._active_player_count(local)
        if active_players < self.chain_attack_min_active_players:
            return False
        if self.chain_attack_max_active_players > 0 and active_players > self.chain_attack_max_active_players:
            return False
        return True

    def _recent_capture_chain_source_age(self, source: Planet, local: LocalObs) -> int | None:
        if not self._recent_capture_chain_active(local):
            return None
        if source.production < self.chain_attack_source_min_production:
            return None
        captured_step = self.recently_captured_steps.get(source.id)
        if captured_step is None:
            return None
        age = local.step - captured_step
        if age < 0 or age > self.chain_attack_source_window:
            return None
        return age

    def _recent_capture_chain_source_priority(self, source: Planet, local: LocalObs) -> float:
        age = self._recent_capture_chain_source_age(source, local)
        if age is None:
            return 0.0
        return self.chain_attack_source_order_bonus + max(0, self.chain_attack_source_window - age)

    def _recent_capture_chain_target_bonus(self, source: Planet, target: Planet, local: LocalObs) -> float:
        age = self._recent_capture_chain_source_age(source, local)
        if age is None:
            return 0.0
        if target.owner == local.player or target.production < self.chain_attack_target_min_production:
            return 0.0
        base_ships = self._base_ships_needed(target, local, source=source)
        if base_ships is None:
            return 0.0
        arrive_tick = self._estimate_arrival_for_requirement(source, target, base_ships, local)
        if arrive_tick > self.chain_attack_max_eta:
            return 0.0
        bonus = self.chain_attack_neutral_bonus if target.owner == -1 else self.chain_attack_enemy_bonus
        urgency = max(0.0, 1.0 - age / max(1, self.chain_attack_source_window))
        eta_relief = max(0.0, 1.0 - arrive_tick / max(1, self.chain_attack_max_eta))
        return bonus * (1.0 + 0.25 * urgency + 0.25 * eta_relief)

    def _mobile_relay_active(self, local: LocalObs) -> bool:
        if not self.enable_mobile_relay_attack:
            return False
        if local.step < self.mobile_relay_min_step or local.step > self.mobile_relay_max_step:
            return False
        active_players = self._active_player_count(local)
        if active_players < self.mobile_relay_min_active_players:
            return False
        if self.mobile_relay_max_active_players > 0 and active_players > self.mobile_relay_max_active_players:
            return False
        return True

    def _mobile_relay_planet(self, planet: Planet, local: LocalObs) -> bool:
        if planet.id not in self.moving_planets and planet.id not in local.comet_planet_ids:
            return False
        if planet.id in local.comet_planet_ids:
            remaining = self.comet_remaining_by_planet.get(planet.id, 0)
            if remaining < self.mobile_relay_comet_min_remaining:
                return False
        return True

    def _mobile_relay_candidate_allowed(self, source: Planet, target: Planet, local: LocalObs) -> bool:
        return self._mobile_relay_score(source, target, local) > 0.0

    def _mobile_relay_score(self, source: Planet, relay: Planet, local: LocalObs) -> float:
        if not self._mobile_relay_active(local):
            return 0.0
        if relay.owner == local.player:
            return 0.0
        if relay.production < self.mobile_relay_min_production or relay.ships > self.mobile_relay_max_ships:
            return 0.0
        if not self._mobile_relay_planet(relay, local):
            return 0.0
        relay_needed = self._base_ships_needed(relay, local, source=source)
        if relay_needed is None:
            return 0.0
        first_eta = self._estimate_arrival_for_requirement(source, relay, relay_needed, local)
        if first_eta > self.mobile_relay_max_first_eta:
            return 0.0

        best_score = 0.0
        for goal in local.targets:
            if goal.id == relay.id or goal.production < self.mobile_relay_goal_min_production:
                continue
            if goal.id in local.comet_planet_ids:
                continue
            goal_needed = self._base_ships_needed(goal, local, source=source)
            if goal_needed is None:
                continue
            direct_eta = self._estimate_arrival_for_requirement(source, goal, goal_needed, local)
            if direct_eta < self.mobile_relay_direct_min_eta:
                continue
            second_eta = self._estimate_arrival_for_requirement(relay, goal, goal_needed, local)
            if second_eta > self.mobile_relay_max_second_eta:
                continue
            savings = direct_eta - (first_eta + second_eta)
            if savings < self.mobile_relay_min_eta_savings:
                continue
            owner_bonus = 12.0 if goal.owner not in (-1, local.player) else 0.0
            score = (
                self.mobile_relay_bonus
                + goal.production * 6.0
                + relay.production * 4.0
                + savings
                + owner_bonus
                - relay.ships * 0.5
            )
            best_score = max(best_score, score)
        return best_score

    def _mobile_relay_source_age(self, source: Planet, local: LocalObs) -> int | None:
        if not self._mobile_relay_active(local):
            return None
        if not self._mobile_relay_planet(source, local):
            return None
        captured_step = self.recently_captured_steps.get(source.id)
        if captured_step is None:
            return None
        age = local.step - captured_step
        if age < 0 or age > self.mobile_relay_recent_source_window:
            return None
        return age

    def _mobile_relay_source_priority(self, source: Planet, local: LocalObs) -> float:
        age = self._mobile_relay_source_age(source, local)
        if age is None:
            return 0.0
        return self.mobile_relay_source_order_bonus + max(0, self.mobile_relay_recent_source_window - age)

    def _mobile_relay_source_target_bonus(self, source: Planet, target: Planet, local: LocalObs) -> float:
        age = self._mobile_relay_source_age(source, local)
        if age is None:
            return 0.0
        if target.owner == local.player or target.production < self.mobile_relay_goal_min_production:
            return 0.0
        ships_needed = self._base_ships_needed(target, local, source=source)
        if ships_needed is None:
            return 0.0
        eta = self._estimate_arrival_for_requirement(source, target, ships_needed, local)
        if eta > self.mobile_relay_max_second_eta:
            return 0.0
        urgency = max(0.0, 1.0 - age / max(1, self.mobile_relay_recent_source_window))
        return self.mobile_relay_source_target_bonus * (1.0 + 0.25 * urgency)

    def _global_attack_score(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        launch_pressure: dict[int, float],
    ) -> float:
        base_ships = self._base_ships_needed(target, local, source=source)
        if base_ships is None:
            return -1e12
        arrival = self._estimate_arrival_for_requirement(source, target, base_ships, local)
        score = self._target_score(source, target, local) + launch_pressure.get(target.id, 0.0)
        if self.global_attack_roi_weight:
            remaining_value = target.production * max(0, 500 - local.step - arrival)
            if target.owner not in (-1, local.player):
                remaining_value *= 1.6
            score += self.global_attack_roi_weight * remaining_value / max(1, base_ships)
        if self.global_attack_arrival_penalty:
            score -= self.global_attack_arrival_penalty * arrival
        return score

    def _enemy_launch_pressure(self, local: LocalObs) -> dict[int, float]:
        if not self.enable_enemy_launch_punish:
            return {}
        if local.step < self.enemy_launch_punish_min_step or local.step > self.enemy_launch_punish_max_step:
            return {}

        own_prod = sum(p.production for p in local.planets if p.owner == local.player)
        enemy_prod = sum(p.production for p in local.planets if p.owner not in (-1, local.player))
        own_planets = sum(1 for p in local.planets if p.owner == local.player)
        enemy_planets = sum(1 for p in local.planets if p.owner not in (-1, local.player))
        own_ships = sum(p.ships for p in local.planets if p.owner == local.player)
        enemy_ships = sum(p.ships for p in local.planets if p.owner not in (-1, local.player))
        own_ships += sum(f.ships for f in local.fleets if f.owner == local.player)
        enemy_ships += sum(f.ships for f in local.fleets if f.owner not in (-1, local.player))
        if own_prod - enemy_prod < self.enemy_launch_punish_min_prod_diff:
            return {}
        if own_planets - enemy_planets < self.enemy_launch_punish_min_planet_diff:
            return {}
        if own_ships / max(enemy_ships, 1) < self.enemy_launch_punish_min_ship_ratio:
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

        if source is not None:
            tail_plan = self._third_party_tail_capture_plan(source, target, local)
            if tail_plan is not None:
                tail_needed = int(tail_plan["ships"])
                if en_route >= tail_needed:
                    return None
                return max(1, tail_needed - en_route)

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
        return max(self._target_min_attack(source, target, local), needed_now - en_route)

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

    def _third_party_tail_capture_plan(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
    ) -> dict[str, float] | None:
        live_plan = self._live_third_party_tail_capture_plan(source, target, local)
        watch_plan = self._watchlist_third_party_tail_capture_plan(source, target, local)
        if live_plan is None:
            return watch_plan
        if watch_plan is None:
            return live_plan
        return watch_plan if watch_plan["score"] > live_plan["score"] else live_plan

    def _live_third_party_tail_capture_plan(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
    ) -> dict[str, float] | None:
        if not self.enable_third_party_tail_capture:
            return None
        if target.owner == local.player:
            return None
        if self.third_party_tail_only_neutral_targets and target.owner != -1:
            return None
        if target.production < self.third_party_tail_min_production:
            return None
        if local.step < self.third_party_tail_min_step or local.step > self.third_party_tail_max_step:
            return None
        if (
            self.third_party_tail_min_active_players > 0
            and self._active_player_count(local) < self.third_party_tail_min_active_players
        ):
            return None

        best: dict[str, float] | None = None
        for fleet in local.fleets:
            if fleet.owner in (-1, local.player) or fleet.ships <= 0:
                continue
            if target.owner == fleet.owner:
                continue
            enemy_arrival = self._fleet_arrival_to_target(fleet, target, local)
            if enemy_arrival is None or enemy_arrival > self.third_party_tail_max_enemy_arrival:
                continue

            target_defense = int(target.ships)
            if target.owner != -1:
                target_defense += int(target.production * enemy_arrival)
            if fleet.ships <= target_defense:
                continue

            enemy_post_capture = int(fleet.ships - target_defense)
            if (
                target.owner == -1
                and enemy_post_capture < self.third_party_tail_neutral_min_enemy_post_capture
            ):
                continue
            plan = self._tail_capture_ships_for_enemy_capture(
                source,
                target,
                local,
                enemy_arrival,
                enemy_post_capture,
            )
            if plan is None:
                continue
            plan["enemy_arrival"] = float(enemy_arrival)
            plan["enemy_post_capture"] = float(enemy_post_capture)
            plan["direct_need"] = float(max(1, target_defense + 1))
            if best is None or plan["score"] > best["score"]:
                best = plan
        return best

    def _watchlist_third_party_tail_capture_plan(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
    ) -> dict[str, float] | None:
        if not (self.enable_third_party_tail_capture and self.enable_third_party_tail_watchlist):
            return None
        if target.owner == local.player:
            return None
        if self.third_party_tail_only_neutral_targets and target.owner != -1:
            return None
        if target.production < self.third_party_tail_min_production:
            return None
        if local.step < self.third_party_tail_min_step or local.step > self.third_party_tail_max_step:
            return None
        if (
            self.third_party_tail_min_active_players > 0
            and self._active_player_count(local) < self.third_party_tail_min_active_players
        ):
            return None

        best: dict[str, float] | None = None
        for entry in self.third_party_tail_watchlist.values():
            if int(entry["target_id"]) != target.id:
                continue
            if local.step - int(entry["observed_step"]) < self.third_party_tail_watchlist_min_recheck_age:
                continue
            expected_step = int(entry["expected_step"])
            if local.step > expected_step + self.third_party_tail_watchlist_post_window:
                continue

            expected_owner = int(entry["expected_owner"])
            if target.owner not in (-1, expected_owner):
                continue

            if target.owner == expected_owner:
                enemy_arrival = 0
                enemy_post_capture = int(target.ships)
                direct_need = int(target.ships + 1)
                if target.owner != -1:
                    direct_need += self._enemy_production_buffer(source, target, local, direct_need)
            else:
                enemy_arrival = max(0, expected_step - local.step)
                if enemy_arrival > self.third_party_tail_max_enemy_arrival:
                    continue
                enemy_post_capture = int(entry["enemy_post_capture"])
                direct_need = int(entry["direct_need"])

            if target.owner == -1 and enemy_post_capture < self.third_party_tail_neutral_min_enemy_post_capture:
                continue

            plan = self._tail_capture_ships_for_enemy_capture(
                source,
                target,
                local,
                enemy_arrival,
                enemy_post_capture,
                direct_need_override=direct_need,
            )
            if plan is None:
                continue
            plan["enemy_arrival"] = float(enemy_arrival)
            plan["enemy_post_capture"] = float(enemy_post_capture)
            plan["direct_need"] = float(max(1, direct_need))
            plan["score"] = float(plan["score"] + self.third_party_tail_watchlist_score_bonus)
            plan["watchlist"] = 1.0
            if best is None or plan["score"] > best["score"]:
                best = plan
        return best

    def _fleet_arrival_to_target(self, fleet: object, target: Planet, local: LocalObs) -> int | None:
        speed = fleet_speed(int(fleet.ships))
        if speed <= 0:
            return None
        target_traj = planet_trajectory(target, local.angular_velocity) if target.id in self.moving_planets else None
        prev_x, prev_y = float(fleet.x), float(fleet.y)
        max_tick = self.third_party_tail_max_enemy_arrival
        if self.enable_third_party_tail_watchlist:
            max_tick = max(max_tick, self.third_party_tail_watchlist_horizon)
        if target_traj is not None:
            max_tick = min(max_tick, len(target_traj))
        for tick in range(1, max_tick + 1):
            next_x = float(fleet.x) + math.cos(float(fleet.angle)) * speed * tick
            next_y = float(fleet.y) + math.sin(float(fleet.angle)) * speed * tick
            tx, ty = target_traj[tick - 1] if target_traj is not None else (target.x, target.y)
            if collides_segment_circle(prev_x, prev_y, next_x, next_y, tx, ty, target.radius):
                return tick
            prev_x, prev_y = next_x, next_y
        return None

    def _tail_capture_ships_for_enemy_capture(
        self,
        source: Planet,
        target: Planet,
        local: LocalObs,
        enemy_arrival: int,
        enemy_post_capture: int,
        direct_need_override: int | None = None,
    ) -> dict[str, float] | None:
        if direct_need_override is None:
            direct_need = target.ships + 1
            if target.owner != -1:
                direct_need += int(target.production * enemy_arrival)
        else:
            direct_need = max(1, int(direct_need_override))

        min_ships = max(1, int(enemy_post_capture) + 1 + self.third_party_tail_margin)
        min_ships = max(min_ships, self.third_party_tail_min_send)
        max_ships = min(int(self.third_party_tail_max_ships), int(source.ships - self.third_party_tail_source_min_after))
        if max_ships < min_ships:
            return None

        best: dict[str, float] | None = None
        for ships in range(min_ships, max_ships + 1):
            arrival = self._estimate_arrival_for_requirement(source, target, ships, local)
            delay = arrival - enemy_arrival
            if delay < self.third_party_tail_min_delay:
                continue
            if delay > self.third_party_tail_max_delay:
                continue
            if target.owner == -1 and arrival > self.third_party_tail_neutral_max_arrival:
                continue

            produced_after_enemy_capture = int(target.production * max(0, delay))
            required = int(enemy_post_capture + produced_after_enemy_capture + 1 + self.third_party_tail_margin)
            required = max(required, self.third_party_tail_min_send)
            if ships < required:
                continue

            remaining_value = target.production * max(0, 500 - local.step - arrival)
            if remaining_value < ships * self.third_party_tail_roi_multiplier:
                continue
            if remaining_value - ships < self.third_party_tail_min_net_value:
                continue

            savings = max(0, direct_need - ships)
            if savings < self.third_party_tail_min_savings:
                continue
            if direct_need > 0 and savings / max(1, direct_need) < self.third_party_tail_min_savings_ratio:
                continue

            post_capture_ships = ships - required
            min_post_capture = self.third_party_tail_min_post_capture_ships
            if target.owner == -1:
                min_post_capture = max(min_post_capture, self.third_party_tail_neutral_min_post_capture_ships)
            if post_capture_ships < min_post_capture:
                continue
            if (
                self.third_party_tail_overpay_min_post_capture > 0
                and ships > direct_need
                and post_capture_ships < self.third_party_tail_overpay_min_post_capture
            ):
                continue
            if self._third_party_tail_hold_filter_blocks(target, local, post_capture_ships, arrival):
                continue

            score = (
                self.third_party_tail_bonus
                + target.production * self.third_party_tail_prod_weight
                + savings * self.third_party_tail_savings_weight
                + post_capture_ships * 0.2
                - ships * 0.15
                - max(0, delay - self.third_party_tail_min_delay)
            )
            plan = {
                "ships": float(ships),
                "arrival": float(arrival),
                "delay": float(delay),
                "post_capture_ships": float(post_capture_ships),
                "score": float(score),
            }
            if best is None or plan["score"] > best["score"]:
                best = plan
        return best

    def _third_party_tail_hold_filter_blocks(
        self,
        target: Planet,
        local: LocalObs,
        post_capture_ships: int,
        our_arrival: int,
    ) -> bool:
        if not self.enable_third_party_tail_hold_filter:
            return False

        for fleet in local.fleets:
            if fleet.owner in (-1, local.player):
                continue
            fleet_arrival = self._fleet_arrival_to_target(fleet, target, local)
            if fleet_arrival is None or fleet_arrival <= our_arrival:
                continue
            recapture_delay = fleet_arrival - our_arrival
            if recapture_delay > self.third_party_tail_hold_enemy_max_arrival:
                continue
            projected_defense = post_capture_ships + int(target.production * recapture_delay)
            if fleet.ships >= projected_defense + self.third_party_tail_hold_margin:
                return True

        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            dist = distance(enemy, target)
            if dist > self.third_party_tail_hold_enemy_radius:
                continue
            sendable = int(enemy.ships * self.third_party_tail_hold_enemy_send_fraction)
            sendable += int(enemy.production * self.third_party_tail_hold_enemy_launch_window)
            sendable -= int(enemy.production * self.third_party_tail_hold_enemy_reserve_turns)
            sendable = max(0, sendable)
            if sendable <= 0:
                continue
            arrival = travel_ticks(enemy, target, sendable)
            if arrival > self.third_party_tail_hold_enemy_max_arrival:
                continue
            projected_defense = post_capture_ships + int(target.production * arrival)
            if sendable >= projected_defense + self.third_party_tail_hold_margin:
                return True
        return False

    def _third_party_anti_tail_hold_blocks(
        self,
        target: Planet,
        local: LocalObs,
        post_capture_ships: int,
        our_arrival: int,
    ) -> bool:
        if not self.enable_third_party_anti_tail_hold_gate:
            return False
        if target.owner == local.player or target.production < self.anti_tail_hold_min_production:
            return False
        if (
            self.anti_tail_hold_min_active_players > 0
            and self._active_player_count(local) < self.anti_tail_hold_min_active_players
        ):
            return False

        for fleet in local.fleets:
            if fleet.owner in (-1, local.player):
                continue
            fleet_arrival = self._fleet_arrival_to_target(fleet, target, local)
            if fleet_arrival is None or fleet_arrival <= our_arrival:
                continue
            recapture_delay = fleet_arrival - our_arrival
            if recapture_delay > self.anti_tail_hold_enemy_max_arrival:
                continue
            projected_defense = post_capture_ships + int(target.production * recapture_delay)
            if fleet.ships >= projected_defense + self.anti_tail_hold_margin:
                return True

        if self.anti_tail_hold_enemy_radius >= 0:
            for enemy in local.planets:
                if enemy.owner in (-1, local.player):
                    continue
                if distance(enemy, target) > self.anti_tail_hold_enemy_radius:
                    continue
                sendable = int(enemy.ships * self.anti_tail_hold_enemy_send_fraction)
                sendable += int(enemy.production * self.anti_tail_hold_enemy_launch_window)
                sendable -= int(enemy.production * self.anti_tail_hold_enemy_reserve_turns)
                sendable = max(0, sendable)
                if sendable <= 0:
                    continue
                arrival = travel_ticks(enemy, target, sendable)
                if arrival > self.anti_tail_hold_enemy_max_arrival:
                    continue
                projected_defense = post_capture_ships + int(target.production * arrival)
                if sendable >= projected_defense + self.anti_tail_hold_margin:
                    return True
        return False

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
        tail_plan = self._third_party_tail_capture_plan(source, target, local)
        is_tail_capture = tail_plan is not None and int(tail_plan["ships"]) <= base_ships
        planned_ships = int(tail_plan["ships"]) if is_tail_capture else base_ships

        available = self._available_local_attack_ships(source, local, under_attack)
        if available < planned_ships:
            return False

        total_ships = planned_ships
        angle: float | None
        arrive_tick: int | None

        if target.id in self.moving_planets:
            angle, arrive_tick = self._moving_attack_plan(
                source,
                target,
                total_ships,
                available,
                local,
                include_target_production=(not self.use_arrival_based_enemy_production and not is_tail_capture),
            )
        else:
            angle, arrive_tick, total_ships = self._static_attack_plan(
                source,
                target,
                total_ships,
                available,
                include_target_production=(not self.use_arrival_based_enemy_production and not is_tail_capture),
            )

        if angle is None or arrive_tick is None:
            return False

        if is_tail_capture:
            tail_plan = self._third_party_tail_capture_plan(source, target, local)
            if tail_plan is None or int(tail_plan["ships"]) > total_ships:
                return False
        else:
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

        if self._contested_stop_loss_blocks(source, target, total_ships, local):
            return False

        if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
            return False

        path_target = self._resolve_path_target(source, target, total_ships, angle, arrive_tick, local)
        if path_target is None:
            return False
        if path_target.id != target.id:
            target = path_target
            base_ships = self._base_ships_needed(target, local, source=source)
            if base_ships is None:
                return False
            total_ships = max(total_ships, base_ships)
            if total_ships > available:
                return False
            angle, arrive_tick = self._angle_and_arrival(source, target, total_ships, local)
            if angle is None or arrive_tick is None:
                return False
            if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
                return False
            if not self._path_hits_target(source, target, total_ships, angle, arrive_tick, local):
                return False

        if not is_tail_capture:
            adjusted_hold_ships = self._capture_hold_adjusted_ships(source, target, total_ships, arrive_tick, available, local)
            if adjusted_hold_ships is None:
                return False
            if adjusted_hold_ships > total_ships:
                total_ships = adjusted_hold_ships
                angle, arrive_tick = self._angle_and_arrival(source, target, total_ships, local)
                if angle is None or arrive_tick is None:
                    return False
                if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
                    return False
                if not self._path_hits_target(source, target, total_ships, angle, arrive_tick, local):
                    return False

            adjusted_recapture_hold_ships = self._recent_loss_recapture_hold_adjusted_ships(
                source,
                target,
                total_ships,
                arrive_tick,
                available,
                local,
            )
            if adjusted_recapture_hold_ships is None:
                return False
            if adjusted_recapture_hold_ships > total_ships:
                total_ships = adjusted_recapture_hold_ships
                angle, arrive_tick = self._angle_and_arrival(source, target, total_ships, local)
                if angle is None or arrive_tick is None:
                    return False
                if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
                    return False
                if not self._path_hits_target(source, target, total_ships, angle, arrive_tick, local):
                    return False

            adjusted_opening_ships = self._opening_hold_adjusted_ships(source, target, total_ships, arrive_tick, available, local)
            if adjusted_opening_ships is None:
                return False
            if adjusted_opening_ships > total_ships:
                total_ships = adjusted_opening_ships
                angle, arrive_tick = self._angle_and_arrival(source, target, total_ships, local)
                if angle is None or arrive_tick is None:
                    return False
                if self.enable_sun_avoidance and sun_collision(source, total_ships, angle):
                    return False
                if not self._path_hits_target(source, target, total_ships, angle, arrive_tick, local):
                    return False

            post_capture_ships = self._post_capture_ships(source, target, total_ships, local)
            if self._third_party_anti_tail_hold_blocks(target, local, post_capture_ships, arrive_tick):
                return False

        if self._source_exposed_after_send(source, target, total_ships, arrive_tick, local):
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

            if self._contested_stop_loss_blocks(source, target, required, local):
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

    def _contested_stop_loss_active(self, local: LocalObs) -> bool:
        if not self.enable_contested_stop_loss:
            return False
        active_players = self._active_player_count(local)
        if active_players < self.contested_stop_loss_min_active_players:
            return False
        if self.contested_stop_loss_max_active_players > 0 and active_players > self.contested_stop_loss_max_active_players:
            return False
        return True

    def _contested_stop_loss_flip_count(self, target: Planet, local: LocalObs) -> int:
        if not self._contested_stop_loss_active(local):
            return 0
        keep_after = local.step - max(1, self.contested_stop_loss_window)
        return sum(1 for step in self.planet_flip_steps.get(target.id, []) if step >= keep_after)

    def _contested_stop_loss_penalty(self, target: Planet, local: LocalObs) -> float:
        if target.owner == local.player:
            return 0.0
        flips = self._contested_stop_loss_flip_count(target, local)
        if flips < self.contested_stop_loss_flip_threshold:
            return 0.0
        if target.production >= self.contested_stop_loss_high_prod_exception_min:
            return 0.0
        if target.production > self.contested_stop_loss_low_prod_max:
            return 0.0
        extra_flips = max(0, flips - self.contested_stop_loss_flip_threshold)
        return self.contested_stop_loss_penalty * (1.0 + 0.35 * extra_flips)

    def _contested_stop_loss_blocks(
        self,
        source: Planet,
        target: Planet,
        total_ships: int,
        local: LocalObs,
    ) -> bool:
        if target.owner == local.player:
            return False
        flips = self._contested_stop_loss_flip_count(target, local)
        if flips < self.contested_stop_loss_flip_threshold:
            return False
        if target.production >= self.contested_stop_loss_high_prod_exception_min:
            return False
        if target.production > self.contested_stop_loss_low_prod_max:
            return False
        post_capture = self._post_capture_ships(source, target, total_ships, local)
        required_hold = max(
            self.contested_stop_loss_min_hold,
            int(target.production * max(0, self.contested_stop_loss_prod_hold_turns)),
        )
        return post_capture < required_hold

    def _target_score(self, source: Planet, target: Planet, local: LocalObs) -> float:
        score = public_custom_score(source, target)
        score += self._early_neutral_score_adjustment(source, target, local)
        score += self._recent_loss_recapture_score(target, local)
        score -= self._contested_stop_loss_penalty(target, local)
        score += self._third_party_tail_capture_score(source, target, local)
        score += self._multiplayer_diplomacy_score(source, target, local)
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

    def _multiplayer_diplomacy_score(self, source: Planet, target: Planet, local: LocalObs) -> float:
        if not self.enable_multiplayer_diplomacy_score:
            return 0.0

        active_players = {
            planet.owner
            for planet in local.planets
            if planet.owner != -1
        } | {
            fleet.owner
            for fleet in local.fleets
            if fleet.owner != -1
        }
        if len(active_players) < self.multiplayer_min_active_players:
            return 0.0

        if target.owner == -1:
            return self.multiplayer_neutral_bonus * target.production
        if target.owner == local.player:
            return 0.0

        my_planets = [planet for planet in local.planets if planet.owner == local.player]
        enemy_planets = [planet for planet in local.planets if planet.owner == target.owner]
        if not my_planets or not enemy_planets:
            return 0.0

        border_distance = min(distance(mine, enemy) for mine in my_planets for enemy in enemy_planets)
        adjustment = 0.0
        if border_distance > self.multiplayer_far_enemy_distance:
            gap = border_distance - self.multiplayer_far_enemy_distance
            adjustment -= self.multiplayer_far_enemy_penalty * gap / max(self.multiplayer_far_enemy_distance, 1.0)
        else:
            closeness = 1.0 - border_distance / max(self.multiplayer_far_enemy_distance, 1.0)
            adjustment += self.multiplayer_local_enemy_bonus * closeness

        prod_by_owner: dict[int, float] = {}
        for planet in local.planets:
            if planet.owner == -1:
                continue
            prod_by_owner[planet.owner] = prod_by_owner.get(planet.owner, 0.0) + planet.production
        if prod_by_owner:
            leader_prod = max(prod_by_owner.values())
            target_owner_prod = prod_by_owner.get(target.owner, 0.0)
            if leader_prod > 0 and target_owner_prod >= leader_prod:
                adjustment += self.multiplayer_leader_prod_bonus * target.production
        return adjustment

    def _recent_loss_recapture_score(self, target: Planet, local: LocalObs) -> float:
        if not self.enable_recent_loss_recapture_bias:
            return 0.0
        if local.step < self.recent_loss_recapture_min_step or local.step > self.recent_loss_recapture_max_step:
            return 0.0
        lost_step = self.recently_lost_steps.get(target.id)
        if lost_step is None or local.step - lost_step > self.recent_loss_recapture_window:
            return 0.0
        min_production = self._recent_loss_recapture_min_production_for(target, local)
        if target.owner in (-1, local.player) or target.production < min_production:
            return 0.0
        age = max(0, local.step - lost_step)
        freshness = 1.0 - age / max(1, self.recent_loss_recapture_window)
        return self.recent_loss_recapture_bonus * freshness + target.production * self.recent_loss_recapture_prod_weight

    def _recent_loss_tracking_min_production(self, local: LocalObs) -> float:
        if not self.enable_dynamic_front_base_recapture:
            return self.recent_loss_recapture_min_production
        active_players = self._active_player_count(local)
        dynamic_possible = False
        if self.dynamic_recapture_enable_2p and active_players == 2:
            dynamic_possible = True
        if self.dynamic_recapture_enable_4p_collapse and active_players >= 4:
            dynamic_possible = True
        if self.dynamic_recapture_enable_front_base:
            dynamic_possible = True
        if not dynamic_possible:
            return self.recent_loss_recapture_min_production
        return min(self.recent_loss_recapture_min_production, self.dynamic_recapture_min_production)

    def _recent_loss_recapture_min_production_for(self, target: Planet, local: LocalObs) -> float:
        if self._dynamic_front_base_recapture_active(target, local):
            return min(self.recent_loss_recapture_min_production, self.dynamic_recapture_min_production)
        return self.recent_loss_recapture_min_production

    def _dynamic_front_base_recapture_active(self, target: Planet, local: LocalObs) -> bool:
        if not self.enable_dynamic_front_base_recapture:
            return False
        active_players = self._active_player_count(local)
        if self.dynamic_recapture_enable_2p and active_players == 2:
            return True
        if self.dynamic_recapture_enable_4p_collapse and active_players >= 4:
            recent_losses = sum(
                1
                for step in self.recently_lost_steps.values()
                if local.step - step <= self.dynamic_recapture_collapse_window
            )
            if recent_losses >= self.dynamic_recapture_collapse_loss_count:
                return True
        if self.dynamic_recapture_enable_front_base:
            own_neighbors = sum(
                1
                for planet in local.mine
                if planet.id != target.id and distance(planet, target) <= self.dynamic_recapture_front_own_radius
            )
            if own_neighbors >= self.dynamic_recapture_front_min_own_neighbors:
                return True
            for planet in local.mine:
                captured_step = self.recently_captured_steps.get(planet.id)
                if captured_step is None or local.step - captured_step > self.dynamic_recapture_anchor_window:
                    continue
                if planet.production < self.dynamic_recapture_anchor_min_production:
                    continue
                if distance(planet, target) <= self.dynamic_recapture_anchor_radius:
                    return True
        return False

    def _third_party_tail_capture_score(self, source: Planet, target: Planet, local: LocalObs) -> float:
        plan = self._third_party_tail_capture_plan(source, target, local)
        if plan is None:
            return 0.0
        return float(plan["score"])

    def _early_neutral_profile(self, local: LocalObs) -> tuple[int, float, float, float, float]:
        active_players = self._active_player_count(local)
        if (
            self.enable_early_neutral_multiplayer_override
            and active_players >= self.early_neutral_multiplayer_min_active_players
            and (
                self.early_neutral_multiplayer_max_active_players <= 0
                or active_players <= self.early_neutral_multiplayer_max_active_players
            )
        ):
            return (
                self.early_neutral_multiplayer_step_limit,
                self.early_neutral_multiplayer_min_production,
                self.early_neutral_multiplayer_bonus,
                self.early_neutral_multiplayer_safe_bonus,
                self.early_neutral_multiplayer_contested_penalty,
            )
        return (
            self.early_neutral_step_limit,
            self.early_neutral_min_production,
            self.early_neutral_bonus,
            self.early_neutral_safe_bonus,
            self.early_neutral_contested_penalty,
        )

    def _early_neutral_allowed(self, source: Planet, target: Planet, local: LocalObs) -> bool:
        step_limit, min_production, _, _, _ = self._early_neutral_profile(local)
        if target.owner != -1 or local.step > step_limit:
            return False
        if target.production < min_production:
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
        if enemy_eta - eta < self.early_neutral_dynamic_min_enemy_gap:
            return False
        if self.early_neutral_dynamic_check_source_safety and not self._early_neutral_source_safe_after_send(source, needed, local):
            return False
        if self.early_neutral_dynamic_check_target_hold and not self._early_neutral_target_holdable(target, needed, eta, local):
            return False
        return True

    def _early_neutral_source_safe_after_send(self, source: Planet, needed: int, local: LocalObs) -> bool:
        remaining = source.ships - needed
        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            if distance(enemy, source) > self.early_neutral_dynamic_source_threat_radius:
                continue
            sendable = self._project_enemy_sendable(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            arrival = travel_ticks(enemy, source, sendable)
            projected = remaining + int(source.production * max(0, arrival))
            if projected < sendable + self.early_neutral_dynamic_source_safety_margin:
                return False
        return True

    def _early_neutral_target_holdable(self, target: Planet, sent: int, capture_eta: int, local: LocalObs) -> bool:
        post_capture = max(1, sent - target.ships)
        for enemy in local.planets:
            if enemy.owner in (-1, local.player):
                continue
            sendable = self._project_enemy_sendable(enemy)
            if sendable < self.min_ships_mine_attack:
                continue
            recapture_eta = travel_ticks(enemy, target, sendable)
            if recapture_eta <= capture_eta:
                continue
            produced_after_capture = max(0, recapture_eta - capture_eta)
            projected = post_capture + int(target.production * produced_after_capture)
            if projected < sendable + self.early_neutral_dynamic_target_hold_margin:
                return False
        return True

    def _early_neutral_score_adjustment(self, source: Planet, target: Planet, local: LocalObs) -> float:
        if target.owner != -1:
            return 0.0

        needed = max(1, target.ships + 1)
        eta = self._estimate_arrival_for_requirement(source, target, needed, local)
        adjustment = self._opening_neutral_territory_score(source, target, local, eta)

        if self.enable_opening_rotating_neutral_filter and local.step <= self.opening_rotating_step_limit:
            if target.id in self.moving_planets and (eta > self.opening_rotating_max_eta or target.production <= self.opening_rotating_low_production):
                adjustment -= self.opening_rotating_penalty

        if not self.enable_early_neutral_bias or not self._early_neutral_allowed(source, target, local):
            return adjustment

        _, _, bonus_weight, safe_bonus, contested_penalty = self._early_neutral_profile(local)
        bonus = bonus_weight * target.production
        if target.id not in self.moving_planets:
            bonus *= self.early_neutral_static_multiplier

        enemy_eta = self._nearest_enemy_eta(target, local)
        if enemy_eta < 10**8:
            gap = enemy_eta - eta
            if gap >= self.early_neutral_reaction_margin:
                bonus += safe_bonus
            elif abs(gap) <= self.early_neutral_reaction_margin:
                bonus -= contested_penalty

        return adjustment + bonus

    def _opening_neutral_territory_score(self, source: Planet, target: Planet, local: LocalObs, eta: int) -> float:
        if not self.enable_opening_neutral_territory_score:
            return 0.0
        if local.step > self.opening_territory_step_limit:
            return 0.0
        if self.opening_territory_min_active_players > 0 and self._active_player_count(local) < self.opening_territory_min_active_players:
            return 0.0

        enemy_planets = [planet for planet in local.planets if planet.owner not in (-1, local.player)]
        if not enemy_planets:
            return 0.0

        own_dist = min(distance(source, target), *(distance(planet, target) for planet in local.mine))
        enemy_dist = min(distance(planet, target) for planet in enemy_planets)
        enemy_advantage = own_dist - enemy_dist
        if enemy_advantage <= self.opening_territory_enemy_closer_margin:
            return 0.0

        enemy_eta = self._nearest_enemy_eta(target, local)
        if enemy_eta - eta >= self.opening_territory_allow_if_safe_gap:
            return 0.0

        penalty = self.opening_territory_penalty + target.production * self.opening_territory_prod_scale
        penalty += max(0.0, enemy_advantage - self.opening_territory_enemy_closer_margin)
        return -penalty

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
            if not self._path_hits_target(source, target, ships, angle, arrive_tick, local):
                return None
            if self._source_exposed_after_send(source, target, ships, arrive_tick, local):
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

    def _resolve_path_target(
        self,
        source: Planet,
        intended: Planet,
        ships: int,
        angle: float,
        arrive_tick: int,
        local: LocalObs,
    ) -> Planet | None:
        if not self._path_first_hit_enabled(local):
            return intended
        first_hit = self._first_planet_hit(source, angle, ships, max(1, arrive_tick), local)
        if first_hit is None or first_hit.id == intended.id:
            return intended
        if not self.enable_path_first_hit_redirect:
            return None
        if first_hit.owner == local.player:
            return None
        redirected_score = self._target_score(source, first_hit, local)
        intended_wait_score = self._target_score(source, intended, local) - self.path_block_wait_penalty
        if redirected_score >= intended_wait_score:
            return first_hit
        return None

    def _path_hits_target(
        self,
        source: Planet,
        target: Planet,
        ships: int,
        angle: float,
        arrive_tick: int,
        local: LocalObs,
    ) -> bool:
        if not self._path_first_hit_enabled(local):
            return True
        first_hit = self._first_planet_hit(source, angle, ships, max(1, arrive_tick), local)
        return first_hit is None or first_hit.id == target.id

    def _path_first_hit_enabled(self, local: LocalObs) -> bool:
        if not self.enable_path_first_hit_check:
            return False
        if self.path_first_hit_min_active_players <= 0:
            return True
        return self._active_player_count(local) >= self.path_first_hit_min_active_players

    def _first_planet_hit(
        self,
        source: Planet,
        angle: float,
        ships: int,
        max_tick: int,
        local: LocalObs,
    ) -> Planet | None:
        speed = fleet_speed(ships)
        prev_x, prev_y = source.x, source.y
        moving_paths = {
            planet.id: planet_trajectory(planet, local.angular_velocity, max_tick)
            for planet in local.planets
            if planet.id in self.moving_planets
        }
        for tick in range(1, max_tick + 1):
            x = source.x + math.cos(angle) * speed * tick
            y = source.y + math.sin(angle) * speed * tick
            hits: list[tuple[float, Planet]] = []
            for planet in local.planets:
                if planet.id == source.id:
                    continue
                px, py = (planet.x, planet.y)
                path = moving_paths.get(planet.id)
                if path is not None and tick - 1 < len(path):
                    px, py = path[tick - 1]
                if collides_segment_circle(prev_x, prev_y, x, y, px, py, planet.radius + self.path_first_hit_padding):
                    hits.append((math.hypot(px - source.x, py - source.y), planet))
            if hits:
                hits.sort(key=lambda row: row[0])
                return hits[0][1]
            prev_x, prev_y = x, y
        return None

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
