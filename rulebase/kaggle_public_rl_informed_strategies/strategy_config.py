"""Central strategy switches and ablation grids.

All public-rule and RL-informed knobs live here so later cluster search or
ML-driven tuning can generate agents from one stable schema.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True, slots=True)
class StrategyConfig:
    # Public rule knobs.
    warmup_steps: int = 2
    min_ships_mine_attack: int = 10
    min_ships_target_coop_attack: int = 20
    coop_planet_cap: int = 8
    target_candidate_limit: int = 3
    skip_comet_targets: bool = True
    enemy_owned_production_buffer_turns: int = 3
    use_arrival_based_enemy_production: bool = False
    arrival_enemy_production_safety_turns: int = 0
    arrival_enemy_production_max_turns: int = 80
    en_route_skip_owned_ratio: float = 0.75
    enable_single_attacks: bool = True
    enable_coop_attacks: bool = True
    enable_reinforcements: bool = True
    enable_sun_avoidance: bool = True
    enable_path_first_hit_check: bool = False
    enable_path_first_hit_redirect: bool = True
    path_first_hit_min_active_players: int = 0
    path_first_hit_padding: float = 0.8
    path_block_wait_penalty: float = 12.0
    enable_comet_evacuation: bool = False
    comet_evacuation_remaining_turns: int = 12
    comet_evacuation_min_ships: int = 8
    comet_evacuation_front_distance: float = 45.0
    comet_evacuation_own_prod_weight: float = 16.0
    comet_evacuation_front_bonus: float = 12.0
    comet_evacuation_target_roi: float = 1.10
    comet_evacuation_target_prod_weight: float = 18.0
    comet_evacuation_target_enemy_bonus: float = 20.0
    enable_endgame_fleet_dump: bool = False
    endgame_dump_min_step: int = 470
    endgame_dump_min_ships: int = 20
    endgame_dump_keep_source_ships: int = 0
    endgame_dump_angle_samples: int = 32
    endgame_dump_min_active_players: int = 0
    enable_arrival_based_under_attack_availability: bool = False
    under_attack_availability_min_step: int = 0
    under_attack_availability_margin: int = 0
    under_attack_availability_horizon: int = 60
    enable_capture_hold_margin_gate: bool = False
    capture_hold_min_active_players: int = 4
    capture_hold_min_production: float = 4.0
    capture_hold_enemy_radius: float = 45.0
    capture_hold_enemy_send_fraction: float = 0.85
    capture_hold_enemy_launch_window: int = 8
    capture_hold_enemy_reserve_turns: int = 2
    capture_hold_enemy_max_arrival: int = 45
    capture_hold_margin: int = 6
    capture_hold_allow_extra_send: bool = True
    capture_hold_use_post_capture_window: bool = False
    enable_contested_target_adjustment: bool = False
    contested_arrival_margin: int = 3
    contested_enemy_weight: float = 1.0
    contested_friendly_credit: float = 0.75
    contested_skip_friendly_covered: bool = True
    enable_dynamic_posture: bool = False
    posture_max_active_players: int = 99
    posture_defensive_min_step: int = 0
    posture_aggressive_prod_deficit: float = 5.0
    posture_aggressive_ship_ratio: float = 0.85
    posture_aggressive_planet_deficit: int = 2
    posture_defensive_prod_lead: float = 6.0
    posture_defensive_ship_ratio: float = 1.15
    posture_defensive_planet_lead: int = 2
    posture_late_step: int = 250
    aggressive_min_attack_delta: int = -2
    aggressive_target_candidate_bonus: int = 1
    defensive_min_attack_delta: int = 2
    defensive_target_candidate_delta: int = -1
    defensive_reserve_turns: int = 3
    defensive_min_garrison: int = 8
    defensive_high_prod_bonus: int = 4
    enable_value_defense: bool = False
    value_defense_min_production: float = 3.0
    value_defense_horizon: int = 55
    value_defense_buffer_turns: int = 3
    value_defense_min_margin: int = 8
    value_defense_max_send: int = 40
    value_defense_roi_multiplier: float = 1.25
    value_defense_multiplayer_min_active_players: int = 0
    value_defense_multiplayer_horizon: int = 0
    value_defense_multiplayer_max_send: int = 0
    value_defense_multiplayer_min_margin: int = 0
    enable_proactive_value_defense: bool = False
    proactive_defense_min_active_players: int = 0
    proactive_defense_max_active_players: int = 99
    proactive_defense_min_production: float = 3.0
    proactive_defense_radius: float = 45.0
    proactive_defense_base_margin: int = 8
    proactive_defense_prod_turns: int = 4
    proactive_defense_enemy_prod_weight: float = 2.0
    proactive_defense_enemy_ship_weight: float = 0.04
    proactive_defense_enemy_launch_window: int = 8
    proactive_defense_enemy_reserve_turns: int = 2
    proactive_defense_enemy_send_fraction: float = 0.85
    proactive_defense_threat_slack: int = 6
    proactive_defense_max_send: int = 28
    proactive_defense_max_targets: int = 1
    proactive_defense_max_arrival: int = 45
    proactive_defense_roi_multiplier: float = 1.50
    proactive_defense_after_attacks: bool = False
    proactive_defense_require_turn_attack: bool = False
    proactive_defense_require_recent_capture: bool = False
    proactive_defense_recent_capture_window: int = 35
    proactive_defense_require_enemy_positive_roi: bool = False
    proactive_defense_enemy_roi_multiplier: float = 1.20
    proactive_defense_enemy_min_net_value: float = 20.0
    proactive_defense_min_step: int = 0
    proactive_defense_max_step: int = 500
    proactive_defense_min_prod_diff: float = -999.0
    proactive_defense_min_planet_diff: int = -999
    proactive_defense_min_ship_ratio: float = 0.0
    proactive_defense_source_min_after: int = 0
    proactive_defense_source_prod_turns_after: int = 0
    proactive_defense_source_front_distance: float = 35.0
    proactive_defense_source_front_bonus: int = 0
    enable_local_source_reserve: bool = False
    local_reserve_min_step: int = 0
    local_reserve_max_step: int = 500
    local_reserve_min_production: float = 3.0
    local_reserve_enemy_distance: float = 35.0
    local_reserve_turns: int = 2
    local_reserve_min_garrison: int = 6
    local_reserve_front_bonus: int = 8
    enable_recent_capture_source_reserve: bool = False
    recent_capture_source_reserve_min_active_players: int = 0
    recent_capture_source_reserve_min_step: int = 0
    recent_capture_source_reserve_max_step: int = 500
    recent_capture_source_reserve_window: int = 35
    recent_capture_source_reserve_min_production: float = 4.0
    recent_capture_source_reserve_min_after: int = 8
    recent_capture_source_reserve_prod_turns_after: int = 1
    recent_capture_source_reserve_enemy_radius: float = 45.0
    recent_capture_source_reserve_front_bonus: int = 4
    enable_source_threat_reserve: bool = False
    source_threat_min_active_players: int = 0
    source_threat_min_step: int = 0
    source_threat_max_step: int = 500
    source_threat_min_production: float = 3.0
    source_threat_radius: float = 45.0
    source_threat_enemy_send_fraction: float = 0.85
    source_threat_enemy_launch_window: int = 8
    source_threat_enemy_reserve_turns: int = 2
    source_threat_max_arrival: int = 45
    source_threat_margin: int = 6
    source_threat_roi_multiplier: float = 1.20
    source_threat_min_net_value: float = 15.0
    enable_source_threat_send_filter: bool = False
    source_threat_send_min_active_players: int = 0
    source_threat_send_max_active_players: int = 99
    source_threat_send_min_step: int = 0
    source_threat_send_max_step: int = 500
    source_threat_send_min_production: float = 3.0
    source_threat_send_radius: float = 45.0
    source_threat_send_enemy_fraction: float = 0.85
    source_threat_send_enemy_launch_window: int = 8
    source_threat_send_enemy_reserve_turns: int = 2
    source_threat_send_max_arrival: int = 45
    source_threat_send_margin: int = 6
    source_threat_send_roi_multiplier: float = 1.25
    source_threat_send_min_net_value: float = 20.0
    source_threat_send_trade_ratio: float = 1.0
    enable_source_threat_target_penalty: bool = False
    source_threat_target_penalty_weight: float = 0.05
    enable_local_source_defense_gate: bool = False
    local_source_defense_gate_min_active_players: int = 0
    local_source_defense_gate_min_step: int = 0
    local_source_defense_gate_max_step: int = 500
    local_source_defense_gate_min_production: float = 3.0
    local_source_defense_gate_front_distance: float = 45.0
    local_source_defense_gate_enemy_fraction: float = 0.85
    local_source_defense_gate_enemy_launch_window: int = 8
    local_source_defense_gate_enemy_reserve_turns: int = 2
    local_source_defense_gate_max_arrival: int = 45
    local_source_defense_gate_margin: int = 8
    local_source_defense_gate_use_arrival_production: bool = True
    enable_holdability_target_score: bool = False
    holdability_radius: float = 35.0
    holdability_weight: float = 0.8
    holdability_enemy_prod_weight: float = 4.0
    holdability_enemy_ship_weight: float = 0.08
    holdability_own_prod_weight: float = 2.0
    holdability_own_ship_weight: float = 0.04
    enable_early_neutral_bias: bool = False
    early_neutral_min_active_players: int = 0
    early_neutral_max_active_players: int = 99
    early_neutral_step_limit: int = 40
    early_neutral_min_production: float = 3.0
    early_neutral_max_ships: int = 15
    early_neutral_max_eta: int = 35
    early_neutral_bonus: float = 8.0
    early_neutral_static_multiplier: float = 1.25
    early_neutral_safe_bonus: float = 12.0
    early_neutral_contested_penalty: float = 8.0
    early_neutral_reaction_margin: int = 2
    early_neutral_holdability_relief: float = 0.50
    enable_early_neutral_multiplayer_override: bool = False
    early_neutral_multiplayer_min_active_players: int = 4
    early_neutral_multiplayer_max_active_players: int = 99
    early_neutral_multiplayer_step_limit: int = 40
    early_neutral_multiplayer_min_production: float = 3.0
    early_neutral_multiplayer_bonus: float = 8.0
    early_neutral_multiplayer_safe_bonus: float = 12.0
    early_neutral_multiplayer_contested_penalty: float = 8.0
    enable_early_neutral_dynamic_max_ships: bool = False
    early_neutral_dynamic_max_ships: int = 20
    early_neutral_dynamic_min_production: float = 4.0
    early_neutral_dynamic_min_enemy_gap: int = 4
    early_neutral_dynamic_source_min_after: int = 10
    early_neutral_dynamic_check_source_safety: bool = False
    early_neutral_dynamic_source_threat_radius: float = 45.0
    early_neutral_dynamic_source_safety_margin: int = 4
    early_neutral_dynamic_check_target_hold: bool = False
    early_neutral_dynamic_target_hold_margin: int = 4
    enable_opening_neutral_territory_score: bool = False
    opening_territory_min_active_players: int = 4
    opening_territory_step_limit: int = 45
    opening_territory_enemy_closer_margin: float = 6.0
    opening_territory_penalty: float = 20.0
    opening_territory_prod_scale: float = 4.0
    opening_territory_allow_if_safe_gap: int = 10
    enable_opening_neutral_hold_margin: bool = False
    opening_hold_min_active_players: int = 4
    opening_hold_max_active_players: int = 99
    opening_hold_step_limit: int = 55
    opening_hold_min_production: float = 3.0
    opening_hold_base_margin: int = 4
    opening_hold_prod_turns: int = 2
    opening_hold_contested_extra: int = 6
    opening_hold_allow_extra_send: bool = True
    enable_opening_rotating_neutral_filter: bool = False
    opening_rotating_step_limit: int = 80
    opening_rotating_max_eta: int = 13
    opening_rotating_low_production: float = 2.0
    opening_rotating_penalty: float = 60.0
    enable_opening_high_prod_trickle: bool = False
    opening_trickle_min_active_players: int = 4
    opening_trickle_max_active_players: int = 99
    opening_trickle_step_limit: int = 30
    opening_trickle_source_min_production: float = 4.0
    opening_trickle_target_min_production: float = 4.0
    opening_trickle_max_target_ships: int = 12
    opening_trickle_min_ships: int = 5
    enable_enemy_launch_punish: bool = False
    enemy_launch_punish_max_fleet_age: int = 12
    enemy_launch_punish_min_outgoing: int = 12
    enemy_launch_punish_min_production: float = 2.0
    enemy_launch_punish_bonus_weight: float = 0.45
    enemy_launch_punish_max_targets: int = 2
    enemy_launch_punish_min_step: int = 0
    enemy_launch_punish_max_step: int = 500
    enemy_launch_punish_min_prod_diff: float = -999.0
    enemy_launch_punish_min_planet_diff: int = -999
    enemy_launch_punish_min_ship_ratio: float = 0.0
    enable_recent_loss_recapture_bias: bool = False
    recent_loss_recapture_min_production: float = 3.0
    recent_loss_recapture_window: int = 60
    recent_loss_recapture_min_step: int = 0
    recent_loss_recapture_max_step: int = 500
    recent_loss_recapture_bonus: float = 25.0
    recent_loss_recapture_prod_weight: float = 5.0
    enable_recent_loss_recapture_hold_gate: bool = False
    recent_loss_recapture_hold_min_active_players: int = 4
    recent_loss_recapture_hold_min_step: int = 45
    recent_loss_recapture_hold_max_step: int = 180
    recent_loss_recapture_hold_window: int = 50
    recent_loss_recapture_hold_min_production: float = 4.0
    recent_loss_recapture_hold_enemy_radius: float = 45.0
    recent_loss_recapture_hold_margin: int = 8
    recent_loss_recapture_hold_allow_extra_send: bool = True
    enable_third_party_tail_capture: bool = False
    third_party_tail_min_active_players: int = 3
    third_party_tail_min_step: int = 0
    third_party_tail_max_step: int = 500
    third_party_tail_min_production: float = 3.5
    third_party_tail_max_enemy_arrival: int = 60
    third_party_tail_min_delay: int = 1
    third_party_tail_max_delay: int = 16
    third_party_tail_margin: int = 1
    third_party_tail_min_send: int = 1
    third_party_tail_max_ships: int = 18
    third_party_tail_source_min_after: int = 8
    third_party_tail_bonus: float = 35.0
    third_party_tail_prod_weight: float = 5.0
    third_party_tail_savings_weight: float = 1.0
    third_party_tail_min_savings: int = 0
    third_party_tail_min_savings_ratio: float = 0.0
    third_party_tail_min_post_capture_ships: int = 0
    third_party_tail_overpay_min_post_capture: int = 0
    third_party_tail_neutral_max_arrival: int = 999
    third_party_tail_neutral_min_post_capture_ships: int = 0
    third_party_tail_neutral_min_enemy_post_capture: int = 0
    third_party_tail_roi_multiplier: float = 1.10
    third_party_tail_min_net_value: float = 8.0
    enable_third_party_tail_candidate_injection: bool = False
    third_party_tail_candidate_limit: int = 2
    third_party_tail_candidate_min_score: float = 0.0
    third_party_tail_candidate_keep_front: int = 0
    third_party_tail_only_neutral_targets: bool = False
    enable_third_party_tail_hold_filter: bool = False
    third_party_tail_hold_enemy_radius: float = 45.0
    third_party_tail_hold_enemy_send_fraction: float = 0.85
    third_party_tail_hold_enemy_launch_window: int = 8
    third_party_tail_hold_enemy_reserve_turns: int = 2
    third_party_tail_hold_enemy_max_arrival: int = 40
    third_party_tail_hold_margin: int = 3
    enable_third_party_anti_tail_hold_gate: bool = False
    anti_tail_hold_min_active_players: int = 3
    anti_tail_hold_min_production: float = 3.0
    anti_tail_hold_enemy_radius: float = -1.0
    anti_tail_hold_enemy_send_fraction: float = 0.85
    anti_tail_hold_enemy_launch_window: int = 8
    anti_tail_hold_enemy_reserve_turns: int = 2
    anti_tail_hold_enemy_max_arrival: int = 24
    anti_tail_hold_margin: int = 2
    enable_third_party_tail_watchlist: bool = False
    third_party_tail_watchlist_horizon: int = 90
    third_party_tail_watchlist_post_window: int = 18
    third_party_tail_watchlist_max_entries: int = 16
    third_party_tail_watchlist_score_bonus: float = 8.0
    third_party_tail_watchlist_min_recheck_age: int = 1
    enable_global_attack_priority: bool = False
    global_attack_roi_weight: float = 0.0
    global_attack_arrival_penalty: float = 0.0
    global_attack_max_failed_pairs: int = 64
    enable_multiplayer_diplomacy_score: bool = False
    multiplayer_min_active_players: int = 4
    multiplayer_far_enemy_distance: float = 45.0
    multiplayer_far_enemy_penalty: float = 18.0
    multiplayer_local_enemy_bonus: float = 8.0
    multiplayer_leader_prod_bonus: float = 2.0
    multiplayer_neutral_bonus: float = 4.0
    enable_home_anchor_source_reserve: bool = False
    home_anchor_min_active_players: int = 0
    home_anchor_min_production: float = 3.0
    home_anchor_step_min: int = 20
    home_anchor_step_max: int = 140
    home_anchor_home_radius: float = 55.0
    home_anchor_min_after: int = 10
    home_anchor_prod_turns_after: int = 2
    home_anchor_front_threat_bonus: int = 8
    enable_midgame_border_source_reserve: bool = False
    midgame_border_min_active_players: int = 4
    midgame_border_step_min: int = 45
    midgame_border_step_max: int = 140
    midgame_border_min_production: float = 3.0
    midgame_border_enemy_radius: float = 60.0
    midgame_border_min_after: int = 12
    midgame_border_prod_turns_after: int = 2
    midgame_border_threat_margin: int = 8

    # RL-informed/custom attack-loop knobs.
    use_custom_attack_loop: bool = False
    use_rl_target_score: bool = True
    rl_score_weight: float = 1.0
    use_late_filter: bool = True
    use_comet_roi_filter: bool = True
    include_comet_targets: bool = True
    custom_candidate_limit: int = 4

    # Optional post-attack support.
    enable_front_support: bool = False
    support_distance_factor: float = 1.3
    support_min_available: int = 20
    support_fraction: float = 0.65
    support_min_send: int = 12
    support_max_arrival: int = 35

    def to_agent_kwargs(self) -> dict:
        return asdict(self)


PUBLIC_EXACT = StrategyConfig()
PRE_CONTESTED_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
)
PRE_LAUNCH_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
    use_arrival_based_enemy_production=True,
    arrival_enemy_production_safety_turns=1,
    enable_contested_target_adjustment=True,
    contested_arrival_margin=2,
    contested_enemy_weight=0.75,
    contested_friendly_credit=0.50,
    enable_value_defense=True,
    enable_holdability_target_score=True,
    holdability_weight=0.50,
    holdability_radius=35.0,
)
PRE_EARLY_NEUTRAL_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
    use_arrival_based_enemy_production=True,
    arrival_enemy_production_safety_turns=1,
    enable_contested_target_adjustment=True,
    contested_arrival_margin=2,
    contested_enemy_weight=0.75,
    contested_friendly_credit=0.50,
    enable_value_defense=True,
    enable_holdability_target_score=True,
    holdability_weight=0.50,
    holdability_radius=35.0,
    enable_enemy_launch_punish=True,
    enemy_launch_punish_min_production=3.0,
    enemy_launch_punish_bonus_weight=0.45,
)
PRE_REACTION_MARGIN_REGULAR_CONFIG = StrategyConfig(
    **{
        **PRE_EARLY_NEUTRAL_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_early_neutral_bias": True,
        "early_neutral_bonus": 4.0,
        "early_neutral_safe_bonus": 20.0,
        "early_neutral_contested_penalty": 16.0,
    }
)
PRE_DYNAMIC_CAP_REGULAR_CONFIG = StrategyConfig(
    **{
        **PRE_REACTION_MARGIN_REGULAR_CONFIG.to_agent_kwargs(),
        "early_neutral_reaction_margin": 3,
    }
)
CAP18_BASE_REGULAR_CONFIG = StrategyConfig(
    **{
        **PRE_DYNAMIC_CAP_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_early_neutral_dynamic_max_ships": True,
        "early_neutral_dynamic_max_ships": 18,
        "early_neutral_dynamic_min_production": 4.0,
        "early_neutral_dynamic_min_enemy_gap": 4,
        "early_neutral_dynamic_source_min_after": 10,
    }
)
LAUNCH_TARGETS4_AGE8_REGULAR_CONFIG = StrategyConfig(
    **{
        **CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "enemy_launch_punish_max_fleet_age": 8,
        "enemy_launch_punish_max_targets": 4,
    }
)
RECENT_LOSS_RECAPTURE_PROD4_B40_REGULAR_CONFIG = StrategyConfig(
    **{
        **LAUNCH_TARGETS4_AGE8_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_recent_loss_recapture_bias": True,
        "recent_loss_recapture_min_production": 4.0,
        "recent_loss_recapture_window": 60,
        "recent_loss_recapture_bonus": 40.0,
    }
)
RECAPTURE_S45_E180_W50_B40_REGULAR_CONFIG = StrategyConfig(
    **{
        **RECENT_LOSS_RECAPTURE_PROD4_B40_REGULAR_CONFIG.to_agent_kwargs(),
        "recent_loss_recapture_min_step": 45,
        "recent_loss_recapture_max_step": 180,
        "recent_loss_recapture_window": 50,
        "recent_loss_recapture_bonus": 40.0,
    }
)
RECAPTURE_S45_E160_W50_B40_REGULAR_CONFIG = StrategyConfig(
    **{
        **RECAPTURE_S45_E180_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
        "recent_loss_recapture_max_step": 160,
    }
)
MP_SOFT_4P_REGULAR_CONFIG = StrategyConfig(
    **{
        **RECAPTURE_S45_E160_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_multiplayer_diplomacy_score": True,
        "multiplayer_far_enemy_penalty": 10.0,
        "multiplayer_local_enemy_bonus": 5.0,
        "multiplayer_leader_prod_bonus": 1.0,
        "multiplayer_neutral_bonus": 3.0,
    }
)
MP_SOFT_LEADER0_4P_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_SOFT_4P_REGULAR_CONFIG.to_agent_kwargs(),
        "multiplayer_leader_prod_bonus": 0.0,
    }
)
MP_LOCAL3_NEU5_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_SOFT_LEADER0_4P_REGULAR_CONFIG.to_agent_kwargs(),
        "multiplayer_local_enemy_bonus": 3.0,
        "multiplayer_neutral_bonus": 5.0,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_comet_evacuation": True,
        "comet_evacuation_remaining_turns": 12,
        "enable_path_first_hit_check": True,
        "enable_path_first_hit_redirect": True,
        "path_first_hit_min_active_players": 4,
        "path_first_hit_padding": 0.8,
        "path_block_wait_penalty": 12.0,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_capture_hold_margin_gate": True,
        "capture_hold_margin": 4,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_opening_neutral_territory_score": True,
        "opening_territory_penalty": 30.0,
        "opening_territory_enemy_closer_margin": 8.0,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG.to_agent_kwargs(),
        "opening_territory_step_limit": 35,
        "enable_home_anchor_source_reserve": True,
        "home_anchor_min_active_players": 4,
        "home_anchor_step_min": 60,
        "home_anchor_step_max": 130,
        "home_anchor_home_radius": 45.0,
        "home_anchor_min_after": 6,
        "home_anchor_prod_turns_after": 1,
        "home_anchor_front_threat_bonus": 4,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_REGULAR_CONFIG.to_agent_kwargs(),
        "capture_hold_use_post_capture_window": True,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_W50_S0_E500_B35_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_REGULAR_CONFIG.to_agent_kwargs(),
        "recent_loss_recapture_min_step": 0,
        "recent_loss_recapture_max_step": 500,
        "recent_loss_recapture_window": 50,
        "recent_loss_recapture_bonus": 35.0,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P2SRC_W002_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_W50_S0_E500_B35_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_source_threat_target_penalty": True,
        "source_threat_send_min_active_players": 2,
        "source_threat_send_max_active_players": 2,
        "source_threat_send_min_step": 25,
        "source_threat_send_max_step": 110,
        "source_threat_send_min_production": 4.0,
        "source_threat_send_radius": 45.0,
        "source_threat_send_margin": 6,
        "source_threat_send_roi_multiplier": 1.25,
        "source_threat_send_min_net_value": 20.0,
        "source_threat_target_penalty_weight": 0.02,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_W50_S0_E500_B35_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_source_threat_target_penalty": True,
        "source_threat_send_min_active_players": 4,
        "source_threat_send_max_active_players": 99,
        "source_threat_send_min_step": 35,
        "source_threat_send_max_step": 160,
        "source_threat_send_min_production": 4.0,
        "source_threat_send_radius": 40.0,
        "source_threat_send_margin": 4,
        "source_threat_send_roi_multiplier": 1.35,
        "source_threat_send_min_net_value": 25.0,
        "source_threat_target_penalty_weight": 0.02,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
        "multiplayer_leader_prod_bonus": 1.0,
    }
)
MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_P4EARLYSRC_S35_80_T120_REGULAR_CONFIG = StrategyConfig(
    **{
        **MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_source_threat_send_filter": True,
        "source_threat_send_min_active_players": 4,
        "source_threat_send_max_active_players": 4,
        "source_threat_send_min_step": 35,
        "source_threat_send_max_step": 80,
        "source_threat_send_min_production": 4.0,
        "source_threat_send_radius": 42.0,
        "source_threat_send_margin": 5,
        "source_threat_send_roi_multiplier": 1.40,
        "source_threat_send_min_net_value": 30.0,
        "source_threat_send_trade_ratio": 1.20,
    }
)
PRE_TAIL_REGULAR_CONFIG = MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_P4EARLYSRC_S35_80_T120_REGULAR_CONFIG
TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG = StrategyConfig(
    **{
        **PRE_TAIL_REGULAR_CONFIG.to_agent_kwargs(),
        "enable_third_party_tail_capture": True,
        "enable_third_party_tail_watchlist": True,
        "enable_third_party_tail_candidate_injection": False,
        "third_party_tail_min_active_players": 3,
        "third_party_tail_min_production": 3.0,
        "third_party_tail_max_enemy_arrival": 65,
        "third_party_tail_min_delay": 1,
        "third_party_tail_max_delay": 20,
        "third_party_tail_margin": 1,
        "third_party_tail_min_send": 1,
        "third_party_tail_max_ships": 12,
        "third_party_tail_source_min_after": 8,
        "third_party_tail_bonus": 45.0,
        "third_party_tail_roi_multiplier": 1.05,
        "third_party_tail_min_net_value": 6.0,
        "third_party_tail_watchlist_horizon": 90,
        "third_party_tail_watchlist_post_window": 18,
        "third_party_tail_watchlist_score_bonus": 4.0,
    }
)
TAIL_M2_MAX14_NET7_REGULAR_CANDIDATE_CONFIG = StrategyConfig(
    **{
        **TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
        "third_party_tail_margin": 2,
        "third_party_tail_max_ships": 14,
        "third_party_tail_min_net_value": 7.0,
    }
)
TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG = StrategyConfig(
    **{
        **TAIL_M2_MAX14_NET7_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
        "third_party_tail_overpay_min_post_capture": 4,
    }
)
TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG = StrategyConfig(
    **{
        **TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
        "enable_early_neutral_multiplayer_override": True,
        "early_neutral_multiplayer_min_active_players": 4,
        "early_neutral_multiplayer_max_active_players": 4,
        "early_neutral_multiplayer_step_limit": 50,
        "early_neutral_multiplayer_min_production": 2.0,
        "early_neutral_multiplayer_bonus": 6.0,
        "early_neutral_multiplayer_safe_bonus": 24.0,
        "early_neutral_multiplayer_contested_penalty": 12.0,
    }
)
REGULAR_CONFIG = TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG
PRE_HOLDABILITY_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
    use_arrival_based_enemy_production=True,
    arrival_enemy_production_safety_turns=1,
    enable_contested_target_adjustment=True,
    contested_arrival_margin=2,
    contested_enemy_weight=0.75,
    contested_friendly_credit=0.50,
    enable_value_defense=True,
)
PRE_VALUE_DEFENSE_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
    use_arrival_based_enemy_production=True,
    arrival_enemy_production_safety_turns=1,
    enable_contested_target_adjustment=True,
    contested_arrival_margin=2,
    contested_enemy_weight=0.75,
    contested_friendly_credit=0.50,
)
PRE_ARRIVAL_REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
    enable_contested_target_adjustment=True,
    contested_arrival_margin=2,
    contested_enemy_weight=0.75,
    contested_friendly_credit=0.50,
)

PUBLIC_LOOP_CONTROL = StrategyConfig(
    use_custom_attack_loop=True,
    use_rl_target_score=False,
    include_comet_targets=False,
    use_late_filter=False,
    use_comet_roi_filter=False,
    custom_candidate_limit=3,
)


def cfg(**overrides) -> dict:
    return StrategyConfig(**{**PUBLIC_EXACT.to_agent_kwargs(), **overrides}).to_agent_kwargs()


def from_base(base: StrategyConfig, **overrides) -> dict:
    return StrategyConfig(**{**base.to_agent_kwargs(), **overrides}).to_agent_kwargs()


PUBLIC_KNOB_VARIANTS = {
    "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
    "public_candidates_2": cfg(target_candidate_limit=2),
    "public_candidates_4": cfg(target_candidate_limit=4),
    "public_min_attack_8": cfg(min_ships_mine_attack=8),
    "public_min_attack_12": cfg(min_ships_mine_attack=12),
    "public_enemy_buffer_2": cfg(enemy_owned_production_buffer_turns=2),
    "public_enemy_buffer_4": cfg(enemy_owned_production_buffer_turns=4),
    "public_no_reinforce": cfg(enable_reinforcements=False),
    "public_no_coop": cfg(enable_coop_attacks=False),
}

FRONT_SUPPORT_GRID_VARIANTS = {
    "front_off_public": PUBLIC_EXACT.to_agent_kwargs(),
    "front_soft_35_035_20_df16": cfg(
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.35,
        support_max_arrival=20,
    ),
    "front_soft_40_030_18_df17": cfg(
        enable_front_support=True,
        support_distance_factor=1.7,
        support_min_available=40,
        support_fraction=0.30,
        support_max_arrival=18,
    ),
    "front_soft_45_025_16_df18": cfg(
        enable_front_support=True,
        support_distance_factor=1.8,
        support_min_available=45,
        support_fraction=0.25,
        support_max_arrival=16,
    ),
    "front_soft_30_030_20_df16": cfg(
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=30,
        support_fraction=0.30,
        support_max_arrival=20,
    ),
    "front_soft_35_025_20_df16": cfg(
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.25,
        support_max_arrival=20,
    ),
    "front_medium_28_050_28_df145": cfg(
        enable_front_support=True,
        support_distance_factor=1.45,
        support_min_available=28,
        support_fraction=0.50,
        support_max_arrival=28,
    ),
    "front_old_20_065_35_df13": cfg(enable_front_support=True),
}

RL_SCORE_GRID_VARIANTS = {
    "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
    **{
        f"rl_score_w{int(weight * 100):03d}": from_base(
            PUBLIC_LOOP_CONTROL,
            use_rl_target_score=True,
            rl_score_weight=weight,
        )
        for weight in [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.65, 0.80, 1.00]
    },
}

ADDITIVE_VARIANTS = {
    "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
    "custom_loop_public_score": PUBLIC_LOOP_CONTROL.to_agent_kwargs(),
    "add_late_filter": from_base(PUBLIC_LOOP_CONTROL, use_late_filter=True),
    "add_comet_targets": from_base(PUBLIC_LOOP_CONTROL, include_comet_targets=True),
    "add_comet_roi_filter": from_base(
        PUBLIC_LOOP_CONTROL,
        include_comet_targets=True,
        use_comet_roi_filter=True,
    ),
    "add_rl_score_w050": RL_SCORE_GRID_VARIANTS["rl_score_w050"],
    "front_support_soft": FRONT_SUPPORT_GRID_VARIANTS["front_soft_35_035_20_df16"],
    "front_support_medium": FRONT_SUPPORT_GRID_VARIANTS["front_medium_28_050_28_df145"],
    "front_support_old": FRONT_SUPPORT_GRID_VARIANTS["front_old_20_065_35_df13"],
}

COMBINED_PROMISING_VARIANTS = {
    "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
    "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
    "public_candidates_2": cfg(target_candidate_limit=2),
    "front_soft": FRONT_SUPPORT_GRID_VARIANTS["front_soft_35_035_20_df16"],
    "candidates2_front_soft": cfg(
        target_candidate_limit=2,
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.35,
        support_max_arrival=20,
    ),
    "candidates2_front_softer": cfg(
        target_candidate_limit=2,
        enable_front_support=True,
        support_distance_factor=1.7,
        support_min_available=40,
        support_fraction=0.30,
        support_max_arrival=18,
    ),
    "candidates2_front_tiny": cfg(
        target_candidate_limit=2,
        enable_front_support=True,
        support_distance_factor=1.8,
        support_min_available=45,
        support_fraction=0.25,
        support_max_arrival=16,
    ),
    "candidates2_min12": cfg(target_candidate_limit=2, min_ships_mine_attack=12),
    "candidates2_enemy_buffer4": cfg(
        target_candidate_limit=2,
        enemy_owned_production_buffer_turns=4,
    ),
    "candidates2_front_min12": cfg(
        target_candidate_limit=2,
        min_ships_mine_attack=12,
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.35,
        support_max_arrival=20,
    ),
    "front_soft_candidates4": cfg(
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.35,
        support_max_arrival=20,
        target_candidate_limit=4,
    ),
    "front_soft_min_attack8": cfg(
        enable_front_support=True,
        support_distance_factor=1.6,
        support_min_available=35,
        support_fraction=0.35,
        support_max_arrival=20,
        min_ships_mine_attack=8,
    ),
}

HISTORICAL_BEST_VARIANTS = {
    "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
    "pre_contested_regular": PRE_CONTESTED_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_arrival_regular": PRE_ARRIVAL_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_value_defense_regular": PRE_VALUE_DEFENSE_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_holdability_regular": PRE_HOLDABILITY_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_launch_regular": PRE_LAUNCH_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_early_neutral_regular": PRE_EARLY_NEUTRAL_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_reaction_margin_regular": PRE_REACTION_MARGIN_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_dynamic_cap_regular": PRE_DYNAMIC_CAP_REGULAR_CONFIG.to_agent_kwargs(),
    "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
    "launch_targets4_age8_regular": LAUNCH_TARGETS4_AGE8_REGULAR_CONFIG.to_agent_kwargs(),
    "recent_loss_recapture_prod4_b40_regular": RECENT_LOSS_RECAPTURE_PROD4_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e180_w50_b40_regular": RECAPTURE_S45_E180_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e160_w50_b40_regular": RECAPTURE_S45_E160_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_soft_4p_regular": MP_SOFT_4P_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_soft_leader0_4p_regular": MP_SOFT_LEADER0_4P_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_regular": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_w50_s0_e500_b35_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_W50_S0_E500_B35_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p2src_w002_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P2SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_lead10_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_lead10_p4earlysrc_s35_80_t120_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_P4EARLYSRC_S35_80_T120_REGULAR_CONFIG.to_agent_kwargs(),
    "orbit_wars_regular_p4src_w002_lead10_slim_20260517": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_REGULAR_CONFIG.to_agent_kwargs(),
    "orbit_wars_regular_p4src_w002_lead10_p4earlysrc_s35_80_t120_slim_20260518": PRE_TAIL_REGULAR_CONFIG.to_agent_kwargs(),
    "orbit_wars_regular_p4src_w002_lead10_p4earlysrc_s35_80_t120_tail_m2_max14_net7_slim_20260518": TAIL_M2_MAX14_NET7_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "orbit_wars_regular_p4src_w002_lead10_p4earlysrc_s35_80_t120_tail_m2_max14_net7_overpay4_slim_20260518": TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "orbit_wars_regular_p4src_w002_lead10_p4earlysrc_s35_80_t120_tail_m2_max14_net7_overpay4_p4lowhome_active4_slim_20260518": TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_regular_candidate": TAIL_M2_MAX14_NET7_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_overpay4_regular_candidate": TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_overpay4_p4lowhome_active4_regular": TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG.to_agent_kwargs(),
    "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
}

MYREPLAY_FOLLOWUP_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "previous_best_mp_local3_neu5": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
    "opening_trickle_4p_s30_min4": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_step_limit=30,
        opening_trickle_min_ships=4,
    ),
    "opening_trickle_4p_s30_min5": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_step_limit=30,
        opening_trickle_min_ships=5,
    ),
    "opening_trickle_4p_s40_min5": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_step_limit=40,
        opening_trickle_min_ships=5,
    ),
    "opening_trickle_4p_s40_min6": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_step_limit=40,
        opening_trickle_min_ships=6,
    ),
    "endgame_dump_s470_min20": from_base(
        REGULAR_CONFIG,
        enable_endgame_fleet_dump=True,
        endgame_dump_min_step=470,
        endgame_dump_min_ships=20,
    ),
    "endgame_dump_s480_min20": from_base(
        REGULAR_CONFIG,
        enable_endgame_fleet_dump=True,
        endgame_dump_min_step=480,
        endgame_dump_min_ships=20,
    ),
    "endgame_dump_s470_min30": from_base(
        REGULAR_CONFIG,
        enable_endgame_fleet_dump=True,
        endgame_dump_min_step=470,
        endgame_dump_min_ships=30,
    ),
}

RECENT_LOSS_REPLAY_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "previous_best_mp_local3_neu5": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
    "arrival_under_attack_m0": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_margin=0,
    ),
    "arrival_under_attack_m6": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_margin=6,
    ),
    "arrival_under_attack_s60_m0": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_min_step=60,
        under_attack_availability_margin=0,
    ),
    "arrival_under_attack_s70_m0": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_min_step=70,
        under_attack_availability_margin=0,
    ),
    "capture_hold_m4": from_base(
        REGULAR_CONFIG,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=4,
    ),
    "capture_hold_m8": from_base(
        REGULAR_CONFIG,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=8,
    ),
    "arrival_m0_plus_hold_m4": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_margin=0,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=4,
    ),
    "arrival_m6_plus_hold_m4": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_margin=6,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=4,
    ),
    "arrival_s70_m0_plus_hold_m4": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_min_step=70,
        under_attack_availability_margin=0,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=4,
    ),
}

RECENT_LOSS_VALIDATE_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "previous_best_mp_local3_neu5": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
    "capture_hold_m4": from_base(
        REGULAR_CONFIG,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=4,
    ),
    "capture_hold_m8": from_base(
        REGULAR_CONFIG,
        enable_capture_hold_margin_gate=True,
        capture_hold_margin=8,
    ),
}

MYREPLAY_REPAIR_VARIANTS = {
    "regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG.to_agent_kwargs(),
    "territory_penalty10_margin4": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=10.0,
        opening_territory_enemy_closer_margin=4.0,
    ),
    "territory_penalty20_margin6": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=20.0,
        opening_territory_enemy_closer_margin=6.0,
    ),
    "territory_penalty30_margin8": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=30.0,
        opening_territory_enemy_closer_margin=8.0,
    ),
    "opening_hold_base4_prod2": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=2,
    ),
    "opening_hold_base6_prod2": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_base_margin=6,
        opening_hold_prod_turns=2,
    ),
    "territory20_plus_opening_hold4": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=20.0,
        opening_territory_enemy_closer_margin=6.0,
        enable_opening_neutral_hold_margin=True,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=2,
    ),
    "home_anchor_4p_after10_prod2": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_min_after=10,
        home_anchor_prod_turns_after=2,
    ),
    "home_anchor_4p_after14_prod2": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_min_after=14,
        home_anchor_prod_turns_after=2,
    ),
    "mid_border_after12_prod2": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_midgame_border_source_reserve=True,
        midgame_border_min_after=12,
        midgame_border_prod_turns_after=2,
    ),
    "territory20_plus_anchor10": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=20.0,
        opening_territory_enemy_closer_margin=6.0,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_min_after=10,
        home_anchor_prod_turns_after=2,
    ),
}

MYREPLAY_TERRITORY_VALIDATE_VARIANTS = {
    "regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG.to_agent_kwargs(),
    "territory_penalty20_margin6": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=20.0,
        opening_territory_enemy_closer_margin=6.0,
    ),
    "territory_penalty30_margin8": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG,
        enable_opening_neutral_territory_score=True,
        opening_territory_penalty=30.0,
        opening_territory_enemy_closer_margin=8.0,
    ),
}

MYREPLAY_TERRITORY_FOLLOWUP_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "terr30_step35": from_base(
        REGULAR_CONFIG,
        opening_territory_step_limit=35,
    ),
    "terr30_step55": from_base(
        REGULAR_CONFIG,
        opening_territory_step_limit=55,
    ),
    "terr30_margin10": from_base(
        REGULAR_CONFIG,
        opening_territory_enemy_closer_margin=10.0,
    ),
    "terr30_margin12": from_base(
        REGULAR_CONFIG,
        opening_territory_enemy_closer_margin=12.0,
    ),
    "terr40_margin8": from_base(
        REGULAR_CONFIG,
        opening_territory_penalty=40.0,
    ),
    "terr20_margin8": from_base(
        REGULAR_CONFIG,
        opening_territory_penalty=20.0,
    ),
    "terr30_prod2": from_base(
        REGULAR_CONFIG,
        opening_territory_prod_scale=2.0,
    ),
    "terr30_prod6": from_base(
        REGULAR_CONFIG,
        opening_territory_prod_scale=6.0,
    ),
    "terr30_safe_gap6": from_base(
        REGULAR_CONFIG,
        opening_territory_allow_if_safe_gap=6,
    ),
    "terr30_safe_gap14": from_base(
        REGULAR_CONFIG,
        opening_territory_allow_if_safe_gap=14,
    ),
    "terr40_margin10_step55": from_base(
        REGULAR_CONFIG,
        opening_territory_penalty=40.0,
        opening_territory_enemy_closer_margin=10.0,
        opening_territory_step_limit=55,
    ),
    "home_anchor_soft_late": from_base(
        REGULAR_CONFIG,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_step_min=60,
        home_anchor_step_max=130,
        home_anchor_home_radius=45.0,
        home_anchor_min_after=6,
        home_anchor_prod_turns_after=1,
        home_anchor_front_threat_bonus=4,
    ),
    "mid_border_soft_late": from_base(
        REGULAR_CONFIG,
        enable_midgame_border_source_reserve=True,
        midgame_border_step_min=70,
        midgame_border_step_max=130,
        midgame_border_enemy_radius=45.0,
        midgame_border_min_after=6,
        midgame_border_prod_turns_after=1,
        midgame_border_threat_margin=4,
    ),
    "terr_margin10_plus_home_anchor_soft": from_base(
        REGULAR_CONFIG,
        opening_territory_enemy_closer_margin=10.0,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_step_min=60,
        home_anchor_step_max=130,
        home_anchor_home_radius=45.0,
        home_anchor_min_after=6,
        home_anchor_prod_turns_after=1,
        home_anchor_front_threat_bonus=4,
    ),
}

MYREPLAY_TERRITORY_TOP_VALIDATE_VARIANTS = {
    "regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG.to_agent_kwargs(),
    "terr30_step35": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG,
        opening_territory_step_limit=35,
    ),
    "home_anchor_soft_late": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_step_min=60,
        home_anchor_step_max=130,
        home_anchor_home_radius=45.0,
        home_anchor_min_after=6,
        home_anchor_prod_turns_after=1,
        home_anchor_front_threat_bonus=4,
    ),
    "terr30_step35_plus_home_anchor_soft": from_base(
        MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG,
        opening_territory_step_limit=35,
        enable_home_anchor_source_reserve=True,
        home_anchor_min_active_players=4,
        home_anchor_step_min=60,
        home_anchor_step_max=130,
        home_anchor_home_radius=45.0,
        home_anchor_min_after=6,
        home_anchor_prod_turns_after=1,
        home_anchor_front_threat_bonus=4,
    ),
}

MYREPLAY_UNUSED_FEATURE_PROBE_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "opening_hold_soft": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=4,
        opening_hold_step_limit=35,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=2,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=2,
    ),
    "opening_hold_medium": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=4,
        opening_hold_step_limit=45,
        opening_hold_min_production=3.0,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=4,
    ),
    "source_penalty_late90_w003": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=90,
        source_threat_send_radius=35.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.50,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.03,
    ),
    "source_penalty_late110_w003": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=110,
        source_threat_send_radius=35.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.50,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.03,
    ),
    "local_gate_late_soft": from_base(
        REGULAR_CONFIG,
        enable_local_source_defense_gate=True,
        local_source_defense_gate_min_active_players=4,
        local_source_defense_gate_min_step=90,
        local_source_defense_gate_min_production=4.0,
        local_source_defense_gate_front_distance=32.0,
        local_source_defense_gate_margin=2,
        local_source_defense_gate_use_arrival_production=True,
    ),
    "local_gate_late_no_arrival": from_base(
        REGULAR_CONFIG,
        enable_local_source_defense_gate=True,
        local_source_defense_gate_min_active_players=4,
        local_source_defense_gate_min_step=90,
        local_source_defense_gate_min_production=4.0,
        local_source_defense_gate_front_distance=32.0,
        local_source_defense_gate_margin=2,
        local_source_defense_gate_use_arrival_production=False,
    ),
    "global_priority_roi005_arr02": from_base(
        REGULAR_CONFIG,
        enable_global_attack_priority=True,
        global_attack_roi_weight=0.05,
        global_attack_arrival_penalty=0.20,
        global_attack_max_failed_pairs=96,
    ),
    "global_priority_arr03": from_base(
        REGULAR_CONFIG,
        enable_global_attack_priority=True,
        global_attack_roi_weight=0.0,
        global_attack_arrival_penalty=0.30,
        global_attack_max_failed_pairs=96,
    ),
    "front_support_tiny": from_base(
        REGULAR_CONFIG,
        enable_front_support=True,
        support_distance_factor=1.8,
        support_min_available=45,
        support_fraction=0.20,
        support_min_send=10,
        support_max_arrival=16,
    ),
    "under_attack_availability_soft": from_base(
        REGULAR_CONFIG,
        enable_arrival_based_under_attack_availability=True,
        under_attack_availability_min_step=40,
        under_attack_availability_margin=2,
        under_attack_availability_horizon=45,
    ),
    "home_anchor_plus_source_penalty": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=90,
        source_threat_send_radius=35.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.50,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.03,
    ),
}

MYREPLAY_PROACTIVE_REPAIR_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "proactive_4p_recent_s90_src10_m24": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=35,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=180,
        proactive_defense_min_prod_diff=1.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.95,
        proactive_defense_max_send=24,
        proactive_defense_base_margin=4,
        proactive_defense_prod_turns=2,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=20.0,
    ),
    "proactive_4p_recent_s110_src10_m24": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=35,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=110,
        proactive_defense_max_step=190,
        proactive_defense_min_prod_diff=1.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.95,
        proactive_defense_max_send=24,
        proactive_defense_base_margin=4,
        proactive_defense_prod_turns=2,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=20.0,
    ),
    "proactive_4p_recent_s90_src15_m20": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=35,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=180,
        proactive_defense_min_prod_diff=1.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=1.0,
        proactive_defense_max_send=20,
        proactive_defense_base_margin=3,
        proactive_defense_prod_turns=2,
        proactive_defense_source_min_after=15,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=30.0,
    ),
    "proactive_4p_adv_s90_src10_no_recent": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=170,
        proactive_defense_min_prod_diff=2.0,
        proactive_defense_min_planet_diff=1,
        proactive_defense_min_ship_ratio=1.05,
        proactive_defense_max_send=20,
        proactive_defense_base_margin=3,
        proactive_defense_prod_turns=1,
        proactive_defense_source_min_after=12,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=35.0,
    ),
    "proactive_4p_recent_s120_src10_m20": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=45,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=120,
        proactive_defense_max_step=210,
        proactive_defense_min_prod_diff=1.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.95,
        proactive_defense_max_send=20,
        proactive_defense_base_margin=3,
        proactive_defense_prod_turns=2,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=20.0,
    ),
}

MYREPLAY_RECAPTURE_HOLD_REPAIR_VARIANTS = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "capture_hold_post_window": from_base(
        REGULAR_CONFIG,
        capture_hold_use_post_capture_window=True,
    ),
    "recent_loss_hold_m8_post": from_base(
        REGULAR_CONFIG,
        capture_hold_use_post_capture_window=True,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=4,
        recent_loss_recapture_hold_min_step=55,
        recent_loss_recapture_hold_max_step=170,
        recent_loss_recapture_hold_window=50,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
    ),
    "recent_loss_hold_m12_post": from_base(
        REGULAR_CONFIG,
        capture_hold_use_post_capture_window=True,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=4,
        recent_loss_recapture_hold_min_step=55,
        recent_loss_recapture_hold_max_step=170,
        recent_loss_recapture_hold_window=50,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=12,
    ),
    "recent_loss_hold_w70_m8_post": from_base(
        REGULAR_CONFIG,
        capture_hold_use_post_capture_window=True,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=4,
        recent_loss_recapture_hold_min_step=45,
        recent_loss_recapture_hold_max_step=190,
        recent_loss_recapture_hold_window=70,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
    ),
    "recent_loss_hold_m8_no_extra": from_base(
        REGULAR_CONFIG,
        capture_hold_use_post_capture_window=True,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=4,
        recent_loss_recapture_hold_min_step=55,
        recent_loss_recapture_hold_max_step=170,
        recent_loss_recapture_hold_window=50,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
        recent_loss_recapture_hold_allow_extra_send=False,
    ),
}

CHAMPION_OPPONENT_VARIANTS = {
    "public_original": None,
    "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_overpay4_p4lowhome_active4_regular": TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_overpay4_regular_candidate": TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "tail_m2_max14_net7_regular_candidate": TAIL_M2_MAX14_NET7_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p2src_w002_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P2SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_lead10_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_lead10_p4earlysrc_s35_80_t120_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P4SRC_W002_LEAD10_P4EARLYSRC_S35_80_T120_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_w50_s0_e500_b35_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_W50_S0_E500_B35_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_terr30_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_hold4_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_comet12_path4p_regular": MP_LOCAL3_NEU5_COMET12_PATH4P_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_local3_neu5_regular": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_soft_leader0_4p_regular": MP_SOFT_LEADER0_4P_REGULAR_CONFIG.to_agent_kwargs(),
    "mp_soft_4p_regular": MP_SOFT_4P_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e160_w50_b40_regular": RECAPTURE_S45_E160_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e180_w50_b40_regular": RECAPTURE_S45_E180_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "recent_loss_recapture_prod4_b40_regular": RECENT_LOSS_RECAPTURE_PROD4_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "launch_targets4_age8_regular": LAUNCH_TARGETS4_AGE8_REGULAR_CONFIG.to_agent_kwargs(),
    "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
    "pre_dynamic_cap_regular": PRE_DYNAMIC_CAP_REGULAR_CONFIG.to_agent_kwargs(),
}

ABLATION_SUITES = {
    "additive": ADDITIVE_VARIANTS,
    "public_knobs": PUBLIC_KNOB_VARIANTS,
    "front_support": FRONT_SUPPORT_GRID_VARIANTS,
    "rl_score": RL_SCORE_GRID_VARIANTS,
    "combined": COMBINED_PROMISING_VARIANTS,
    "historical_best": HISTORICAL_BEST_VARIANTS,
    "myreplay_followup": MYREPLAY_FOLLOWUP_VARIANTS,
    "recent_loss_replay": RECENT_LOSS_REPLAY_VARIANTS,
    "recent_loss_validate": RECENT_LOSS_VALIDATE_VARIANTS,
    "myreplay_repair": MYREPLAY_REPAIR_VARIANTS,
    "myreplay_territory_validate": MYREPLAY_TERRITORY_VALIDATE_VARIANTS,
    "myreplay_territory_followup": MYREPLAY_TERRITORY_FOLLOWUP_VARIANTS,
    "myreplay_territory_top_validate": MYREPLAY_TERRITORY_TOP_VALIDATE_VARIANTS,
    "myreplay_unused_feature_probe": MYREPLAY_UNUSED_FEATURE_PROBE_VARIANTS,
    "myreplay_proactive_repair": MYREPLAY_PROACTIVE_REPAIR_VARIANTS,
    "myreplay_recapture_hold_repair": MYREPLAY_RECAPTURE_HOLD_REPAIR_VARIANTS,
    "regular_verify": {
        "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
        "candidate2": cfg(target_candidate_limit=2),
        "pre_contested_regular": PRE_CONTESTED_REGULAR_CONFIG.to_agent_kwargs(),
        "pre_arrival_regular": PRE_ARRIVAL_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
    },
    "vs_regular_additions": {
        "regular_mirror": REGULAR_CONFIG.to_agent_kwargs(),
        "pre_arrival_regular": PRE_ARRIVAL_REGULAR_CONFIG.to_agent_kwargs(),
        "pre_contested_regular": PRE_CONTESTED_REGULAR_CONFIG.to_agent_kwargs(),
        "contested_light": from_base(
            PRE_CONTESTED_REGULAR_CONFIG,
            enable_contested_target_adjustment=True,
            contested_arrival_margin=2,
            contested_enemy_weight=0.75,
            contested_friendly_credit=0.50,
        ),
        "contested_default": from_base(
            PRE_CONTESTED_REGULAR_CONFIG,
            enable_contested_target_adjustment=True,
        ),
        "contested_strict": from_base(
            PRE_CONTESTED_REGULAR_CONFIG,
            enable_contested_target_adjustment=True,
            contested_arrival_margin=5,
            contested_enemy_weight=1.25,
            contested_friendly_credit=1.00,
        ),
        "contested_no_skip": from_base(
            PRE_CONTESTED_REGULAR_CONFIG,
            enable_contested_target_adjustment=True,
            contested_skip_friendly_covered=False,
        ),
    },
    "arrival_production": {
        "regular_fixed_buffer": PRE_ARRIVAL_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "arrival_exact": from_base(
            PRE_ARRIVAL_REGULAR_CONFIG,
            use_arrival_based_enemy_production=True,
            arrival_enemy_production_safety_turns=0,
        ),
        "arrival_plus1": from_base(
            PRE_ARRIVAL_REGULAR_CONFIG,
            use_arrival_based_enemy_production=True,
            arrival_enemy_production_safety_turns=1,
        ),
        "arrival_plus2": from_base(
            PRE_ARRIVAL_REGULAR_CONFIG,
            use_arrival_based_enemy_production=True,
            arrival_enemy_production_safety_turns=2,
        ),
        "arrival_plus3": from_base(
            PRE_ARRIVAL_REGULAR_CONFIG,
            use_arrival_based_enemy_production=True,
            arrival_enemy_production_safety_turns=3,
        ),
    },
    "arrival_cap_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "arrival_cap30": from_base(REGULAR_CONFIG, arrival_enemy_production_max_turns=30),
        "arrival_cap45": from_base(REGULAR_CONFIG, arrival_enemy_production_max_turns=45),
        "arrival_cap60": from_base(REGULAR_CONFIG, arrival_enemy_production_max_turns=60),
        "arrival_safety0_cap45": from_base(
            REGULAR_CONFIG,
            arrival_enemy_production_safety_turns=0,
            arrival_enemy_production_max_turns=45,
        ),
        "arrival_safety2_cap45": from_base(
            REGULAR_CONFIG,
            arrival_enemy_production_safety_turns=2,
            arrival_enemy_production_max_turns=45,
        ),
    },
    "dynamic_posture": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "dynamic_default": from_base(REGULAR_CONFIG, enable_dynamic_posture=True),
        "dynamic_light_reserve": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=2,
        ),
        "dynamic_strong_reserve": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            defensive_reserve_turns=4,
            defensive_min_garrison=10,
            defensive_high_prod_bonus=6,
        ),
        "dynamic_more_aggressive": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            aggressive_min_attack_delta=-3,
            aggressive_target_candidate_bonus=2,
        ),
        "dynamic_late_sensitive": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_late_step=180,
        ),
    },
    "champion_dynamic_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "dynamic_more_aggressive": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            aggressive_min_attack_delta=-3,
            aggressive_target_candidate_bonus=2,
        ),
        "dynamic_aggr_bonus1": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            aggressive_min_attack_delta=-3,
            aggressive_target_candidate_bonus=1,
        ),
        "dynamic_aggr_delta2_bonus2": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            aggressive_min_attack_delta=-2,
            aggressive_target_candidate_bonus=2,
        ),
        "dynamic_aggr_delta4_bonus2": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            aggressive_min_attack_delta=-4,
            aggressive_target_candidate_bonus=2,
        ),
        "dynamic_aggr_late180": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_late_step=180,
            aggressive_min_attack_delta=-3,
            aggressive_target_candidate_bonus=2,
        ),
        "dynamic_aggr_late320": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_late_step=320,
            aggressive_min_attack_delta=-3,
            aggressive_target_candidate_bonus=2,
        ),
    },
    "recent_loss_recapture": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recapture_b20_w40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_bonus=20.0,
            recent_loss_recapture_window=40,
        ),
        "recapture_b30_w40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_bonus=30.0,
            recent_loss_recapture_window=40,
        ),
        "recapture_b40_w60": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_b60_w60": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_bonus=60.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod4_b40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_mid_b40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_step=40,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod_weight8": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_bonus=30.0,
            recent_loss_recapture_window=60,
            recent_loss_recapture_prod_weight=8.0,
        ),
    },
    "recent_loss_recapture_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recapture_prod4_b40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod4_b30": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=30.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod4_b50": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=50.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod4_w40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=40,
        ),
        "recapture_prod4_w80": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=80,
        ),
        "recapture_prod45_b40": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.5,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
        "recapture_prod4_mid": from_base(
            REGULAR_CONFIG,
            enable_recent_loss_recapture_bias=True,
            recent_loss_recapture_min_production=4.0,
            recent_loss_recapture_min_step=40,
            recent_loss_recapture_max_step=260,
            recent_loss_recapture_bonus=40.0,
            recent_loss_recapture_window=60,
        ),
    },
    "champion_loss_repair": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "source_reserve_prod4_front35": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=40,
            local_reserve_max_step=180,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=35.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=6,
            local_reserve_front_bonus=6,
        ),
        "source_reserve_prod4_front45": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=40,
            local_reserve_max_step=180,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=45.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=6,
            local_reserve_front_bonus=8,
        ),
        "source_reserve_prod5_front35": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=40,
            local_reserve_max_step=180,
            local_reserve_min_production=5.0,
            local_reserve_enemy_distance=35.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=6,
            local_reserve_front_bonus=6,
        ),
        "source_reserve_late_soft": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=60,
            local_reserve_max_step=220,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=35.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=4,
            local_reserve_front_bonus=4,
        ),
        "value_defense_h80_send55": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=80,
            value_defense_max_send=55,
            value_defense_min_margin=10,
        ),
        "value_defense_h70_margin12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
        "recapture_mid_only": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
        ),
        "recapture_source_reserve": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=40,
            local_reserve_max_step=180,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=35.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=6,
            local_reserve_front_bonus=6,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
        ),
    },
    "champion_loss_repair_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recap_s45_e220_w60_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=60,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s60_e220_w60_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=60,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=60,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s70_e220_w60_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=70,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=60,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s60_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=60,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e220_w80_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=80,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_s45_e220_w40_b45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=40,
            recent_loss_recapture_bonus=45.0,
        ),
        "vd_h65_send45_m10": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=65,
            value_defense_max_send=45,
            value_defense_min_margin=10,
        ),
        "vd_h70_send45_m12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
        "vd_h75_send50_m12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=75,
            value_defense_max_send=50,
            value_defense_min_margin=12,
        ),
        "vd_h70_send50_m10": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=50,
            value_defense_min_margin=10,
        ),
        "recap45_vd70_m12": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
        "recap60_vd70_m12": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=60,
            recent_loss_recapture_max_step=220,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
    },
    "champion_loss_repair_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recap_s60_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=60,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "vd_h70_send45_m12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
        "recap_s45_e220_w60_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=220,
            recent_loss_recapture_window=60,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap60_vd70_m12": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=60,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
    },
    "champion_recap_window_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recap_s30_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=30,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s35_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=35,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s40_e180_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=40,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e160_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=160,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e200_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=200,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e180_w40_b45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=40,
            recent_loss_recapture_bonus=45.0,
        ),
        "recap_s45_e180_w60_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=60,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_s45_e180_w50_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_s45_e180_w50_b45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=45.0,
        ),
        "recap_prod_weight7": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_prod_weight=7.0,
        ),
        "recap_prod_weight9": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_prod_weight=9.0,
        ),
        "recap_min_prod35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_production=3.5,
        ),
        "recap_min_prod45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_production=4.5,
        ),
        "recap_vd_h65_m10": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=65,
            value_defense_max_send=45,
            value_defense_min_margin=10,
        ),
        "recap_vd_h70_m12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
    },
    "champion_recap_window_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recap_s45_e160_w50_b40": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=160,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=40.0,
        ),
        "recap_s45_e180_w50_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_step=45,
            recent_loss_recapture_max_step=180,
            recent_loss_recapture_window=50,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_vd_h65_m10": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=65,
            value_defense_max_send=45,
            value_defense_min_margin=10,
        ),
        "recap_vd_h70_m12": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
    },
    "global_attack_priority": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "global_priority_plain": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
        ),
        "global_priority_roi05": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
            global_attack_roi_weight=0.5,
        ),
        "global_priority_roi10": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
            global_attack_roi_weight=1.0,
        ),
        "global_priority_eta005": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
            global_attack_arrival_penalty=0.05,
        ),
        "global_priority_roi05_eta005": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
            global_attack_roi_weight=0.5,
            global_attack_arrival_penalty=0.05,
        ),
        "global_priority_roi10_eta010": from_base(
            REGULAR_CONFIG,
            enable_global_attack_priority=True,
            global_attack_roi_weight=1.0,
            global_attack_arrival_penalty=0.10,
        ),
    },
    "multiplayer_diplomacy": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_local_enemy": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=8.0,
            multiplayer_leader_prod_bonus=0.0,
            multiplayer_neutral_bonus=0.0,
        ),
        "mp_local_leader": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=8.0,
            multiplayer_leader_prod_bonus=2.0,
            multiplayer_neutral_bonus=0.0,
        ),
        "mp_neutral_first": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_soft": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_neutral_n6_local0": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=0.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
    },
    "multiplayer_diplomacy_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_neutral_first": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_soft": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_neutral_n6_local0": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=0.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
    },
    "multiplayer_diplomacy_search": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_neutral_n4_far18": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=4.0,
        ),
        "mp_neutral_n6_far18": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_neutral_n8_far18": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=8.0,
        ),
        "mp_neutral_n6_far10": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_neutral_n6_far25": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=25.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_neutral_n6_local0": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=0.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_neutral_n6_leader0": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=18.0,
            multiplayer_local_enemy_bonus=4.0,
            multiplayer_leader_prod_bonus=0.0,
            multiplayer_neutral_bonus=6.0,
        ),
        "mp_neutral_n8_far25_local0": from_base(
            REGULAR_CONFIG,
            enable_multiplayer_diplomacy_score=True,
            multiplayer_far_enemy_penalty=25.0,
            multiplayer_local_enemy_bonus=0.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=8.0,
        ),
    },
    "late_lead_guard": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "duel_guard_soft": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
        ),
        "duel_guard_mid": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_prod_lead=3.0,
            posture_defensive_ship_ratio=0.95,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=3,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=8,
            defensive_high_prod_bonus=6,
        ),
        "duel_guard_strong": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_prod_lead=2.0,
            posture_defensive_ship_ratio=0.90,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=4,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=3,
            defensive_min_garrison=8,
            defensive_high_prod_bonus=8,
        ),
        "duel_guard_late_source": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=120,
            local_reserve_max_step=260,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=45.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=8,
            local_reserve_front_bonus=10,
        ),
        "duel_guard_mid_source": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_prod_lead=3.0,
            posture_defensive_ship_ratio=0.95,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=3,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=8,
            defensive_high_prod_bonus=6,
            enable_local_source_reserve=True,
            local_reserve_min_step=120,
            local_reserve_max_step=260,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=45.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=8,
            local_reserve_front_bonus=10,
        ),
        "duel_guard_value_h80": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=80,
            value_defense_max_send=55,
            value_defense_min_margin=12,
        ),
    },
    "late_lead_guard_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "duel_guard_soft": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
        ),
        "duel_guard_value_h80": from_base(
            REGULAR_CONFIG,
            value_defense_horizon=80,
            value_defense_max_send=55,
            value_defense_min_margin=12,
        ),
        "mp_value_h80": from_base(
            REGULAR_CONFIG,
            value_defense_multiplayer_min_active_players=4,
            value_defense_multiplayer_horizon=80,
            value_defense_multiplayer_max_send=55,
            value_defense_multiplayer_min_margin=12,
        ),
    },
    "late_lead_guard_timing": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "duel_guard_s120": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_min_step=120,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
        ),
        "duel_guard_s150": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_min_step=150,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
        ),
        "duel_guard_s180": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_min_step=180,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
        ),
        "duel_guard_s150_weak": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_min_step=150,
            posture_defensive_prod_lead=5.0,
            posture_defensive_ship_ratio=1.05,
            posture_defensive_planet_lead=1,
            defensive_min_attack_delta=1,
            defensive_target_candidate_delta=0,
            defensive_reserve_turns=1,
            defensive_min_garrison=4,
            defensive_high_prod_bonus=2,
        ),
        "duel_guard_s150_vd70": from_base(
            REGULAR_CONFIG,
            enable_dynamic_posture=True,
            posture_max_active_players=2,
            posture_defensive_min_step=150,
            posture_defensive_prod_lead=4.0,
            posture_defensive_ship_ratio=1.00,
            posture_defensive_planet_lead=0,
            defensive_min_attack_delta=2,
            defensive_target_candidate_delta=-1,
            defensive_reserve_turns=2,
            defensive_min_garrison=6,
            defensive_high_prod_bonus=4,
            value_defense_horizon=70,
            value_defense_max_send=45,
            value_defense_min_margin=12,
        ),
    },
    "current_holdability_search": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "hold_w025_r35": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.25,
            holdability_radius=35.0,
        ),
        "hold_w050_r30": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=30.0,
        ),
        "hold_w050_r40": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=40.0,
        ),
        "hold_w075_r35": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.75,
            holdability_radius=35.0,
        ),
        "hold_enemy_prod5": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=35.0,
            holdability_enemy_prod_weight=5.0,
        ),
        "hold_own_support": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=35.0,
            holdability_own_prod_weight=3.0,
            holdability_own_ship_weight=0.06,
        ),
        "early_neutral_more_safe": from_base(
            REGULAR_CONFIG,
            early_neutral_safe_bonus=28.0,
            early_neutral_contested_penalty=22.0,
            early_neutral_reaction_margin=4,
        ),
        "early_neutral_less_safe": from_base(
            REGULAR_CONFIG,
            early_neutral_safe_bonus=12.0,
            early_neutral_contested_penalty=10.0,
            early_neutral_reaction_margin=2,
        ),
    },
    "recapture_micro_search": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "recap_e150": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_max_step=150,
        ),
        "recap_e170": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_max_step=170,
        ),
        "recap_w45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_window=45,
        ),
        "recap_w55": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_window=55,
        ),
        "recap_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_b45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_bonus=45.0,
        ),
        "recap_prod45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_min_production=4.5,
        ),
        "recap_e170_b35": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_max_step=170,
            recent_loss_recapture_bonus=35.0,
        ),
        "recap_e150_w45": from_base(
            REGULAR_CONFIG,
            recent_loss_recapture_max_step=150,
            recent_loss_recapture_window=45,
        ),
    },
    "mp_soft_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_far08_local5_neu3": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=8.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_far12_local5_neu3": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=12.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_far10_local3_neu3": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=3.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_far10_local7_neu3": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=7.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_far10_local5_neu2": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=2.0,
        ),
        "mp_far10_local5_neu4": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=1.0,
            multiplayer_neutral_bonus=4.0,
        ),
        "mp_far10_local5_leader0": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=0.0,
            multiplayer_neutral_bonus=3.0,
        ),
        "mp_far10_local5_leader2": from_base(
            REGULAR_CONFIG,
            multiplayer_far_enemy_penalty=10.0,
            multiplayer_local_enemy_bonus=5.0,
            multiplayer_leader_prod_bonus=2.0,
            multiplayer_neutral_bonus=3.0,
        ),
    },
    "value_defense": {
        "pre_value_defense_regular": PRE_VALUE_DEFENSE_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "value_defense_default": REGULAR_CONFIG.to_agent_kwargs(),
        "value_defense_light": from_base(
            PRE_VALUE_DEFENSE_REGULAR_CONFIG,
            enable_value_defense=True,
            value_defense_buffer_turns=2,
            value_defense_min_margin=6,
            value_defense_max_send=28,
        ),
        "value_defense_strong": from_base(
            PRE_VALUE_DEFENSE_REGULAR_CONFIG,
            enable_value_defense=True,
            value_defense_buffer_turns=4,
            value_defense_min_margin=10,
            value_defense_max_send=55,
        ),
        "value_defense_prod2": from_base(
            PRE_VALUE_DEFENSE_REGULAR_CONFIG,
            enable_value_defense=True,
            value_defense_min_production=2.0,
        ),
        "value_defense_roi2": from_base(
            PRE_VALUE_DEFENSE_REGULAR_CONFIG,
            enable_value_defense=True,
            value_defense_roi_multiplier=2.0,
        ),
    },
    "local_posture": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "local_source_reserve": from_base(REGULAR_CONFIG, enable_local_source_reserve=True),
        "local_source_reserve_light": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_turns=1,
            local_reserve_min_garrison=4,
            local_reserve_front_bonus=4,
        ),
        "holdability_light": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.5,
        ),
        "holdability_strong": from_base(
            REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=1.2,
        ),
        "source_reserve_holdability": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            enable_holdability_target_score=True,
            holdability_weight=0.5,
        ),
    },
    "holdability_refine": {
        "pre_holdability_regular": PRE_HOLDABILITY_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "holdability_w025_r35": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.25,
            holdability_radius=35.0,
        ),
        "holdability_w050_r35": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=35.0,
        ),
        "holdability_w080_r35": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.80,
            holdability_radius=35.0,
        ),
        "holdability_w050_r25": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=25.0,
        ),
        "holdability_w050_r45": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_radius=45.0,
        ),
        "holdability_enemy_prod_heavy": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_enemy_prod_weight=6.0,
        ),
        "holdability_own_support_heavy": from_base(
            PRE_HOLDABILITY_REGULAR_CONFIG,
            enable_holdability_target_score=True,
            holdability_weight=0.50,
            holdability_own_prod_weight=3.0,
            holdability_own_ship_weight=0.06,
        ),
    },
    "proactive_defense": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "predictive_light": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_base_margin=6,
            proactive_defense_prod_turns=2,
            proactive_defense_enemy_launch_window=4,
            proactive_defense_enemy_reserve_turns=3,
            proactive_defense_enemy_send_fraction=0.75,
            proactive_defense_threat_slack=2,
            proactive_defense_max_send=18,
            proactive_defense_max_arrival=35,
        ),
        "predictive_default": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
        ),
        "predictive_urgent_only": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_enemy_launch_window=4,
            proactive_defense_threat_slack=0,
            proactive_defense_max_arrival=28,
            proactive_defense_max_send=20,
        ),
        "predictive_all_in": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_enemy_launch_window=8,
            proactive_defense_enemy_reserve_turns=0,
            proactive_defense_enemy_send_fraction=1.0,
            proactive_defense_threat_slack=8,
            proactive_defense_max_send=28,
        ),
        "predictive_larger_send": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=40,
        ),
        "predictive_two_targets": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_targets=2,
            proactive_defense_max_send=20,
        ),
        "predictive_wide_light": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_radius=55.0,
            proactive_defense_base_margin=6,
            proactive_defense_prod_turns=2,
            proactive_defense_enemy_launch_window=4,
            proactive_defense_enemy_reserve_turns=3,
            proactive_defense_enemy_send_fraction=0.75,
            proactive_defense_threat_slack=2,
            proactive_defense_max_send=18,
        ),
    },
    "proactive_predictive_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "predictive_send32": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=32,
        ),
        "predictive_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=40,
        ),
        "predictive_send50": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=50,
        ),
        "predictive_send40_roi2": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=40,
            proactive_defense_roi_multiplier=2.0,
        ),
        "predictive_send40_slack2": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=40,
            proactive_defense_threat_slack=2,
        ),
        "predictive_send40_slack10": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_max_send=40,
            proactive_defense_threat_slack=10,
        ),
    },
    "proactive_context_search": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "after_attack_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
        ),
        "after_attack_mid_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_step=60,
            proactive_defense_max_step=240,
        ),
        "after_attack_lead_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
        "after_attack_mid_lead_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_step=60,
            proactive_defense_max_step=240,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
        "after_attack_ship_lead_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_ship_ratio=1.05,
        ),
        "after_attack_high_value_only": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_min_production=4.0,
            proactive_defense_max_send=40,
        ),
        "after_attack_conservative_threat": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_enemy_launch_window=4,
            proactive_defense_enemy_reserve_turns=3,
            proactive_defense_enemy_send_fraction=0.75,
            proactive_defense_threat_slack=2,
        ),
        "before_attack_mid_lead_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=False,
            proactive_defense_max_send=40,
            proactive_defense_min_step=60,
            proactive_defense_max_step=240,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
    },
    "proactive_lead_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "lead_prod0_planet0_send35": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=35,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
        "lead_prod0_planet0_send40": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
        "lead_prod_neg2_planet0": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=-2.0,
            proactive_defense_min_planet_diff=0,
        ),
        "lead_prod3_planet0": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=3.0,
            proactive_defense_min_planet_diff=0,
        ),
        "lead_prod0_planet_neg1": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=-1,
        ),
        "lead_prod0_planet1": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=1,
        ),
        "lead_high_prod_only": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_min_production=4.0,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
        "lead_low_margin": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_base_margin=4,
            proactive_defense_prod_turns=2,
            proactive_defense_min_prod_diff=0.0,
            proactive_defense_min_planet_diff=0,
        ),
    },
    "proactive_lead_validate": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "lead_prod_neg2_planet0": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=-2.0,
            proactive_defense_min_planet_diff=0,
        ),
    },
    "proactive_trigger_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "prev_best_weak": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=-2.0,
            proactive_defense_min_planet_diff=0,
        ),
        "step90_prod_lead": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
        ),
        "step100_planet_lead": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=100,
            proactive_defense_max_send=40,
            proactive_defense_min_planet_diff=1,
        ),
        "step120_prod_or_planet_like": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=120,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_min_planet_diff=0,
        ),
        "source_after10": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
        ),
        "source_after15": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=15,
        ),
        "source_after10_prod1": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
            proactive_defense_source_prod_turns_after=1,
        ),
        "recent_capture35": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
        ),
        "recent_capture50_source10": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_recent_capture_window=50,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
        ),
        "strict_countergrab": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=100,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_min_planet_diff=1,
            proactive_defense_source_min_after=10,
            proactive_defense_source_prod_turns_after=1,
        ),
    },
    "proactive_roi_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "roi_prev_best": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=-2.0,
            proactive_defense_min_planet_diff=0,
        ),
        "roi_step90_prod_lead": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
        ),
        "roi_step90_recent35": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
        ),
        "roi_step90_recent60": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_recent_capture_window=60,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
        ),
        "roi_strict_countergrab": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=100,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_min_planet_diff=1,
        ),
        "roi_high_value": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_min_step=90,
            proactive_defense_min_production=4.0,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_enemy_roi_multiplier=1.50,
            proactive_defense_enemy_min_net_value=60.0,
        ),
        "roi_source_after10": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
        ),
        "roi_recent35_source10": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
        ),
        "roi_recent35_source_prod1": from_base(
            REGULAR_CONFIG,
            enable_proactive_value_defense=True,
            proactive_defense_after_attacks=True,
            proactive_defense_require_turn_attack=True,
            proactive_defense_require_recent_capture=True,
            proactive_defense_require_enemy_positive_roi=True,
            proactive_defense_recent_capture_window=35,
            proactive_defense_min_step=90,
            proactive_defense_max_send=40,
            proactive_defense_min_prod_diff=1.0,
            proactive_defense_source_min_after=10,
            proactive_defense_source_prod_turns_after=1,
        ),
    },
    "enemy_launch_punish": {
        "pre_launch_regular": PRE_LAUNCH_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "launch_default": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
        ),
        "launch_recent8": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_max_fleet_age=8,
        ),
        "launch_recent16": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_max_fleet_age=16,
        ),
        "launch_big20": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_min_outgoing=20,
        ),
        "launch_high_prod": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_min_production=3.0,
        ),
        "launch_bonus_soft": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_bonus_weight=0.25,
        ),
        "launch_bonus_strong": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_bonus_weight=0.80,
        ),
        "launch_three_targets": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_max_targets=3,
        ),
        "launch_candidate3": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            target_candidate_limit=3,
        ),
    },
    "enemy_launch_validate": {
        "pre_launch_regular": PRE_LAUNCH_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "launch_high_prod": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_min_production=3.0,
        ),
        "launch_three_targets": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_max_targets=3,
        ),
        "launch_bonus_soft": from_base(
            PRE_LAUNCH_REGULAR_CONFIG,
            enable_enemy_launch_punish=True,
            enemy_launch_punish_bonus_weight=0.25,
        ),
    },
    "early_expansion_refine": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "min_attack_10": from_base(REGULAR_CONFIG, min_ships_mine_attack=10),
        "min_attack_11": from_base(REGULAR_CONFIG, min_ships_mine_attack=11),
        "candidate3": from_base(REGULAR_CONFIG, target_candidate_limit=3),
        "holdability_soft": from_base(REGULAR_CONFIG, holdability_weight=0.25),
        "holdability_off": from_base(REGULAR_CONFIG, enable_holdability_target_score=False),
        "min10_holdability_soft": from_base(
            REGULAR_CONFIG,
            min_ships_mine_attack=10,
            holdability_weight=0.25,
        ),
        "min10_candidate3": from_base(
            REGULAR_CONFIG,
            min_ships_mine_attack=10,
            target_candidate_limit=3,
        ),
        "min11_candidate3": from_base(
            REGULAR_CONFIG,
            min_ships_mine_attack=11,
            target_candidate_limit=3,
        ),
    },
    "early_expansion_validate": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "holdability_soft": from_base(REGULAR_CONFIG, holdability_weight=0.25),
        "holdability_off": from_base(REGULAR_CONFIG, enable_holdability_target_score=False),
        "min_attack_11": from_base(REGULAR_CONFIG, min_ships_mine_attack=11),
    },
    "notebook_early_neutral": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "pre_early_neutral_regular": PRE_EARLY_NEUTRAL_REGULAR_CONFIG.to_agent_kwargs(),
        "early_neutral_bias": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
        ),
        "early_neutral_prod4": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_min_production=4.0,
        ),
        "early_neutral_static_heavy": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_static_multiplier=1.60,
        ),
        "early_neutral_safe_only": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_bonus=4.0,
            early_neutral_safe_bonus=20.0,
            early_neutral_contested_penalty=16.0,
        ),
        "early_neutral_relief_light": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_holdability_relief=0.75,
        ),
        "opening_rotating_filter": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_opening_rotating_neutral_filter=True,
        ),
        "early_bias_plus_rotating_filter": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            enable_opening_rotating_neutral_filter=True,
        ),
    },
    "notebook_early_neutral_validate": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "pre_early_neutral_regular": PRE_EARLY_NEUTRAL_REGULAR_CONFIG.to_agent_kwargs(),
        "early_neutral_bias": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
        ),
        "early_neutral_safe_only": from_base(
            PRE_EARLY_NEUTRAL_REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_bonus=4.0,
            early_neutral_safe_bonus=20.0,
            early_neutral_contested_penalty=16.0,
        ),
    },
    "early_neutral_coarse_search": {
        **HISTORICAL_BEST_VARIANTS,
        "step30": from_base(REGULAR_CONFIG, early_neutral_step_limit=30),
        "step50": from_base(REGULAR_CONFIG, early_neutral_step_limit=50),
        "step60": from_base(REGULAR_CONFIG, early_neutral_step_limit=60),
        "prod25": from_base(REGULAR_CONFIG, early_neutral_min_production=2.5),
        "prod35": from_base(REGULAR_CONFIG, early_neutral_min_production=3.5),
        "prod40": from_base(REGULAR_CONFIG, early_neutral_min_production=4.0),
        "max_ships10": from_base(REGULAR_CONFIG, early_neutral_max_ships=10),
        "max_ships20": from_base(REGULAR_CONFIG, early_neutral_max_ships=20),
        "eta25": from_base(REGULAR_CONFIG, early_neutral_max_eta=25),
        "eta45": from_base(REGULAR_CONFIG, early_neutral_max_eta=45),
        "reaction_gap3": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=3),
        "safe_bonus16": from_base(REGULAR_CONFIG, early_neutral_safe_bonus=16.0),
        "safe_bonus24": from_base(REGULAR_CONFIG, early_neutral_safe_bonus=24.0),
        "contested_penalty20": from_base(REGULAR_CONFIG, early_neutral_contested_penalty=20.0),
        "static_mult14": from_base(REGULAR_CONFIG, early_neutral_static_multiplier=1.40),
        "holdability_relief075": from_base(REGULAR_CONFIG, early_neutral_holdability_relief=0.75),
    },
    "early_neutral_promising_validate": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "pre_value_defense_regular": PRE_VALUE_DEFENSE_REGULAR_CONFIG.to_agent_kwargs(),
        "pre_early_neutral_regular": PRE_EARLY_NEUTRAL_REGULAR_CONFIG.to_agent_kwargs(),
        "step30": from_base(REGULAR_CONFIG, early_neutral_step_limit=30),
        "step50": from_base(REGULAR_CONFIG, early_neutral_step_limit=50),
        "max_ships20": from_base(REGULAR_CONFIG, early_neutral_max_ships=20),
        "reaction_gap3": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=3),
        "safe_bonus16": from_base(REGULAR_CONFIG, early_neutral_safe_bonus=16.0),
        "holdability_relief075": from_base(REGULAR_CONFIG, early_neutral_holdability_relief=0.75),
        "step30_reaction_gap3": from_base(
            REGULAR_CONFIG,
            early_neutral_step_limit=30,
            early_neutral_reaction_margin=3,
        ),
        "step30_holdability075": from_base(
            REGULAR_CONFIG,
            early_neutral_step_limit=30,
            early_neutral_holdability_relief=0.75,
        ),
        "reaction_gap3_holdability075": from_base(
            REGULAR_CONFIG,
            early_neutral_reaction_margin=3,
            early_neutral_holdability_relief=0.75,
        ),
        "max20_reaction_gap3": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=20,
            early_neutral_reaction_margin=3,
        ),
    },
    "early_neutral_reaction_refine": {
        "pre_reaction_margin_regular": PRE_REACTION_MARGIN_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "reaction_gap4": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=4),
        "reaction_gap5": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=5),
        "step30": from_base(REGULAR_CONFIG, early_neutral_step_limit=30),
        "step35": from_base(REGULAR_CONFIG, early_neutral_step_limit=35),
        "step45": from_base(REGULAR_CONFIG, early_neutral_step_limit=45),
        "step50": from_base(REGULAR_CONFIG, early_neutral_step_limit=50),
        "step30_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_step_limit=30,
            early_neutral_reaction_margin=4,
        ),
        "step35_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_step_limit=35,
            early_neutral_reaction_margin=4,
        ),
        "step50_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_step_limit=50,
            early_neutral_reaction_margin=4,
        ),
        "max20": from_base(REGULAR_CONFIG, early_neutral_max_ships=20),
        "max20_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=20,
            early_neutral_reaction_margin=4,
        ),
        "max18": from_base(REGULAR_CONFIG, early_neutral_max_ships=18),
        "safe18_contested18": from_base(
            REGULAR_CONFIG,
            early_neutral_safe_bonus=18.0,
            early_neutral_contested_penalty=18.0,
        ),
        "safe22_contested18": from_base(
            REGULAR_CONFIG,
            early_neutral_safe_bonus=22.0,
            early_neutral_contested_penalty=18.0,
        ),
    },
    "early_neutral_balanced_validate": {
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "reaction_gap4": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=4),
        "reaction_gap5": from_base(REGULAR_CONFIG, early_neutral_reaction_margin=5),
        "max16": from_base(REGULAR_CONFIG, early_neutral_max_ships=16),
        "max17": from_base(REGULAR_CONFIG, early_neutral_max_ships=17),
        "max18": from_base(REGULAR_CONFIG, early_neutral_max_ships=18),
        "max19": from_base(REGULAR_CONFIG, early_neutral_max_ships=19),
        "max20": from_base(REGULAR_CONFIG, early_neutral_max_ships=20),
        "max16_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=16,
            early_neutral_reaction_margin=4,
        ),
        "max17_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=17,
            early_neutral_reaction_margin=4,
        ),
        "max18_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=18,
            early_neutral_reaction_margin=4,
        ),
        "max19_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=19,
            early_neutral_reaction_margin=4,
        ),
        "max20_gap4": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=20,
            early_neutral_reaction_margin=4,
        ),
        "max18_gap5": from_base(
            REGULAR_CONFIG,
            early_neutral_max_ships=18,
            early_neutral_reaction_margin=5,
        ),
    },
    "early_neutral_dynamic_cap": {
        "pre_dynamic_cap_regular": PRE_DYNAMIC_CAP_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "dynamic_cap18_prod4_gap4_after10": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "dynamic_cap20_prod4_gap4_after10": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "dynamic_cap20_prod5_gap4_after10": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=5.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "dynamic_cap20_prod4_gap6_after10": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=6,
            early_neutral_dynamic_source_min_after=10,
        ),
        "dynamic_cap20_prod4_gap4_after15": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=15,
        ),
        "dynamic_cap18_prod4_gap6_after15": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=6,
            early_neutral_dynamic_source_min_after=15,
        ),
        "dynamic_cap20_gap4_reaction4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_reaction_margin=4,
        ),
    },
    "early_neutral_dynamic_cap_safety": {
        "pre_dynamic_cap_regular": PRE_DYNAMIC_CAP_REGULAR_CONFIG.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "cap18_source_safe_m4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_source_safety=True,
            early_neutral_dynamic_source_safety_margin=4,
        ),
        "cap18_source_safe_m8": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_source_safety=True,
            early_neutral_dynamic_source_safety_margin=8,
        ),
        "cap18_target_hold_m4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=4,
        ),
        "cap18_source_and_hold_m4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_source_safety=True,
            early_neutral_dynamic_source_safety_margin=4,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=4,
        ),
        "cap20_prod5_source_safe": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=5.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_source_safety=True,
            early_neutral_dynamic_source_safety_margin=4,
        ),
        "cap20_prod5_source_and_hold": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=5.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_source_safety=True,
            early_neutral_dynamic_source_safety_margin=4,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=4,
        ),
    },
    "champion_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_gap5": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=5,
            early_neutral_dynamic_source_min_after=10,
        ),
        "cap18_after12": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=12,
        ),
        "cap18_step35": from_base(REGULAR_CONFIG, early_neutral_step_limit=35),
        "cap18_step45": from_base(REGULAR_CONFIG, early_neutral_step_limit=45),
        "cap18_holdability_w025": from_base(REGULAR_CONFIG, holdability_weight=0.25),
        "cap18_holdability_w075": from_base(REGULAR_CONFIG, holdability_weight=0.75),
        "cap18_safe_bonus24": from_base(REGULAR_CONFIG, early_neutral_safe_bonus=24.0),
        "cap18_contested_penalty20": from_base(REGULAR_CONFIG, early_neutral_contested_penalty=20.0),
        "cap19_prod4_gap4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=19,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "cap20_prod4_gap4": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=20,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
        ),
        "cap18_target_hold_m0": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=0,
        ),
        "cap18_target_hold_m2": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=4,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=2,
        ),
    },
    "champion_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_gap5": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=5,
            early_neutral_dynamic_source_min_after=10,
        ),
        "cap18_safe_bonus24": from_base(REGULAR_CONFIG, early_neutral_safe_bonus=24.0),
        "cap18_gap5_safe_bonus24": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=5,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_safe_bonus=24.0,
        ),
        "cap18_gap5_after12": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=5,
            early_neutral_dynamic_source_min_after=12,
        ),
        "cap18_gap5_target_hold_m0": from_base(
            PRE_DYNAMIC_CAP_REGULAR_CONFIG,
            enable_early_neutral_dynamic_max_ships=True,
            early_neutral_dynamic_max_ships=18,
            early_neutral_dynamic_min_production=4.0,
            early_neutral_dynamic_min_enemy_gap=5,
            early_neutral_dynamic_source_min_after=10,
            early_neutral_dynamic_check_target_hold=True,
            early_neutral_dynamic_target_hold_margin=0,
        ),
    },
    "midgame_source_reserve": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mid180_tiny_highprod": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=180,
            local_reserve_min_production=3.0,
            local_reserve_enemy_distance=0.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=0,
        ),
        "mid220_tiny_highprod": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=220,
            local_reserve_min_production=3.0,
            local_reserve_enemy_distance=0.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=0,
        ),
        "mid250_tiny_highprod": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=250,
            local_reserve_min_production=3.0,
            local_reserve_enemy_distance=0.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=0,
        ),
        "mid220_front25_soft": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=220,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=25.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=2,
        ),
        "mid220_front35_soft": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=220,
            local_reserve_min_production=4.0,
            local_reserve_enemy_distance=35.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=2,
        ),
        "mid220_tiny_until320": from_base(
            REGULAR_CONFIG,
            enable_local_source_reserve=True,
            local_reserve_min_step=220,
            local_reserve_max_step=320,
            local_reserve_min_production=3.0,
            local_reserve_enemy_distance=0.0,
            local_reserve_turns=1,
            local_reserve_min_garrison=2,
            local_reserve_front_bonus=0,
        ),
    },
    "champion_launch_punish_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "launch_age8": from_base(REGULAR_CONFIG, enemy_launch_punish_max_fleet_age=8),
        "launch_age16": from_base(REGULAR_CONFIG, enemy_launch_punish_max_fleet_age=16),
        "launch_age20": from_base(REGULAR_CONFIG, enemy_launch_punish_max_fleet_age=20),
        "launch_outgoing8": from_base(REGULAR_CONFIG, enemy_launch_punish_min_outgoing=8),
        "launch_outgoing16": from_base(REGULAR_CONFIG, enemy_launch_punish_min_outgoing=16),
        "launch_outgoing20": from_base(REGULAR_CONFIG, enemy_launch_punish_min_outgoing=20),
        "launch_bonus025": from_base(REGULAR_CONFIG, enemy_launch_punish_bonus_weight=0.25),
        "launch_bonus070": from_base(REGULAR_CONFIG, enemy_launch_punish_bonus_weight=0.70),
        "launch_bonus100": from_base(REGULAR_CONFIG, enemy_launch_punish_bonus_weight=1.00),
        "launch_targets3": from_base(REGULAR_CONFIG, enemy_launch_punish_max_targets=3),
        "launch_targets4": from_base(REGULAR_CONFIG, enemy_launch_punish_max_targets=4),
        "launch_age16_targets3": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_fleet_age=16,
            enemy_launch_punish_max_targets=3,
        ),
        "launch_age16_bonus070_targets3": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_fleet_age=16,
            enemy_launch_punish_bonus_weight=0.70,
            enemy_launch_punish_max_targets=3,
        ),
        "launch_outgoing8_age16": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_min_outgoing=8,
            enemy_launch_punish_max_fleet_age=16,
        ),
        "launch_outgoing16_bonus070": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_min_outgoing=16,
            enemy_launch_punish_bonus_weight=0.70,
        ),
    },
    "champion_launch_punish_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "cap18_base_regular": CAP18_BASE_REGULAR_CONFIG.to_agent_kwargs(),
        "launch_targets4": from_base(REGULAR_CONFIG, enemy_launch_punish_max_targets=4),
        "launch_age8": from_base(REGULAR_CONFIG, enemy_launch_punish_max_fleet_age=8),
        "launch_targets4_age8": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_targets=4,
            enemy_launch_punish_max_fleet_age=8,
        ),
        "launch_targets4_outgoing16": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_targets=4,
            enemy_launch_punish_min_outgoing=16,
        ),
        "launch_targets4_bonus025": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_targets=4,
            enemy_launch_punish_bonus_weight=0.25,
        ),
        "launch_targets4_bonus070": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_max_targets=4,
            enemy_launch_punish_bonus_weight=0.70,
        ),
    },
    "champion_launch_context_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "launch_gate_step40": from_base(REGULAR_CONFIG, enemy_launch_punish_min_step=40),
        "launch_gate_step60": from_base(REGULAR_CONFIG, enemy_launch_punish_min_step=60),
        "launch_gate_step80": from_base(REGULAR_CONFIG, enemy_launch_punish_min_step=80),
        "launch_gate_step100": from_base(REGULAR_CONFIG, enemy_launch_punish_min_step=100),
        "launch_gate_prod_m4": from_base(REGULAR_CONFIG, enemy_launch_punish_min_prod_diff=-4.0),
        "launch_gate_prod0": from_base(REGULAR_CONFIG, enemy_launch_punish_min_prod_diff=0.0),
        "launch_gate_planet_m2": from_base(REGULAR_CONFIG, enemy_launch_punish_min_planet_diff=-2),
        "launch_gate_planet0": from_base(REGULAR_CONFIG, enemy_launch_punish_min_planet_diff=0),
        "launch_gate_ship080": from_base(REGULAR_CONFIG, enemy_launch_punish_min_ship_ratio=0.80),
        "launch_gate_ship090": from_base(REGULAR_CONFIG, enemy_launch_punish_min_ship_ratio=0.90),
        "launch_gate_step60_ship080": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_min_step=60,
            enemy_launch_punish_min_ship_ratio=0.80,
        ),
        "launch_gate_step60_prod0": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_min_step=60,
            enemy_launch_punish_min_prod_diff=0.0,
        ),
        "launch_gate_step80_ship090": from_base(
            REGULAR_CONFIG,
            enemy_launch_punish_min_step=80,
            enemy_launch_punish_min_ship_ratio=0.90,
        ),
    },
    "candidate_refine": {
        "public_exact": PUBLIC_EXACT.to_agent_kwargs(),
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
        "candidate1": cfg(target_candidate_limit=1),
        "candidate2": cfg(target_candidate_limit=2),
        "candidate3": cfg(target_candidate_limit=3),
        "candidate2_min8": cfg(target_candidate_limit=2, min_ships_mine_attack=8),
        "candidate2_min12": cfg(target_candidate_limit=2, min_ships_mine_attack=12),
        "candidate2_min14": cfg(target_candidate_limit=2, min_ships_mine_attack=14),
        "candidate2_buffer2": cfg(
            target_candidate_limit=2,
            enemy_owned_production_buffer_turns=2,
        ),
        "candidate2_buffer4": cfg(
            target_candidate_limit=2,
            enemy_owned_production_buffer_turns=4,
        ),
        "candidate2_coopcap6": cfg(target_candidate_limit=2, coop_planet_cap=6),
        "candidate2_coopcap10": cfg(target_candidate_limit=2, coop_planet_cap=10),
        "candidate2_min12_buffer4": cfg(
            target_candidate_limit=2,
            min_ships_mine_attack=12,
            enemy_owned_production_buffer_turns=4,
        ),
    },
    "online_replay_followup_config": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_local4": from_base(REGULAR_CONFIG, multiplayer_local_enemy_bonus=4.0),
        "mp_local3": from_base(REGULAR_CONFIG, multiplayer_local_enemy_bonus=3.0),
        "mp_local2": from_base(REGULAR_CONFIG, multiplayer_local_enemy_bonus=2.0),
        "mp_local0": from_base(REGULAR_CONFIG, multiplayer_local_enemy_bonus=0.0),
        "mp_local3_neu4": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            multiplayer_neutral_bonus=4.0,
        ),
        "mp_local3_neu5": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            multiplayer_neutral_bonus=5.0,
        ),
        "early_dyn22_after7": from_base(
            REGULAR_CONFIG,
            early_neutral_dynamic_max_ships=22,
            early_neutral_dynamic_source_min_after=7,
        ),
        "early_dyn25_after5": from_base(
            REGULAR_CONFIG,
            early_neutral_dynamic_max_ships=25,
            early_neutral_dynamic_source_min_after=5,
        ),
        "early_dyn30_after3": from_base(
            REGULAR_CONFIG,
            early_neutral_dynamic_max_ships=30,
            early_neutral_dynamic_source_min_after=3,
        ),
        "front_support_soft": from_base(
            REGULAR_CONFIG,
            enable_front_support=True,
            support_distance_factor=1.6,
            support_min_available=35,
            support_fraction=0.35,
            support_max_arrival=20,
        ),
        "front_support_medium": from_base(
            REGULAR_CONFIG,
            enable_front_support=True,
            support_distance_factor=1.45,
            support_min_available=28,
            support_fraction=0.50,
            support_max_arrival=28,
        ),
        "mp_local3_early25_after5": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            early_neutral_dynamic_max_ships=25,
            early_neutral_dynamic_source_min_after=5,
        ),
        "mp_local3_front_soft": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            enable_front_support=True,
            support_distance_factor=1.6,
            support_min_available=35,
            support_fraction=0.35,
            support_max_arrival=20,
        ),
    },
    "online_replay_followup_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "mp_local3_neu5": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            multiplayer_neutral_bonus=5.0,
        ),
        "early_dyn30_after3": from_base(
            REGULAR_CONFIG,
            early_neutral_dynamic_max_ships=30,
            early_neutral_dynamic_source_min_after=3,
        ),
        "mp_local3_neu5_early30_after3": from_base(
            REGULAR_CONFIG,
            multiplayer_local_enemy_bonus=3.0,
            multiplayer_neutral_bonus=5.0,
            early_neutral_dynamic_max_ships=30,
            early_neutral_dynamic_source_min_after=3,
        ),
        "mp_local4": from_base(REGULAR_CONFIG, multiplayer_local_enemy_bonus=4.0),
    },
    "path_first_hit_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "path_pad05_wait6": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.5,
            path_block_wait_penalty=6.0,
        ),
        "path_pad05_wait12": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.5,
            path_block_wait_penalty=12.0,
        ),
        "path_pad05_wait24": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.5,
            path_block_wait_penalty=24.0,
        ),
        "path_pad08_wait6": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=6.0,
        ),
        "path_pad08_wait12": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=12.0,
        ),
        "path_pad08_wait24": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=24.0,
        ),
        "path_pad12_wait12": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=1.2,
            path_block_wait_penalty=12.0,
        ),
    },
    "source_threat_reserve_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "threat_soft": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_radius=35.0,
            source_threat_margin=4,
            source_threat_roi_multiplier=1.50,
            source_threat_min_net_value=30.0,
        ),
        "threat_medium": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_radius=45.0,
            source_threat_margin=6,
            source_threat_roi_multiplier=1.20,
            source_threat_min_net_value=15.0,
        ),
        "threat_frontier": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_radius=35.0,
            source_threat_margin=8,
            source_threat_roi_multiplier=1.20,
            source_threat_min_net_value=20.0,
            source_threat_min_production=4.0,
        ),
        "threat_late_soft": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_min_step=60,
            source_threat_radius=35.0,
            source_threat_margin=4,
            source_threat_roi_multiplier=1.50,
            source_threat_min_net_value=30.0,
        ),
        "threat_late_soft_4p": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_min_active_players=4,
            source_threat_min_step=60,
            source_threat_radius=35.0,
            source_threat_margin=4,
            source_threat_roi_multiplier=1.50,
            source_threat_min_net_value=30.0,
        ),
        "path05w12_threat_soft": from_base(
            REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            path_first_hit_padding=0.5,
            path_block_wait_penalty=12.0,
            enable_source_threat_reserve=True,
            source_threat_radius=35.0,
            source_threat_margin=4,
            source_threat_roi_multiplier=1.50,
            source_threat_min_net_value=30.0,
        ),
    },
    "source_threat_reserve_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "threat_late_soft_4p": from_base(
            REGULAR_CONFIG,
            enable_source_threat_reserve=True,
            source_threat_min_active_players=4,
            source_threat_min_step=60,
            source_threat_radius=35.0,
            source_threat_margin=4,
            source_threat_roi_multiplier=1.50,
            source_threat_min_net_value=30.0,
        ),
    },
    "source_threat_send_filter_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "send_filter_soft": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.0,
        ),
        "send_filter_4p_soft": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.0,
        ),
        "send_filter_4p_strict_trade": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.5,
        ),
        "send_filter_4p_late90": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=90,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.0,
        ),
        "send_filter_4p_high_prod": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_min_production=4.0,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.0,
        ),
        "send_filter_2p_soft": from_base(
            REGULAR_CONFIG,
            enable_source_threat_send_filter=True,
            source_threat_send_min_active_players=2,
            source_threat_send_min_step=80,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_send_trade_ratio=1.0,
        ),
    },
    "source_threat_target_penalty_refine": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "penalty_4p_w002": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.02,
        ),
        "penalty_4p_w005": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.05,
        ),
        "penalty_4p_w010": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.10,
        ),
        "penalty_4p_late90_w005": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=90,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.05,
        ),
        "penalty_4p_relaxed_w002": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=45.0,
            source_threat_send_margin=2,
            source_threat_send_roi_multiplier=1.20,
            source_threat_send_min_net_value=15.0,
            source_threat_target_penalty_weight=0.02,
        ),
        "penalty_4p_cand3_w005": from_base(
            REGULAR_CONFIG,
            target_candidate_limit=3,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=60,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.05,
        ),
    },
    "source_threat_target_penalty_validate": {
        "regular": REGULAR_CONFIG.to_agent_kwargs(),
        "penalty_4p_late90_w005": from_base(
            REGULAR_CONFIG,
            enable_source_threat_target_penalty=True,
            source_threat_send_min_active_players=4,
            source_threat_send_min_step=90,
            source_threat_send_radius=35.0,
            source_threat_send_margin=4,
            source_threat_send_roi_multiplier=1.50,
            source_threat_send_min_net_value=30.0,
            source_threat_target_penalty_weight=0.05,
        ),
    },
    "path_comet_source_gate_refine": {
        "regular": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
        "current_regular": REGULAR_CONFIG.to_agent_kwargs(),
        "A_first_hit_skip": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=False,
            path_first_hit_padding=0.8,
        ),
        "B_first_hit_redirect": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=True,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=12.0,
        ),
        "B_first_hit_redirect_4p": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=True,
            path_first_hit_min_active_players=4,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=12.0,
        ),
        "C_comet_evac_8": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_comet_evacuation=True,
            comet_evacuation_remaining_turns=8,
        ),
        "C_comet_evac_12": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_comet_evacuation=True,
            comet_evacuation_remaining_turns=12,
        ),
        "C_comet_evac_16": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_comet_evacuation=True,
            comet_evacuation_remaining_turns=16,
        ),
        "D_source_gate_soft": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_local_source_defense_gate=True,
            local_source_defense_gate_min_production=3.0,
            local_source_defense_gate_front_distance=45.0,
            local_source_defense_gate_margin=6,
            local_source_defense_gate_use_arrival_production=True,
        ),
        "D_source_gate_4p": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_local_source_defense_gate=True,
            local_source_defense_gate_min_active_players=4,
            local_source_defense_gate_min_step=60,
            local_source_defense_gate_min_production=3.0,
            local_source_defense_gate_front_distance=45.0,
            local_source_defense_gate_margin=6,
            local_source_defense_gate_use_arrival_production=True,
        ),
        "D_source_gate_strict": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_local_source_defense_gate=True,
            local_source_defense_gate_min_active_players=4,
            local_source_defense_gate_min_step=60,
            local_source_defense_gate_min_production=4.0,
            local_source_defense_gate_front_distance=35.0,
            local_source_defense_gate_margin=10,
            local_source_defense_gate_use_arrival_production=False,
        ),
        "E_first_hit_source_gate": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=False,
            path_first_hit_padding=0.8,
            enable_local_source_defense_gate=True,
            local_source_defense_gate_min_active_players=4,
            local_source_defense_gate_min_step=60,
            local_source_defense_gate_min_production=3.0,
            local_source_defense_gate_front_distance=45.0,
            local_source_defense_gate_margin=6,
            local_source_defense_gate_use_arrival_production=True,
        ),
    },
    "path_comet_source_gate_validate": {
        "regular": MP_LOCAL3_NEU5_REGULAR_CONFIG.to_agent_kwargs(),
        "current_regular": REGULAR_CONFIG.to_agent_kwargs(),
        "B_first_hit_redirect_4p": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=True,
            path_first_hit_min_active_players=4,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=12.0,
        ),
        "C_comet_evac_12": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_comet_evacuation=True,
            comet_evacuation_remaining_turns=12,
        ),
        "B4p_plus_C12": from_base(
            MP_LOCAL3_NEU5_REGULAR_CONFIG,
            enable_path_first_hit_check=True,
            enable_path_first_hit_redirect=True,
            path_first_hit_min_active_players=4,
            path_first_hit_padding=0.8,
            path_block_wait_penalty=12.0,
            enable_comet_evacuation=True,
            comet_evacuation_remaining_turns=12,
        ),
    },
}

ABLATION_SUITES["regular_followup_validate"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "recent_loss_prod4_b40_historical": RECENT_LOSS_RECAPTURE_PROD4_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e160_w50_b40_historical": RECAPTURE_S45_E160_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
    "source_threat_frontier": from_base(
        REGULAR_CONFIG,
        enable_source_threat_reserve=True,
        source_threat_radius=35.0,
        source_threat_margin=8,
        source_threat_roi_multiplier=1.20,
        source_threat_min_net_value=20.0,
        source_threat_min_production=4.0,
    ),
    "current_recapture_w60_full": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=60,
        recent_loss_recapture_min_production=4.0,
        recent_loss_recapture_bonus=40.0,
    ),
    "current_recapture_w60_full_plus_threat": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=60,
        recent_loss_recapture_min_production=4.0,
        recent_loss_recapture_bonus=40.0,
        enable_source_threat_reserve=True,
        source_threat_radius=35.0,
        source_threat_margin=8,
        source_threat_roi_multiplier=1.20,
        source_threat_min_net_value=20.0,
        source_threat_min_production=4.0,
    ),
}

ABLATION_SUITES["regular_recapture_window_refine"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "recap_w50_s0_e500": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
    ),
    "recap_w60_s0_e500": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=60,
    ),
    "recap_w70_s0_e500": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=70,
    ),
    "recap_w60_s30_e500": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=30,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=60,
    ),
    "recap_w60_s0_e220": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=220,
        recent_loss_recapture_window=60,
    ),
    "recap_w60_s30_e220": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=30,
        recent_loss_recapture_max_step=220,
        recent_loss_recapture_window=60,
    ),
}

ABLATION_SUITES["regular_recapture_w50_source_refine"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "recap_w50_s0_e160": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=160,
        recent_loss_recapture_window=50,
    ),
    "recap_w50_s45_e500": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=45,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
    ),
    "recap_w50_s0_e500_b35": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_bonus=35.0,
    ),
    "recap_w50_s0_e500_b45": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_bonus=45.0,
    ),
    "recap_w50_s0_e500_pw4": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_prod_weight=4.0,
    ),
    "recap_w50_s0_e500_pw6": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_prod_weight=6.0,
    ),
    "recap_w50_s0_e500_prod35": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_min_production=3.5,
    ),
    "recap_w50_s0_e500_prod45": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_min_production=4.5,
    ),
}

ABLATION_SUITES["regular_recapture_b35_validate"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "recap_w50_s0_e500_b35": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_bonus=35.0,
    ),
    "recap_w50_s0_e500_prod35": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_min_production=3.5,
    ),
    "recap_w50_s0_e500_b35_prod35": from_base(
        REGULAR_CONFIG,
        recent_loss_recapture_min_step=0,
        recent_loss_recapture_max_step=500,
        recent_loss_recapture_window=50,
        recent_loss_recapture_min_production=3.5,
        recent_loss_recapture_bonus=35.0,
    ),
}

ABLATION_SUITES["third_party_tail_capture"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_capture_soft": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=2,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=60,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=18,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=18,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=30.0,
        third_party_tail_roi_multiplier=1.10,
        third_party_tail_min_net_value=8.0,
    ),
    "tail_capture_medium": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=2,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=55,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=14,
        third_party_tail_margin=2,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=16,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=40.0,
        third_party_tail_roi_multiplier=1.20,
        third_party_tail_min_net_value=12.0,
    ),
    "tail_capture_tiny": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=3,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=20,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=12,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=45.0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=6.0,
    ),
    "tail_capture_wide": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=3,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=2.5,
        third_party_tail_max_enemy_arrival=75,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=28,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=28,
        third_party_tail_source_min_after=6,
        third_party_tail_bonus=55.0,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=4.0,
    ),
    "tail_capture_precise": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=2,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=10,
        third_party_tail_margin=0,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=22,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=65.0,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=6.0,
    ),
    "tail_capture_priority": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=4,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.0,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=22,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=24,
        third_party_tail_source_min_after=6,
        third_party_tail_bonus=85.0,
        third_party_tail_prod_weight=8.0,
        third_party_tail_savings_weight=1.5,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=4.0,
    ),
    "tail_capture_noinject_tiny": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=20,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=12,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=45.0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=6.0,
    ),
    "tail_capture_noinject_precise": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=10,
        third_party_tail_margin=0,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=22,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=65.0,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=6.0,
    ),
    "tail_capture_neutral_precise": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        third_party_tail_only_neutral_targets=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=10,
        third_party_tail_margin=0,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=22,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=65.0,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=6.0,
    ),
    "tail_capture_neutral_safe": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        third_party_tail_only_neutral_targets=True,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=2,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=24,
        third_party_tail_source_min_after=8,
        third_party_tail_bonus=70.0,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=6.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=2,
    ),
}

ABLATION_SUITES["third_party_tail_capture_focused"] = {
    name: ABLATION_SUITES["third_party_tail_capture"][name]
    for name in (
        "regular",
        "tail_capture_noinject_tiny",
        "tail_capture_noinject_precise",
        "tail_capture_neutral_precise",
        "tail_capture_neutral_safe",
    )
}

ABLATION_SUITES["third_party_tail_capture_priority_gated"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_gated_keep1_saves8": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=70.0,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=16,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=8,
        third_party_tail_min_savings_ratio=0.35,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=8.0,
    ),
    "tail_gated_keep1_saves12": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=80.0,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=60,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=10,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=18,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=6.0,
        third_party_tail_savings_weight=1.4,
        third_party_tail_min_savings=12,
        third_party_tail_min_savings_ratio=0.45,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=10.0,
    ),
    "tail_gated_noinject_saves8": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=16,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=8,
        third_party_tail_min_savings_ratio=0.35,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=8.0,
    ),
    "tail_gated_neutral_safe_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=75.0,
        third_party_tail_only_neutral_targets=True,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=2,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=55.0,
        third_party_tail_prod_weight=6.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=8,
        third_party_tail_min_savings_ratio=0.35,
        third_party_tail_roi_multiplier=1.00,
        third_party_tail_min_net_value=8.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=2,
    ),
    "tail_gated_late_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=70.0,
        third_party_tail_min_active_players=3,
        third_party_tail_min_step=35,
        third_party_tail_max_step=220,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=16,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=50.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=8,
        third_party_tail_min_savings_ratio=0.35,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=8.0,
    ),
}

ABLATION_SUITES["third_party_tail_capture_replay_guard"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_guard_noinject": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=10,
        third_party_tail_min_savings_ratio=0.40,
        third_party_tail_min_post_capture_ships=4,
        third_party_tail_neutral_max_arrival=14,
        third_party_tail_neutral_min_post_capture_ships=6,
        third_party_tail_neutral_min_enemy_post_capture=2,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
    "tail_guard_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=85.0,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=10,
        third_party_tail_min_savings_ratio=0.40,
        third_party_tail_min_post_capture_ships=4,
        third_party_tail_neutral_max_arrival=14,
        third_party_tail_neutral_min_post_capture_ships=6,
        third_party_tail_neutral_min_enemy_post_capture=2,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
    "tail_guard_enemy_owned_only": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=12,
        third_party_tail_min_savings_ratio=0.45,
        third_party_tail_min_post_capture_ships=4,
        third_party_tail_neutral_max_arrival=0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
}

ABLATION_SUITES["third_party_tail_capture_effective_shape"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_enemy_owned_close_saver_noinject": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=False,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=8,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=3,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=18,
        third_party_tail_source_min_after=16,
        third_party_tail_bonus=40.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.0,
        third_party_tail_min_savings=18,
        third_party_tail_min_savings_ratio=0.45,
        third_party_tail_min_post_capture_ships=0,
        third_party_tail_neutral_max_arrival=0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=12.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
    "tail_enemy_owned_close_saver_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=80.0,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=8,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=3,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=18,
        third_party_tail_source_min_after=16,
        third_party_tail_bonus=40.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.0,
        third_party_tail_min_savings=18,
        third_party_tail_min_savings_ratio=0.45,
        third_party_tail_min_post_capture_ships=0,
        third_party_tail_neutral_max_arrival=0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=12.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
}

ABLATION_SUITES["third_party_tail_capture_watchlist"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_watch_tiny_noinject": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
    ),
    "tail_watch_tiny_regular_candidate": TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "tail_watch_guard_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_watchlist=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=88.0,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=10,
        third_party_tail_min_savings_ratio=0.40,
        third_party_tail_min_post_capture_ships=4,
        third_party_tail_neutral_max_arrival=14,
        third_party_tail_neutral_min_post_capture_ships=6,
        third_party_tail_neutral_min_enemy_post_capture=2,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
        third_party_tail_watchlist_horizon=90,
        third_party_tail_watchlist_post_window=18,
        third_party_tail_watchlist_score_bonus=8.0,
    ),
    "tail_watch_noinject_guard": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_watchlist=True,
        enable_third_party_tail_candidate_injection=False,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=10,
        third_party_tail_min_savings_ratio=0.40,
        third_party_tail_min_post_capture_ships=4,
        third_party_tail_neutral_max_arrival=14,
        third_party_tail_neutral_min_post_capture_ships=6,
        third_party_tail_neutral_min_enemy_post_capture=2,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
        third_party_tail_watchlist_horizon=90,
        third_party_tail_watchlist_post_window=18,
        third_party_tail_watchlist_score_bonus=8.0,
    ),
    "tail_watch_loose_keep1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_watchlist=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=76.0,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=3.5,
        third_party_tail_max_enemy_arrival=70,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=16,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=24,
        third_party_tail_source_min_after=10,
        third_party_tail_bonus=48.0,
        third_party_tail_prod_weight=5.5,
        third_party_tail_savings_weight=1.1,
        third_party_tail_min_savings=7,
        third_party_tail_min_savings_ratio=0.30,
        third_party_tail_min_post_capture_ships=3,
        third_party_tail_neutral_max_arrival=18,
        third_party_tail_neutral_min_post_capture_ships=4,
        third_party_tail_neutral_min_enemy_post_capture=1,
        third_party_tail_roi_multiplier=1.03,
        third_party_tail_min_net_value=8.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
        third_party_tail_watchlist_horizon=100,
        third_party_tail_watchlist_post_window=22,
        third_party_tail_watchlist_score_bonus=10.0,
    ),
    "tail_watch_enemy_owned": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_capture=True,
        enable_third_party_tail_watchlist=True,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=1,
        third_party_tail_candidate_min_score=80.0,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_min_active_players=3,
        third_party_tail_min_production=4.0,
        third_party_tail_max_enemy_arrival=65,
        third_party_tail_min_delay=1,
        third_party_tail_max_delay=12,
        third_party_tail_margin=1,
        third_party_tail_min_send=1,
        third_party_tail_max_ships=20,
        third_party_tail_source_min_after=12,
        third_party_tail_bonus=45.0,
        third_party_tail_prod_weight=5.0,
        third_party_tail_savings_weight=1.2,
        third_party_tail_min_savings=10,
        third_party_tail_min_savings_ratio=0.35,
        third_party_tail_min_post_capture_ships=3,
        third_party_tail_neutral_max_arrival=0,
        third_party_tail_roi_multiplier=1.05,
        third_party_tail_min_net_value=10.0,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
        third_party_tail_watchlist_horizon=100,
        third_party_tail_watchlist_post_window=24,
        third_party_tail_watchlist_score_bonus=12.0,
    ),
}

ABLATION_SUITES["third_party_tail_tiny_param_search"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_tiny_base_watch": TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "tail_tiny_nowatch": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        enable_third_party_tail_watchlist=False,
    ),
    "tail_tiny_prod35": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_min_production=3.5,
    ),
    "tail_tiny_prod25": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_min_production=2.5,
    ),
    "tail_tiny_max10_net6": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_max_ships=10,
    ),
    "tail_tiny_max14_net7": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_max_ships=14,
        third_party_tail_min_net_value=7.0,
    ),
    "tail_tiny_margin0": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=0,
    ),
    "tail_tiny_margin2": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
    ),
    "tail_tiny_delay14": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_max_delay=14,
    ),
    "tail_tiny_delay26": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_max_delay=26,
    ),
    "tail_tiny_holdfilter": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_hold_enemy_radius=42.0,
        third_party_tail_hold_enemy_max_arrival=35,
        third_party_tail_hold_margin=3,
    ),
    "tail_tiny_keep1_soft": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=0,
        third_party_tail_candidate_min_score=78.0,
    ),
}

ABLATION_SUITES["third_party_tail_tiny_top_confirm"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "tail_m2_base_watch": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
    ),
    "tail_m2_nowatch": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        enable_third_party_tail_watchlist=False,
    ),
    "tail_m2_prod25": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        third_party_tail_min_production=2.5,
    ),
    "tail_m2_prod25_nowatch": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        third_party_tail_min_production=2.5,
        enable_third_party_tail_watchlist=False,
    ),
    "tail_m2_delay26": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        third_party_tail_max_delay=26,
    ),
    "tail_m2_max14_net7": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        third_party_tail_max_ships=14,
        third_party_tail_min_net_value=7.0,
    ),
    "tail_m2_soft_keep1": from_base(
        TAIL_WATCH_TINY_REGULAR_CANDIDATE_CONFIG,
        third_party_tail_margin=2,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_keep_front=0,
        third_party_tail_candidate_min_score=78.0,
    ),
}

ABLATION_SUITES["myreplay_76762389_source_guard"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "local_reserve_s45_160_p4": from_base(
        REGULAR_CONFIG,
        enable_local_source_reserve=True,
        local_reserve_min_step=45,
        local_reserve_max_step=160,
        local_reserve_min_production=4.0,
        local_reserve_enemy_distance=50.0,
        local_reserve_turns=2,
        local_reserve_min_garrison=8,
        local_reserve_front_bonus=8,
    ),
    "mid_border_s45_170_p4": from_base(
        REGULAR_CONFIG,
        enable_midgame_border_source_reserve=True,
        midgame_border_min_active_players=4,
        midgame_border_step_min=45,
        midgame_border_step_max=170,
        midgame_border_min_production=4.0,
        midgame_border_enemy_radius=55.0,
        midgame_border_min_after=8,
        midgame_border_prod_turns_after=2,
        midgame_border_threat_margin=6,
    ),
    "send_filter_s45_p4": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=45,
        source_threat_send_max_step=170,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.0,
    ),
    "local_reserve_plus_send_filter": from_base(
        REGULAR_CONFIG,
        enable_local_source_reserve=True,
        local_reserve_min_step=45,
        local_reserve_max_step=160,
        local_reserve_min_production=4.0,
        local_reserve_enemy_distance=50.0,
        local_reserve_turns=2,
        local_reserve_min_garrison=8,
        local_reserve_front_bonus=8,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=45,
        source_threat_send_max_step=170,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.0,
    ),
}

ABLATION_SUITES["myreplay_source_stability_plan001"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "early_4p_local_reserve_p4_soft": from_base(
        REGULAR_CONFIG,
        enable_local_source_reserve=True,
        local_reserve_min_step=25,
        local_reserve_max_step=130,
        local_reserve_min_production=4.0,
        local_reserve_enemy_distance=50.0,
        local_reserve_turns=1,
        local_reserve_min_garrison=6,
        local_reserve_front_bonus=4,
    ),
    "mid_all_local_reserve_p4_medium": from_base(
        REGULAR_CONFIG,
        enable_local_source_reserve=True,
        local_reserve_min_step=50,
        local_reserve_max_step=190,
        local_reserve_min_production=4.0,
        local_reserve_enemy_distance=50.0,
        local_reserve_turns=2,
        local_reserve_min_garrison=8,
        local_reserve_front_bonus=6,
    ),
    "early_4p_send_filter_p4": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=25,
        source_threat_send_max_step=140,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.0,
    ),
    "recent_loss_hold_p4_w70_m8": from_base(
        REGULAR_CONFIG,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=0,
        recent_loss_recapture_hold_min_step=35,
        recent_loss_recapture_hold_max_step=190,
        recent_loss_recapture_hold_window=70,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
    ),
    "source_guard_plus_recent_loss_hold": from_base(
        REGULAR_CONFIG,
        enable_local_source_reserve=True,
        local_reserve_min_step=50,
        local_reserve_max_step=190,
        local_reserve_min_production=4.0,
        local_reserve_enemy_distance=50.0,
        local_reserve_turns=2,
        local_reserve_min_garrison=8,
        local_reserve_front_bonus=6,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=0,
        recent_loss_recapture_hold_min_step=35,
        recent_loss_recapture_hold_max_step=190,
        recent_loss_recapture_hold_window=70,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
    ),
}

ABLATION_SUITES["myreplay_recent_capture_source_plan002"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "recent_cap_src_4p_w25_soft": from_base(
        REGULAR_CONFIG,
        enable_recent_capture_source_reserve=True,
        recent_capture_source_reserve_min_active_players=4,
        recent_capture_source_reserve_min_step=25,
        recent_capture_source_reserve_max_step=140,
        recent_capture_source_reserve_window=25,
        recent_capture_source_reserve_min_production=4.0,
        recent_capture_source_reserve_min_after=6,
        recent_capture_source_reserve_prod_turns_after=1,
        recent_capture_source_reserve_enemy_radius=45.0,
        recent_capture_source_reserve_front_bonus=2,
    ),
    "recent_cap_src_4p_w35_medium": from_base(
        REGULAR_CONFIG,
        enable_recent_capture_source_reserve=True,
        recent_capture_source_reserve_min_active_players=4,
        recent_capture_source_reserve_min_step=25,
        recent_capture_source_reserve_max_step=160,
        recent_capture_source_reserve_window=35,
        recent_capture_source_reserve_min_production=4.0,
        recent_capture_source_reserve_min_after=8,
        recent_capture_source_reserve_prod_turns_after=1,
        recent_capture_source_reserve_enemy_radius=45.0,
        recent_capture_source_reserve_front_bonus=4,
    ),
    "recent_cap_src_all_w35_soft": from_base(
        REGULAR_CONFIG,
        enable_recent_capture_source_reserve=True,
        recent_capture_source_reserve_min_active_players=0,
        recent_capture_source_reserve_min_step=35,
        recent_capture_source_reserve_max_step=190,
        recent_capture_source_reserve_window=35,
        recent_capture_source_reserve_min_production=4.0,
        recent_capture_source_reserve_min_after=6,
        recent_capture_source_reserve_prod_turns_after=1,
        recent_capture_source_reserve_enemy_radius=45.0,
        recent_capture_source_reserve_front_bonus=2,
    ),
    "recent_cap_src_4p_w35_plus_loss_hold": from_base(
        REGULAR_CONFIG,
        enable_recent_capture_source_reserve=True,
        recent_capture_source_reserve_min_active_players=4,
        recent_capture_source_reserve_min_step=25,
        recent_capture_source_reserve_max_step=160,
        recent_capture_source_reserve_window=35,
        recent_capture_source_reserve_min_production=4.0,
        recent_capture_source_reserve_min_after=8,
        recent_capture_source_reserve_prod_turns_after=1,
        recent_capture_source_reserve_enemy_radius=45.0,
        recent_capture_source_reserve_front_bonus=4,
        enable_recent_loss_recapture_hold_gate=True,
        recent_loss_recapture_hold_min_active_players=0,
        recent_loss_recapture_hold_min_step=35,
        recent_loss_recapture_hold_max_step=190,
        recent_loss_recapture_hold_window=70,
        recent_loss_recapture_hold_min_production=4.0,
        recent_loss_recapture_hold_enemy_radius=45.0,
        recent_loss_recapture_hold_margin=8,
    ),
    "target_penalty_s35_p4_soft": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=160,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.02,
    ),
}

ABLATION_SUITES["myreplay_target_penalty_plan002_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "target_penalty_s35_p4_soft": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=160,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.02,
    ),
}

ABLATION_SUITES["myreplay_plan003_4p_soft_source"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_prod3_w001_s45_180": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=45,
        source_threat_send_max_step=180,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.01,
    ),
    "p4_prod3_w0015_s45_180": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=45,
        source_threat_send_max_step=180,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.015,
    ),
    "p4_prod4_w0015_s35_180": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=180,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.015,
    ),
}

ABLATION_SUITES["myreplay_plan003_2p_core_source"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2_prod4_w0015_s25_110": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=110,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.015,
    ),
    "p2_prod4_w002_s25_110": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=110,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.02,
    ),
    "p2_prod4_filter_soft_s25_110": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=110,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_send_trade_ratio=1.0,
    ),
}

ABLATION_SUITES["myreplay_plan003_2p_core_source_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2_prod4_w002_s25_110": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=110,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.02,
    ),
}

ABLATION_SUITES["myreplay_plan004_p2src_w002_only"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2src_w002_only": MP_LOCAL3_NEU5_COMET12_PATH4P_HOLD4_TERR30_S35_HOME6_POSTHOLD_RECAP_B35_P2SRC_W002_REGULAR_CONFIG.to_agent_kwargs(),
}

ABLATION_SUITES["myreplay_plan005_p2src_refine"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2src_w001_s25_110": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=110,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.01,
    ),
    "p2src_w0015_s25_90": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=90,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.015,
    ),
    "p2src_w0015_s35_100": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=35,
        source_threat_send_max_step=100,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.25,
        source_threat_send_min_net_value=20.0,
        source_threat_target_penalty_weight=0.015,
    ),
    "p2src_w002_s25_80_net30": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=80,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.02,
    ),
}

ABLATION_SUITES["myreplay_plan005_p2src_net30_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2src_w002_s25_80_net30": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=25,
        source_threat_send_max_step=80,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.02,
    ),
}

ABLATION_SUITES["myreplay_plan006_lowprod_opening"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "open_hold2p_b2_t1_s45": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=2,
        opening_hold_max_active_players=2,
        opening_hold_step_limit=45,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=2,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=4,
    ),
    "open_hold2p_b4_t1_s45": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=2,
        opening_hold_max_active_players=2,
        opening_hold_step_limit=45,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=4,
    ),
    "open_hold2p_b4_t2_s55": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=2,
        opening_hold_max_active_players=2,
        opening_hold_step_limit=55,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=2,
        opening_hold_contested_extra=6,
    ),
    "early_dyn22_after7": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=22,
        early_neutral_dynamic_source_min_after=7,
    ),
    "early_dyn25_after5": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=25,
        early_neutral_dynamic_source_min_after=5,
    ),
    "dyn22_hold2p_b2_t1": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=22,
        early_neutral_dynamic_source_min_after=7,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=2,
        opening_hold_max_active_players=2,
        opening_hold_step_limit=45,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=2,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=4,
    ),
    "dyn25_hold2p_b4_t1": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=25,
        early_neutral_dynamic_source_min_after=5,
        enable_opening_neutral_hold_margin=True,
        opening_hold_min_active_players=2,
        opening_hold_max_active_players=2,
        opening_hold_step_limit=45,
        opening_hold_min_production=4.0,
        opening_hold_base_margin=4,
        opening_hold_prod_turns=1,
        opening_hold_contested_extra=4,
    ),
}

ABLATION_SUITES["myreplay_plan007_lowprod_dyn_refine"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "early_dyn19_after9": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=19,
        early_neutral_dynamic_source_min_after=9,
    ),
    "early_dyn19_after8": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=19,
        early_neutral_dynamic_source_min_after=8,
    ),
    "early_dyn20_after9": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=20,
        early_neutral_dynamic_source_min_after=9,
    ),
    "early_dyn20_after8": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=20,
        early_neutral_dynamic_source_min_after=8,
    ),
    "early_dyn21_after9": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=21,
        early_neutral_dynamic_source_min_after=9,
    ),
    "early_dyn21_after8": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=21,
        early_neutral_dynamic_source_min_after=8,
    ),
    "early_dyn20_after8_targethold0": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=20,
        early_neutral_dynamic_source_min_after=8,
        early_neutral_dynamic_check_target_hold=True,
        early_neutral_dynamic_target_hold_margin=0,
    ),
    "early_dyn20_after8_targethold2": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=20,
        early_neutral_dynamic_source_min_after=8,
        early_neutral_dynamic_check_target_hold=True,
        early_neutral_dynamic_target_hold_margin=2,
    ),
}

ABLATION_SUITES["myreplay_plan008_historical_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "launch_targets4_age8_regular": LAUNCH_TARGETS4_AGE8_REGULAR_CONFIG.to_agent_kwargs(),
    "recapture_s45_e180_w50_b40_regular": RECAPTURE_S45_E180_W50_B40_REGULAR_CONFIG.to_agent_kwargs(),
}

ABLATION_SUITES["myreplay_plan009_4p_soft_source_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_prod4_w0015_s35_180": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=180,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=40.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.015,
    ),
}

ABLATION_SUITES["myreplay_plan011_leader_bonus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "leader_bonus_05": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=0.5,
    ),
    "leader_bonus_10": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.0,
    ),
    "leader_bonus_15": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.5,
    ),
    "leader_bonus_20": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=2.0,
    ),
}

ABLATION_SUITES["myreplay_plan011_leader_bonus_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "leader_bonus_10": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.0,
    ),
    "leader_bonus_15": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.5,
    ),
}

ABLATION_SUITES["myreplay_plan012_p2_prod3_source"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2src3_w005_s55_100_net25": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=55,
        source_threat_send_max_step=100,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=42.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.005,
    ),
    "p2src3_w010_s55_100_net25": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=55,
        source_threat_send_max_step=100,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=42.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_target_penalty_weight=0.010,
    ),
    "p2src3_w010_s45_110_net30": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=45,
        source_threat_send_max_step=110,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=4,
        source_threat_send_roi_multiplier=1.40,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.010,
    ),
    "p2src3_w015_s55_95_net30": from_base(
        REGULAR_CONFIG,
        enable_source_threat_target_penalty=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=55,
        source_threat_send_max_step=95,
        source_threat_send_min_production=3.0,
        source_threat_send_radius=42.0,
        source_threat_send_margin=5,
        source_threat_send_roi_multiplier=1.45,
        source_threat_send_min_net_value=30.0,
        source_threat_target_penalty_weight=0.015,
    ),
}

ABLATION_SUITES["myreplay_plan013_lead_stabilize"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "lead_stab_recent35_m32_s90_210": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=3,
        proactive_defense_max_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=35,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=210,
        proactive_defense_min_production=4.0,
        proactive_defense_min_prod_diff=0.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.80,
        proactive_defense_max_send=32,
        proactive_defense_base_margin=4,
        proactive_defense_prod_turns=2,
        proactive_defense_enemy_launch_window=8,
        proactive_defense_enemy_reserve_turns=1,
        proactive_defense_enemy_send_fraction=0.90,
        proactive_defense_threat_slack=4,
        proactive_defense_max_arrival=50,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=25.0,
    ),
    "lead_stab_recent50_m32_s90_220": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=3,
        proactive_defense_max_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_recent_capture=True,
        proactive_defense_recent_capture_window=50,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=220,
        proactive_defense_min_production=4.0,
        proactive_defense_min_prod_diff=0.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.80,
        proactive_defense_max_send=32,
        proactive_defense_base_margin=4,
        proactive_defense_prod_turns=2,
        proactive_defense_enemy_launch_window=8,
        proactive_defense_enemy_reserve_turns=1,
        proactive_defense_enemy_send_fraction=0.90,
        proactive_defense_threat_slack=4,
        proactive_defense_max_arrival=50,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=25.0,
    ),
    "lead_stab_high_value_m36_s90_210": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=3,
        proactive_defense_max_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=90,
        proactive_defense_max_step=210,
        proactive_defense_min_production=5.0,
        proactive_defense_min_prod_diff=0.0,
        proactive_defense_min_planet_diff=0,
        proactive_defense_min_ship_ratio=0.75,
        proactive_defense_max_send=36,
        proactive_defense_base_margin=5,
        proactive_defense_prod_turns=2,
        proactive_defense_enemy_launch_window=10,
        proactive_defense_enemy_reserve_turns=1,
        proactive_defense_enemy_send_fraction=0.90,
        proactive_defense_threat_slack=6,
        proactive_defense_max_arrival=55,
        proactive_defense_source_min_after=10,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=30.0,
    ),
    "lead_stab_no_recent_m28_s100_190": from_base(
        REGULAR_CONFIG,
        enable_proactive_value_defense=True,
        proactive_defense_min_active_players=3,
        proactive_defense_max_active_players=4,
        proactive_defense_after_attacks=True,
        proactive_defense_require_turn_attack=True,
        proactive_defense_require_enemy_positive_roi=True,
        proactive_defense_min_step=100,
        proactive_defense_max_step=190,
        proactive_defense_min_production=4.0,
        proactive_defense_min_prod_diff=2.0,
        proactive_defense_min_planet_diff=1,
        proactive_defense_min_ship_ratio=0.90,
        proactive_defense_max_send=28,
        proactive_defense_base_margin=3,
        proactive_defense_prod_turns=2,
        proactive_defense_enemy_launch_window=8,
        proactive_defense_enemy_reserve_turns=1,
        proactive_defense_enemy_send_fraction=0.90,
        proactive_defense_threat_slack=4,
        proactive_defense_max_arrival=45,
        proactive_defense_source_min_after=12,
        proactive_defense_source_prod_turns_after=1,
        proactive_defense_source_front_bonus=4,
        proactive_defense_enemy_min_net_value=35.0,
    ),
}

ABLATION_SUITES["myreplay_plan014_early_4p_source_guard"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_early_send_filter_s35_90_m6": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=4,
        source_threat_send_max_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=90,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.0,
    ),
    "p4_early_send_filter_s35_80_trade120": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=4,
        source_threat_send_max_active_players=4,
        source_threat_send_min_step=35,
        source_threat_send_max_step=80,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=42.0,
        source_threat_send_margin=5,
        source_threat_send_roi_multiplier=1.40,
        source_threat_send_min_net_value=30.0,
        source_threat_send_trade_ratio=1.20,
    ),
    "p4_early_local_gate_s35_90_m6": from_base(
        REGULAR_CONFIG,
        enable_local_source_defense_gate=True,
        local_source_defense_gate_min_active_players=4,
        local_source_defense_gate_min_step=35,
        local_source_defense_gate_max_step=90,
        local_source_defense_gate_min_production=4.0,
        local_source_defense_gate_front_distance=45.0,
        local_source_defense_gate_margin=6,
        local_source_defense_gate_enemy_fraction=0.85,
        local_source_defense_gate_enemy_launch_window=8,
        local_source_defense_gate_enemy_reserve_turns=2,
        local_source_defense_gate_max_arrival=45,
        local_source_defense_gate_use_arrival_production=True,
    ),
    "p4_early_source_reserve_s35_90_m8": from_base(
        REGULAR_CONFIG,
        enable_source_threat_reserve=True,
        source_threat_min_active_players=4,
        source_threat_min_step=35,
        source_threat_max_step=90,
        source_threat_min_production=4.0,
        source_threat_radius=45.0,
        source_threat_enemy_send_fraction=0.85,
        source_threat_enemy_launch_window=8,
        source_threat_enemy_reserve_turns=2,
        source_threat_max_arrival=45,
        source_threat_margin=8,
        source_threat_roi_multiplier=1.35,
        source_threat_min_net_value=25.0,
    ),
}

ABLATION_SUITES["myreplay_plan014_early_4p_source_guard_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_early_send_filter_s35_80_trade120": ABLATION_SUITES["myreplay_plan014_early_4p_source_guard"][
        "p4_early_send_filter_s35_80_trade120"
    ],
}

ABLATION_SUITES["myreplay_plan015_p2_highprod_source_guard"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2_highprod_send_filter_s45_115_m6": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=45,
        source_threat_send_max_step=115,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=6,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.15,
    ),
    "p2_highprod_send_filter_s45_160_m8": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=True,
        source_threat_send_min_active_players=2,
        source_threat_send_max_active_players=2,
        source_threat_send_min_step=45,
        source_threat_send_max_step=160,
        source_threat_send_min_production=4.0,
        source_threat_send_radius=45.0,
        source_threat_send_margin=8,
        source_threat_send_roi_multiplier=1.35,
        source_threat_send_min_net_value=25.0,
        source_threat_send_trade_ratio=1.20,
    ),
    "p2_highprod_source_reserve_s45_140_m8": from_base(
        REGULAR_CONFIG,
        enable_source_threat_reserve=True,
        source_threat_min_active_players=2,
        source_threat_min_step=45,
        source_threat_max_step=140,
        source_threat_min_production=4.0,
        source_threat_radius=45.0,
        source_threat_enemy_send_fraction=0.85,
        source_threat_enemy_launch_window=8,
        source_threat_enemy_reserve_turns=2,
        source_threat_max_arrival=45,
        source_threat_margin=8,
        source_threat_roi_multiplier=1.35,
        source_threat_min_net_value=25.0,
    ),
    "p2_highprod_source_reserve_s80_220_m10": from_base(
        REGULAR_CONFIG,
        enable_source_threat_reserve=True,
        source_threat_min_active_players=2,
        source_threat_min_step=80,
        source_threat_max_step=220,
        source_threat_min_production=5.0,
        source_threat_radius=45.0,
        source_threat_enemy_send_fraction=0.85,
        source_threat_enemy_launch_window=8,
        source_threat_enemy_reserve_turns=2,
        source_threat_max_arrival=45,
        source_threat_margin=10,
        source_threat_roi_multiplier=1.35,
        source_threat_min_net_value=25.0,
    ),
    "p2_highprod_recent_reserve_s45_150_m6": from_base(
        REGULAR_CONFIG,
        enable_recent_capture_source_reserve=True,
        recent_capture_source_reserve_min_active_players=2,
        recent_capture_source_reserve_min_step=45,
        recent_capture_source_reserve_max_step=150,
        recent_capture_source_reserve_window=35,
        recent_capture_source_reserve_min_production=4.0,
        recent_capture_source_reserve_min_after=8,
        recent_capture_source_reserve_prod_turns_after=1,
        recent_capture_source_reserve_enemy_radius=45.0,
        recent_capture_source_reserve_front_bonus=6,
    ),
}

ABLATION_SUITES["myreplay_plan015_p2_highprod_source_guard_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2_highprod_send_filter_s45_115_m6": ABLATION_SUITES["myreplay_plan015_p2_highprod_source_guard"][
        "p2_highprod_send_filter_s45_115_m6"
    ],
    "p2_highprod_source_reserve_s45_140_m8": ABLATION_SUITES["myreplay_plan015_p2_highprod_source_guard"][
        "p2_highprod_source_reserve_s45_140_m8"
    ],
    "p2_highprod_recent_reserve_s45_150_m6": ABLATION_SUITES["myreplay_plan015_p2_highprod_source_guard"][
        "p2_highprod_recent_reserve_s45_150_m6"
    ],
}

ABLATION_SUITES["myreplay_plan018_p2_low_home_expansion_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p2_low_home_trickle_s30_src1_tgt4_m5": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_min_active_players=2,
        opening_trickle_max_active_players=2,
        opening_trickle_step_limit=30,
        opening_trickle_source_min_production=1.0,
        opening_trickle_target_min_production=4.0,
        opening_trickle_max_target_ships=12,
        opening_trickle_min_ships=5,
    ),
    "p2_low_home_trickle_s40_src1_tgt3_m5": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_min_active_players=2,
        opening_trickle_max_active_players=2,
        opening_trickle_step_limit=40,
        opening_trickle_source_min_production=1.0,
        opening_trickle_target_min_production=3.0,
        opening_trickle_max_target_ships=14,
        opening_trickle_min_ships=5,
    ),
    "p2_low_home_dyn22_after6": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=22,
        early_neutral_dynamic_source_min_after=6,
        early_neutral_dynamic_min_enemy_gap=2,
    ),
    "p2_low_home_dyn24_after5_prod3": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=24,
        early_neutral_dynamic_source_min_after=5,
        early_neutral_dynamic_min_production=3.0,
        early_neutral_dynamic_min_enemy_gap=2,
    ),
    "p2_low_home_bonus_prod2_s55": from_base(
        REGULAR_CONFIG,
        early_neutral_step_limit=55,
        early_neutral_min_production=2.0,
        early_neutral_bonus=6.0,
        early_neutral_safe_bonus=24.0,
        early_neutral_contested_penalty=12.0,
    ),
}

ABLATION_SUITES["myreplay_plan019_p4_low_home_opening_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_low_home_dyn20_after8_prod3_gap2": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=20,
        early_neutral_dynamic_source_min_after=8,
        early_neutral_dynamic_min_production=3.0,
        early_neutral_dynamic_min_enemy_gap=2,
    ),
    "p4_low_home_dyn22_after6_prod3_gap2": from_base(
        REGULAR_CONFIG,
        early_neutral_dynamic_max_ships=22,
        early_neutral_dynamic_source_min_after=6,
        early_neutral_dynamic_min_production=3.0,
        early_neutral_dynamic_min_enemy_gap=2,
    ),
    "p4_low_home_bonus_prod2_s50": from_base(
        REGULAR_CONFIG,
        early_neutral_step_limit=50,
        early_neutral_min_production=2.0,
        early_neutral_bonus=6.0,
        early_neutral_safe_bonus=24.0,
        early_neutral_contested_penalty=12.0,
    ),
    "p4_low_home_bonus_prod2_s50_active4": from_base(
        REGULAR_CONFIG,
        enable_early_neutral_multiplayer_override=True,
        early_neutral_multiplayer_min_active_players=4,
        early_neutral_multiplayer_max_active_players=4,
        early_neutral_multiplayer_step_limit=50,
        early_neutral_multiplayer_min_production=2.0,
        early_neutral_multiplayer_bonus=6.0,
        early_neutral_multiplayer_safe_bonus=24.0,
        early_neutral_multiplayer_contested_penalty=12.0,
    ),
    "p4_low_home_trickle_s35_src1_tgt3_m5": from_base(
        REGULAR_CONFIG,
        enable_opening_high_prod_trickle=True,
        opening_trickle_min_active_players=4,
        opening_trickle_max_active_players=4,
        opening_trickle_step_limit=35,
        opening_trickle_source_min_production=1.0,
        opening_trickle_target_min_production=3.0,
        opening_trickle_max_target_ships=14,
        opening_trickle_min_ships=5,
    ),
    "p4_low_home_soft_territory": from_base(
        REGULAR_CONFIG,
        opening_territory_penalty=12.0,
        opening_territory_enemy_closer_margin=12.0,
        opening_territory_allow_if_safe_gap=6,
    ),
}

ABLATION_SUITES["myreplay_plan019_p4_low_home_opening_confirm"] = {
    "previous_tail_m2_max14_net7_overpay4_regular": TAIL_M2_MAX14_NET7_OVERPAY4_REGULAR_CANDIDATE_CONFIG.to_agent_kwargs(),
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "p4_low_home_bonus_prod2_s50_active4": TAIL_M2_MAX14_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_REGULAR_CONFIG.to_agent_kwargs(),
}

ABLATION_SUITES["myreplay_plan019_regular_counter_probe"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "counter_no_opening_territory_penalty": from_base(
        REGULAR_CONFIG,
        enable_opening_neutral_territory_score=False,
    ),
    "counter_soft_opening_territory_penalty": from_base(
        REGULAR_CONFIG,
        opening_territory_penalty=10.0,
        opening_territory_enemy_closer_margin=12.0,
        opening_territory_allow_if_safe_gap=6,
    ),
    "counter_no_early_source_filter": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=False,
    ),
    "counter_tail_inject_limit1": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_candidate_injection=True,
        third_party_tail_candidate_limit=1,
        third_party_tail_candidate_min_score=52.0,
        third_party_tail_candidate_keep_front=1,
    ),
    "counter_tail_hold_filter": from_base(
        REGULAR_CONFIG,
        enable_third_party_tail_hold_filter=True,
        third_party_tail_hold_margin=3,
        third_party_tail_hold_enemy_max_arrival=40,
    ),
    "counter_leader_pressure_15": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.5,
    ),
    "counter_local_pressure_5_leader15": from_base(
        REGULAR_CONFIG,
        multiplayer_local_enemy_bonus=5.0,
        multiplayer_leader_prod_bonus=1.5,
    ),
}

ABLATION_SUITES["myreplay_plan019_regular_counter_focus"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "counter_no_early_source_filter": ABLATION_SUITES["myreplay_plan019_regular_counter_probe"][
        "counter_no_early_source_filter"
    ],
    "counter_leader_pressure_15": ABLATION_SUITES["myreplay_plan019_regular_counter_probe"][
        "counter_leader_pressure_15"
    ],
    "counter_local_pressure_5_leader15": ABLATION_SUITES["myreplay_plan019_regular_counter_probe"][
        "counter_local_pressure_5_leader15"
    ],
}

ABLATION_SUITES["myreplay_plan020_regular_counter_refine"] = {
    "regular": REGULAR_CONFIG.to_agent_kwargs(),
    "leader_pressure_125": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.25,
    ),
    "leader_pressure_150": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.50,
    ),
    "leader_pressure_175": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=1.75,
    ),
    "leader_pressure_200": from_base(
        REGULAR_CONFIG,
        multiplayer_leader_prod_bonus=2.00,
    ),
    "no_early_source_filter": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=False,
    ),
    "no_source_filter_leader150": from_base(
        REGULAR_CONFIG,
        enable_source_threat_send_filter=False,
        multiplayer_leader_prod_bonus=1.50,
    ),
    "relaxed_source_filter_s35_65": from_base(
        REGULAR_CONFIG,
        source_threat_send_max_step=65,
    ),
    "relaxed_source_filter_trade140": from_base(
        REGULAR_CONFIG,
        source_threat_send_trade_ratio=1.40,
    ),
}
