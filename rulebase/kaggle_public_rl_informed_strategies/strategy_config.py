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
    enable_contested_target_adjustment: bool = False
    contested_arrival_margin: int = 3
    contested_enemy_weight: float = 1.0
    contested_friendly_credit: float = 0.75
    contested_skip_friendly_covered: bool = True
    enable_dynamic_posture: bool = False
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
    enable_proactive_value_defense: bool = False
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
    local_reserve_min_production: float = 3.0
    local_reserve_enemy_distance: float = 35.0
    local_reserve_turns: int = 2
    local_reserve_min_garrison: int = 6
    local_reserve_front_bonus: int = 8
    enable_holdability_target_score: bool = False
    holdability_radius: float = 35.0
    holdability_weight: float = 0.8
    holdability_enemy_prod_weight: float = 4.0
    holdability_enemy_ship_weight: float = 0.08
    holdability_own_prod_weight: float = 2.0
    holdability_own_ship_weight: float = 0.04
    enable_early_neutral_bias: bool = False
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
    enable_opening_rotating_neutral_filter: bool = False
    opening_rotating_step_limit: int = 80
    opening_rotating_max_eta: int = 13
    opening_rotating_low_production: float = 2.0
    opening_rotating_penalty: float = 60.0
    enable_enemy_launch_punish: bool = False
    enemy_launch_punish_max_fleet_age: int = 12
    enemy_launch_punish_min_outgoing: int = 12
    enemy_launch_punish_min_production: float = 2.0
    enemy_launch_punish_bonus_weight: float = 0.45
    enemy_launch_punish_max_targets: int = 2

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
REGULAR_CONFIG = StrategyConfig(
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

ABLATION_SUITES = {
    "additive": ADDITIVE_VARIANTS,
    "public_knobs": PUBLIC_KNOB_VARIANTS,
    "front_support": FRONT_SUPPORT_GRID_VARIANTS,
    "rl_score": RL_SCORE_GRID_VARIANTS,
    "combined": COMBINED_PROMISING_VARIANTS,
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
        "early_neutral_bias": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
        ),
        "early_neutral_prod4": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_min_production=4.0,
        ),
        "early_neutral_static_heavy": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_static_multiplier=1.60,
        ),
        "early_neutral_safe_only": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_bonus=4.0,
            early_neutral_safe_bonus=20.0,
            early_neutral_contested_penalty=16.0,
        ),
        "early_neutral_relief_light": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            early_neutral_holdability_relief=0.75,
        ),
        "opening_rotating_filter": from_base(
            REGULAR_CONFIG,
            enable_opening_rotating_neutral_filter=True,
        ),
        "early_bias_plus_rotating_filter": from_base(
            REGULAR_CONFIG,
            enable_early_neutral_bias=True,
            enable_opening_rotating_neutral_filter=True,
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
}
