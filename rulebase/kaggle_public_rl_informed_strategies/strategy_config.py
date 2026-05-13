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
    en_route_skip_owned_ratio: float = 0.75
    enable_single_attacks: bool = True
    enable_coop_attacks: bool = True
    enable_reinforcements: bool = True
    enable_sun_avoidance: bool = True

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
REGULAR_CONFIG = StrategyConfig(
    target_candidate_limit=2,
    min_ships_mine_attack=12,
    enemy_owned_production_buffer_turns=4,
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
        "regular_config": REGULAR_CONFIG.to_agent_kwargs(),
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
