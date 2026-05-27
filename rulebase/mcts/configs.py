"""Candidate regular-family configs used by the MCTS rollout selector."""

from __future__ import annotations

from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    ABLATION_SUITES,
    CHAMPION_OPPONENT_VARIANTS,
    REGULAR_CONFIG,
)


DEFAULT_CANDIDATE_NAMES = (
    "regular",
    "tail_m2_max14_net7_overpay4_p4lowhome_active4_regular",
    "tail_m2_max12_net7_overpay4_p4lowhome_active4_p4seed_prod4_regular",
    "unread260520_p4relay_savings5_window45_bonus70_regular",
    "mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_regular",
)

DEFAULT_OPPONENT_MODEL_NAMES = (
    "regular",
    "tail_m2_max14_net7_overpay4_p4lowhome_active4_regular",
    "unread260520_p4relay_savings5_window45_bonus70_regular",
)


def named_configs(names: tuple[str, ...] = DEFAULT_CANDIDATE_NAMES) -> dict[str, dict]:
    configs: dict[str, dict] = {}
    for name in names:
        params = config_params(name)
        if params is not None:
            configs[name] = dict(params)
    if "regular" not in configs:
        configs["regular"] = REGULAR_CONFIG.to_agent_kwargs()
    return configs


def opponent_model_names(names: tuple[str, ...] = DEFAULT_OPPONENT_MODEL_NAMES) -> tuple[str, ...]:
    return tuple(name for name in names if config_params(name) is not None)


def config_params(name: str) -> dict | None:
    params = CHAMPION_OPPONENT_VARIANTS.get(name)
    if params is not None:
        return dict(params)
    for suite in ABLATION_SUITES.values():
        params = suite.get(name)
        if params is not None:
            return dict(params)
    if name == "regular":
        return REGULAR_CONFIG.to_agent_kwargs()
    return None
