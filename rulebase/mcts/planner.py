"""Small rolling-horizon MCTS selector over regular rule-agent configurations.

This planner treats each child of the root as "use this regular-family config
for the current move".  It then rolls the game forward for a few turns in the
fast simulator while opponents are sampled from a small pool of plausible
regular configs.  No neural model is trained or loaded.
"""

from __future__ import annotations

import copy
import math
import random
from dataclasses import dataclass, field
from dataclasses import fields as dataclass_fields
from typing import Any

from rulebase.kaggle_public_rl_informed_strategies.rl_informed_agent import (
    RLInformedPublicRuleAgent,
)
from rulebase.kaggle_public_rl_informed_strategies.strategy_config import (
    REGULAR_CONFIG,
)
from training2.fast_orbit_wars import FastOrbitWarsEnv

from .configs import DEFAULT_CANDIDATE_NAMES, DEFAULT_OPPONENT_MODEL_NAMES, config_params, named_configs


_ALLOWED_AGENT_KWARGS = {field.name for field in dataclass_fields(RLInformedPublicRuleAgent)}


@dataclass(frozen=True, slots=True)
class MCTSPlannerConfig:
    candidate_names: tuple[str, ...] = DEFAULT_CANDIDATE_NAMES
    opponent_model_names: tuple[str, ...] = DEFAULT_OPPONENT_MODEL_NAMES
    simulations: int = 3
    rollout_depth: int = 3
    exploration_c: float = 1.2
    production_weight: float = 2.5
    planet_weight: float = 6.0
    enemy_score_weight: float = 0.25
    seed: int = 20260522
    use_numba: bool = False


@dataclass(slots=True)
class _Stats:
    visits: int = 0
    total_value: float = 0.0

    @property
    def mean(self) -> float:
        return self.total_value / self.visits if self.visits else 0.0


@dataclass(slots=True)
class MCTSConfigSelector:
    config: MCTSPlannerConfig = field(default_factory=MCTSPlannerConfig)
    _rng: random.Random = field(init=False, repr=False)
    _candidate_params: dict[str, dict] = field(init=False, repr=False)
    _live_agents: dict[str, RLInformedPublicRuleAgent] = field(init=False, repr=False)
    _fallback_agent: RLInformedPublicRuleAgent = field(init=False, repr=False)
    _opponent_names: tuple[str, ...] = field(init=False, repr=False)
    _turns_seen: int = field(init=False, repr=False, default=0)

    def __post_init__(self) -> None:
        self._rng = random.Random(self.config.seed)
        self._candidate_params = named_configs(self.config.candidate_names)
        if not self._candidate_params:
            self._candidate_params = {"regular": REGULAR_CONFIG.to_agent_kwargs()}
        self._live_agents = {
            name: self._make_agent(params) for name, params in self._candidate_params.items()
        }
        self._fallback_agent = self._live_agents.get("regular") or self._make_agent(
            REGULAR_CONFIG.to_agent_kwargs()
        )
        self._opponent_names = tuple(
            name
            for name in self.config.opponent_model_names
            if config_params(name) is not None
        ) or ("regular",)

    def act(self, obs: Any, configuration: Any = None) -> list:
        current_step = _current_step(obs, self._turns_seen)
        candidate_actions = self._candidate_actions(obs)
        if not candidate_actions:
            self._turns_seen = current_step + 1
            return self._safe_act(self._fallback_agent, obs)
        if len(candidate_actions) == 1 or self.config.simulations <= 0 or self.config.rollout_depth <= 0:
            self._turns_seen = current_step + 1
            return candidate_actions[0][1]

        player = int(_obs_get(obs, "player", 0))
        stats = {name: _Stats() for name, _action in candidate_actions}
        for sim_idx in range(max(self.config.simulations, len(candidate_actions))):
            name, action = self._select_child(stats, candidate_actions, sim_idx)
            value = self._rollout_value(obs, player, action, sim_idx, current_step)
            stats[name].visits += 1
            stats[name].total_value += value

        best_name, best_action = max(
            candidate_actions,
            key=lambda item: (stats[item[0]].mean, stats[item[0]].visits, -candidate_actions.index(item)),
        )
        self._turns_seen = current_step + 1
        return best_action

    def _candidate_actions(self, obs: Any) -> list[tuple[str, list]]:
        seen: set[tuple] = set()
        actions: list[tuple[str, list]] = []
        for name, agent in self._live_agents.items():
            action = self._safe_act(agent, obs)
            key = _action_key(action)
            if key in seen:
                continue
            seen.add(key)
            actions.append((name, action))
        return actions

    def _select_child(
        self,
        stats: dict[str, _Stats],
        candidates: list[tuple[str, list]],
        sim_idx: int,
    ) -> tuple[str, list]:
        for name, action in candidates:
            if stats[name].visits == 0:
                return name, action
        total = sum(stat.visits for stat in stats.values()) + 1
        return max(
            candidates,
            key=lambda item: stats[item[0]].mean
            + self.config.exploration_c * math.sqrt(math.log(total) / stats[item[0]].visits),
        )

    def _rollout_value(
        self,
        obs: Any,
        player: int,
        root_action: list,
        sim_idx: int,
        current_step: int,
    ) -> float:
        env = _env_from_observation(obs, step_override=current_step, use_numba=self.config.use_numba)
        agents = self._make_rollout_agents(env.num_agents, player, sim_idx)
        actions = []
        for pid in range(env.num_agents):
            if pid == player:
                actions.append(copy.deepcopy(root_action))
            else:
                actions.append(self._safe_act(agents[pid], _with_step(env.steps[-1][pid]["observation"], env._step)))
        env.step(actions)

        depth = 1
        while not env.done and depth < self.config.rollout_depth:
            actions = [
                self._safe_act(agents[pid], _with_step(env.steps[-1][pid]["observation"], env._step))
                for pid in range(env.num_agents)
            ]
            env.step(actions)
            depth += 1
        return self._evaluate(env.steps[-1][player]["observation"], player)

    def _make_rollout_agents(self, num_agents: int, player: int, sim_idx: int) -> list[RLInformedPublicRuleAgent]:
        agents = []
        for pid in range(num_agents):
            if pid == player:
                name = "regular"
            else:
                name = self._opponent_names[(sim_idx + pid) % len(self._opponent_names)]
            params = config_params(name) or REGULAR_CONFIG.to_agent_kwargs()
            agents.append(self._make_agent(params))
        return agents

    def _evaluate(self, obs: Any, player: int) -> float:
        player_score = 0.0
        enemy_best = 0.0
        active_players = _infer_num_agents(obs)
        scores = [0.0 for _ in range(active_players)]
        production = [0.0 for _ in range(active_players)]
        planets_count = [0 for _ in range(active_players)]

        for planet in _obs_get(obs, "planets", []) or []:
            owner = int(planet[1])
            if owner < 0 or owner >= active_players:
                continue
            scores[owner] += float(planet[5])
            production[owner] += float(planet[6])
            planets_count[owner] += 1
        for fleet in _obs_get(obs, "fleets", []) or []:
            owner = int(fleet[1])
            if 0 <= owner < active_players:
                scores[owner] += float(fleet[6])

        for pid in range(active_players):
            value = (
                scores[pid]
                + self.config.production_weight * production[pid]
                + self.config.planet_weight * planets_count[pid]
            )
            if pid == player:
                player_score = value
            else:
                enemy_best = max(enemy_best, value)
        return player_score - self.config.enemy_score_weight * enemy_best

    @staticmethod
    def _make_agent(params: dict) -> RLInformedPublicRuleAgent:
        return RLInformedPublicRuleAgent(
            **{key: value for key, value in dict(params).items() if key in _ALLOWED_AGENT_KWARGS}
        )

    @staticmethod
    def _safe_act(agent: RLInformedPublicRuleAgent, obs: Any) -> list:
        try:
            return agent.act(obs)
        except Exception:
            return []


def _obs_get(obs: Any, key: str, default: Any = None) -> Any:
    if isinstance(obs, dict):
        return obs.get(key, default)
    return getattr(obs, key, default)


def _with_step(obs: Any, step: int) -> Any:
    try:
        obs.step = step
    except Exception:
        pass
    return obs


def _current_step(obs: Any, fallback: int) -> int:
    raw_step = _obs_get(obs, "step", None)
    if raw_step is None:
        return fallback
    try:
        return int(raw_step)
    except (TypeError, ValueError):
        return fallback


def _action_key(action: list) -> tuple:
    normalized = []
    for move in action or []:
        if len(move) != 3:
            continue
        normalized.append((int(move[0]), round(float(move[1]), 5), int(move[2])))
    return tuple(sorted(normalized))


def _infer_num_agents(obs: Any) -> int:
    max_player = int(_obs_get(obs, "player", 0))
    for planet in _obs_get(obs, "planets", []) or []:
        owner = int(planet[1])
        if owner >= 0:
            max_player = max(max_player, owner)
    for fleet in _obs_get(obs, "fleets", []) or []:
        owner = int(fleet[1])
        if owner >= 0:
            max_player = max(max_player, owner)
    return max(2, max_player + 1)


def _env_from_observation(
    obs: Any,
    step_override: int | None = None,
    use_numba: bool = False,
) -> FastOrbitWarsEnv:
    env = FastOrbitWarsEnv(
        {"episodeSteps": 500, "seed": 0},
        keep_history=False,
        copy_observations=True,
        use_numba=use_numba,
    )
    env.num_agents = _infer_num_agents(obs)
    env.done = False
    env._step = _current_step(obs, 0) if step_override is None else int(step_override)
    env._angular_velocity = float(_obs_get(obs, "angular_velocity", 0.0) or 0.0)
    env._state_planets = copy.deepcopy(_obs_get(obs, "planets", []) or [])
    env._initial_planets = copy.deepcopy(_obs_get(obs, "initial_planets", []) or env._state_planets)
    env._fleets = copy.deepcopy(_obs_get(obs, "fleets", []) or [])
    env._comets = copy.deepcopy(_obs_get(obs, "comets", []) or [])
    env._comet_planet_ids = list(copy.deepcopy(_obs_get(obs, "comet_planet_ids", []) or []))
    env._next_fleet_id = int(_obs_get(obs, "next_fleet_id", len(env._fleets)))
    env._initial_geom = env._build_initial_geom(env._initial_planets)
    env.info["seed"] = int(_obs_get(obs, "seed", 0) or 0)
    env.steps = [env._make_step([[] for _ in range(env.num_agents)], [0] * env.num_agents, ["ACTIVE"] * env.num_agents)]
    return env
