"""Fast simulator cloning helpers used by shallow search."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from training2.fast_orbit_wars import make_fast_orbit_wars


def clone_fast_env(env: Any):
    """Clone the repository fast simulator by copying its private state.

    This is deliberately isolated here.  If ``FastOrbitWarsEnv`` grows an
    official clone method later, only this file should need to change.
    """

    cfg = dict(getattr(env.configuration, "__dict__", {}))
    cloned = make_fast_orbit_wars(
        cfg,
        debug=getattr(env, "debug", True),
        keep_history=getattr(env, "keep_history", False),
        use_numba=getattr(env, "use_numba", False),
        copy_observations=getattr(env, "copy_observations", True),
    )
    cloned.num_agents = env.num_agents
    cloned.done = env.done
    cloned._step = env._step
    if hasattr(env, "_rng"):
        cloned._rng = deepcopy(env._rng)
    cloned._state_planets = deepcopy(env._state_planets)
    cloned._initial_planets = deepcopy(env._initial_planets)
    cloned._fleets = deepcopy(env._fleets)
    cloned._comets = deepcopy(env._comets)
    cloned._comet_planet_ids = deepcopy(env._comet_planet_ids)
    cloned.steps = deepcopy(env.steps[-1:]) if getattr(env, "steps", None) else []
    return cloned
