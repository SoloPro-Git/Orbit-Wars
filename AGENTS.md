# Codex Project Notes

## Local Orbit Wars Environment

For local development, tests, data generation, and training smoke checks, prefer
the repository fast simulator instead of Kaggle's official environment runner.

Use:

```python
from training2 import make_fast_orbit_wars

env = make_fast_orbit_wars(
    {"episodeSteps": 500, "seed": 0},
    keep_history=False,
    use_numba=True,
)
env.reset(4)
```

Avoid using this by default in local loops:

```python
from kaggle_environments import make

env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": 0}, debug=True)
```

The official Kaggle environment should still be used when explicitly validating
compatibility, reproducing Kaggle behavior, or investigating suspected simulator
drift.

## Compatibility Check

After changing simulator rules or collision logic, compare against the official
environment:

```bash
uv run python scripts/compare_fast_orbit_wars.py --seeds 3 --players 4 --episode-steps 180 --numba
uv run python scripts/compare_fast_orbit_wars.py --seeds 3 --players 2 --episode-steps 180 --numba
```

For broader confidence before large training runs:

```bash
uv run python scripts/compare_fast_orbit_wars.py --seeds 100 --players 4 --episode-steps 500 --bench-games 10 --numba
uv run python scripts/compare_fast_orbit_wars.py --seeds 100 --players 2 --episode-steps 500 --bench-games 10 --numba
```

## Performance Notes

`use_numba=True` compiles the fleet movement collision kernel. The first call has
JIT startup cost; long local training/data generation jobs benefit the most.

`copy_observations=False` can improve throughput in read-only loops, but only use
it when agents and feature builders do not mutate `obs`.

