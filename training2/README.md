# training2: rulebase-aligned policy iteration

`training/` is a PPO prototype. It is useful, but its action space is too far
from the current best rule agent: one target and one ship ratio per owned planet
does not naturally represent coordinated attacks, defense-first moves, multiple
launches from one source, moving-target aiming, or rule-filtered no-op choices.

`training2/` uses a smaller AlphaZero-like loop:

1. Build legal candidate move lists from the strongest rulebase package.
2. Pretrain a neural prior/value model to choose the rulebase candidate.
3. Run self-play against rulebase/checkpoint opponents and train on final result.
4. Keep the rulebase as a candidate generator and let the model rerank candidates.

This is intentionally not a raw angle regressor. The rulebase already encodes
hard physics and tactical constraints; the model should first learn when each
rule-generated action set is valuable, then improve candidate ranking through
self-play.

## Commands

```bash
# Smoke-test feature/action construction.
python -m training2.smoke

# Generate supervised data from the strongest available rulebase agent.
python -m training2.generate_rulebase_data --episodes 100 --out data/training2/rulebase.jsonl

# Behavior cloning pretrain.
python -m training2.train_bc --data data/training2/rulebase.jsonl --out training2/checkpoints/bc.pt

# Ray stage-1 alignment against REGULAR_CONFIG. Gate: 100 eval games, 50% win rate.
python -m training2.train_stage1_ray --config training2/config/default.yaml

# Self-play policy/value improvement.
python -m training2.self_play --checkpoint training2/checkpoints/bc.pt --out training2/checkpoints/selfplay.pt
```

## Fast simulator

`training2.fast_orbit_wars` provides a standalone simulator that mirrors the
official Kaggle Orbit Wars rules without going through Kaggle's schema
validation, deepcopy, and agent-runner layers.

```python
from training2 import make_fast_orbit_wars

env = make_fast_orbit_wars({"episodeSteps": 500, "seed": 0}, keep_history=False)
env.reset(4)
obs = env.steps[-1][0]["observation"]
env.step([[], [], [], []])
```

For maximum throughput in read-only training loops, pass
`copy_observations=False`. This reuses internal state lists in the latest
observation and is not safe for agents that mutate `obs`.

If `numba` is installed, pass `use_numba=True` to compile the fleet movement
collision kernel:

```python
env = make_fast_orbit_wars({"episodeSteps": 500, "seed": 0}, keep_history=False, use_numba=True)
```

Use the comparison/benchmark script after simulator changes:

```bash
uv run python scripts/compare_fast_orbit_wars.py --seeds 3 --players 4 --episode-steps 180
uv run python scripts/compare_fast_orbit_wars.py --seeds 3 --players 4 --episode-steps 180 --numba
```

## Current best rulebase source

The default oracle is
`rulebase.kaggle_public_rl_informed_strategies.agent`, which currently wraps
`RLInformedPublicRuleAgent(**REGULAR_CONFIG)`. If an ablation proves the plain
public package is stronger again, change `training2.rulebase_bridge`.
