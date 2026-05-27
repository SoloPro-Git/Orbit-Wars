# Rulebase MCTS

This directory adds no-training selectors on top of the existing regular-family
rule agents.

The default `agent` currently uses a fast rule-fusion table:

- 2P: `p2move_regular`
- 4P: `regular`

The 2P choice came from fast-simulator probes against current `regular`.  A
first 4P low-connector candidate was negative in a larger paired probe, so 4P
is intentionally conservative until a mode-specific 4P candidate clears the
same bar. The rollout planner remains available as `MCTSConfigSelector` for
further online-search experiments.

Latest local probes:

- 2026-05-22 paired 10 games/seat, 180 steps:
  - 2P baseline regular: `11` strict wins, `5` losses, `4` same-reward draws,
    top reward `15/20`, average rank `1.250`.
  - 2P fusion with `p2move_regular`: `17` strict wins, `3` losses, `0` draws,
    top reward `17/20`, average rank `1.150`.
  - Paired reward movement: `+4 / -2 / =14`, rank movement `+4 / -2`.
- 2026-05-22 paired 10 games/seat, 180 steps:
  - 4P baseline regular: `11` strict wins, `27` losses, `2` same-reward draws,
    top reward `13/40`, average rank `1.675`.
  - 4P `p2move_lowconn_regular`: `11` strict wins, `29` losses, `0` draws,
    top reward `11/40`, average rank `1.725`.
  - Paired reward movement: `+7 / -9 / =24`, rank movement `+7 / -9`.
- 2026-05-22 4P candidate sweep over `myreplay_plan045` and
  `myreplay_plan046`, 3 games/seat, 160 steps:
  - Baseline regular scored `6/12` top, average rank `1.500`.
  - No tested 4P candidate beat baseline; the best group scored `4/12` top,
    average rank `1.667`, with negative paired movement.

The rollout planner does not learn weights.  At each real turn it:

1. asks several historical `regular` configs for candidate moves;
2. copies the current observation into `training2.fast_orbit_wars`;
3. rolls forward a small number of turns while opponents are sampled from
   plausible regular configs;
4. chooses the candidate with the best simulated heuristic value.

Entrypoint:

```python
from rulebase.mcts import agent
```

Smoke check:

```bash
uv run python - <<'PY'
from rulebase.mcts import agent
from training2 import make_fast_orbit_wars

env = make_fast_orbit_wars({"episodeSteps": 30, "seed": 0}, keep_history=False, use_numba=False)
env.run([agent, agent])
print(env.steps[-1][0]["reward"], env.steps[-1][1]["reward"])
PY
```

MCTS itself does not require model training.  Training is only needed if we add
a learned policy/value model to replace the hand-written rollout agents or the
heuristic evaluator.
