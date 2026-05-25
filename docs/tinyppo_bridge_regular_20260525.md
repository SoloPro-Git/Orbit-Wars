# tinyPPO Regular Bridge Notes

Date: 2026-05-25

Goal: record a later bridge idea for controlled experiments after imitation
pretraining. This is not the phase-1 BC objective.

Phase-1 BC objective: make tinyPPO choose source/target/ship/action-count labels
that match `regular` on the same observations. Online winrate against `regular`
is diagnostic only in this phase; it should not be used to force destructive
action filtering or checkpoint promotion. The bridge below belongs to a later
controlled experiment, after a regular-like checkpoint exists.

## Strategy

Use `regular` as the legal action anchor. The model only gates source planets:

- regular still chooses target, angle, and ship count;
- tinyPPO reads source launch probabilities from the BC checkpoint;
- low-confidence regular source actions can be removed;
- never ask the model to output angle;
- do not use the model target/ship heads in this bridge.

This follows the earlier lesson that source-only bridge is safer than full
proposal projection. The added guardrails are:

- only filter when regular already has at least 4 actions in that step;
- keep at least 1 action;
- drop at most 1 action per step;
- drop at most 15% of actions per step;
- use max over the 3 source slots as the source probability.

## 40x500 Fastenv Eval

Command:

```bash
.venv/bin/python -m tinyPPO.eval_bridge_ray \
  --ray-address auto \
  --checkpoint tinyPPO/regular_bc.pt \
  --games 40 \
  --games-per-task 1 \
  --workers 40 \
  --cpus-per-worker 1 \
  --gpus-per-worker 0 \
  --device cpu \
  --episode-steps 500 \
  --thresholds 0.5,0.6,0.7 \
  --apply-probs 1.0 \
  --max-source-drops-list=1 \
  --max-drop-frac 0.15 \
  --min-anchor-actions-to-filter 4 \
  --reduce max
```

Results:

| Variant | W/L/D | Winrate | Nonloss | Kept | Attempted |
| --- | ---: | ---: | ---: | ---: | ---: |
| regular anchor | 20/20/0 | 0.500 | 0.500 | 1.000 | 0.000 |
| bridge threshold 0.5 | 23/16/1 | 0.575 | 0.600 | 0.997 | 0.118 |
| bridge threshold 0.6 | 15/24/1 | 0.375 | 0.400 | 0.996 | 0.093 |
| bridge threshold 0.7 | 20/19/1 | 0.500 | 0.525 | 0.990 | 0.113 |

Current candidate:

```text
threshold=0.5
apply_prob=1.0
max_source_drops=1
max_drop_frac=0.15
min_anchor_actions_to_filter=4
reduce=max
```

## Interpretation

The bridge is deliberately conservative. The best setting removed only about
0.3% of regular actions, but it still improved this 40-game seed set from
20/20/0 to 23/16/1. More aggressive thresholds are not monotonic: `0.6` was
bad, and `0.7` only matched the anchor. The next useful step is not more BC
epochs; it is to diagnose the few removed source actions and learn a better
"when is it safe to delete this regular source" classifier.
