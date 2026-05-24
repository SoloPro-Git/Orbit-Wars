# tinyPPO Regular-Aware Candidate Policy

Date: 2026-05-24

## Goal

The current tinyPPO model can learn to launch fleets, but regular wins because
it does not search the raw action space. It narrows every decision with game
mechanics: target ROI, required ships, source reserves, moving-target aiming,
sun filtering, and holdability checks.

This experiment keeps the tinyPPO network small and changes the action
representation first:

- no angle head; angles are still computed from map coordinates and moving
  target interception;
- each owned source still has up to 3 launch slots;
- target selection is masked to a small set of legal, high-ROI candidates;
- ship amount is a categorical multiplier around a rule-computed required ship
  count instead of an arbitrary source-fraction Beta sample.

This is an architecture/action-space change, so it should start from a fresh
checkpoint. Old `ship_buckets=0` checkpoints are not semantically compatible
with `ship_buckets=5`.

## Regular Strategy Summary

Regular is strong mostly because its action space is structured:

- It reserves ships for planets under attack before attacking.
- It scores targets by distance, production, required ships, ETA, enemy-owned
  production, early neutral value, recapture value, and holdability.
- It computes required ships as target ships plus production before arrival,
  plus contested/race margins.
- It supports multi-source attacks when one source cannot pay the cost.
- It tracks recent captures/losses and fleets already in flight.
- It rejects sun paths and uses tick-by-tick moving-target interception.

tinyPPO should not be asked to rediscover all of that from sparse win/loss
reward while sampling arbitrary ship fractions.

## Implemented Delta

### Candidate Target Mask

`candidate_target_mask(obs, player, top_k=6)` first applies the safe path mask,
then scores non-owned targets per source with a regular-like heuristic:

- high production is good;
- shorter distance/ETA is good;
- lower required ship cost is good;
- early high-production neutral captures are boosted;
- moving targets with long ETA are penalized.

Only the top candidates per source remain selectable. If a source has no
candidate, launch logits for that source are forced to no-launch.

### Required-Ship Buckets

When `--ship-buckets 5` is enabled, the ship head outputs categorical logits
over these multipliers:

```text
0.75x, 1.00x, 1.25x, 1.50x, 2.00x
```

The environment helper computes `required_ships(obs, player, source, target)`.
The chosen bucket sends:

```text
ceil(required_ships * multiplier)
```

clamped by the source's remaining ships for that turn.

### PPO Compatibility

The checkpoint model config now records `ship_buckets`. PPO supports both:

- `ship_buckets=0`: legacy Beta ship fraction.
- `ship_buckets>0`: categorical required-ship multiplier.

## Why Start From 0

Resume is not recommended for this run because the old policy learned:

- target logits over the full safe map, not the candidate set;
- continuous ship fractions, not required-ship buckets;
- a different log-prob distribution for PPO.

Loading old weights would mix incompatible action semantics. A fresh run gives
cleaner diagnostics: if rollout launches, candidate hit rate, and eval improve,
the gain is attributable to the action-space delta.

## Expected Signs Of Life

Watch these before trusting eval winrate:

- `mean_launches` should be nonzero but lower than the raw full-map policy.
- Generated actions should have no sun/out-of-bounds/miss problems.
- Early random winrate should improve with fewer wasted fleets.
- Against regular, losses should show better economy: fewer over-sends and
  fewer low-value neutral captures.

## Next Deltas, Not In This Run

Do not add all of these at once:

- source reserve mask for high-production/frontline planets;
- explicit own-target reinforcement candidates;
- multi-source coordinated attack action;
- imitation warm start from regular action labels;
- larger hidden size or attention.

The next best single delta is probably source reserve features/mask, because
regular often wins by punishing sources that were emptied too aggressively.
