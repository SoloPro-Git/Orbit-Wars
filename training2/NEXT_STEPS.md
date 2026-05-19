# training2 Next Steps

## Why Restart

The current training2 model inputs changed:

- `PLANET_DIM` now includes fleet pseudo-entities and entity type flags.
- `ACTION_DIM` now includes tactical candidate features inspired by promoted
  rulebase history.
- Candidate generation now injects a small number of source-safe ROI actions in
  addition to rulebase-derived candidates.

Old checkpoints can be used as partial warm starts through compatible loading,
but old in-flight training should not continue because it was launched with the
previous feature schema.

## Current Training Plan

1. Generate a fresh offline stage1 dataset using the new feature schema.
   - Write samples under
     `data/training2/stage1_tactical_entities_20260519` so generated data stays
     outside git.
   - Use the fast simulator with numba and the promoted regular oracle.
   - Start with a 3000-episode bootstrap dataset because the richer entity and
     action features make each JSON row much larger than the previous 10k run.

2. Run a fresh stage1 alignment from that offline dataset.
   - Start from `training2/checkpoints/stage1_regular_current_20260518/latest.pt`
     as a compatible warm start.
   - Write new outputs to
     `training2/checkpoints/stage1_tactical_entities_20260519`.
   - Load the offline dataset in trainer shards rather than reading the full
     dataset into every trainer.
   - Connect to the multi-node Ray cluster and use fractional GPU trainer
     actors, three trainers per GPU, to keep the full H20 pool saturated during
     stage1 and stage1.5.
   - Keep per-trainer updates low enough that total global updates stay close
     to the earlier 7-trainer run.
   - Keep model proposals disabled in stage1 eval.

3. Watch the first 10-20 iterations.
   - Expected: lower initial oracle top1 than the old run because
     `planet_proj` and `action_proj` are reinitialized.
   - Healthy sign: `align/new_oracle_top1` climbs steadily while proposal loss
     remains finite.
   - If loss or proposal metrics explode, temporarily disable heuristic
     candidates in `build_candidates` during offline stage1 and retrain only the
     representation first.

4. After stage1 recovers alignment, run stage1.5 advantage training.
   - Resume from the new stage1 checkpoint.
   - Connect to the multi-node Ray cluster at `10.0.104.198:6380`.
   - Use fast simulator with numba.
   - Schedule rollout actors on the remote `rollout_cpu` resource and use
     fractional GPU trainers across the full remote GPU pool.
   - Keep paired-margin advantage mode.
   - Track `selected_proposal`, `fallback_targets`, and
     `eval/win_rate_vs_regular`.

## Rulebase Signals To Keep Learning From

Promoted regular history points to four high-value signals:

- 4P early source guard: avoid sending from production 4+ sources when nearby
  enemies can profitably recapture.
- Third-party tail capture: follow enemy fleets into high-production targets
  with small, delayed, source-safe sends.
- Leader containment: prefer local pressure on the production leader in 4P
  without regressing 2P.
- Post-capture hold: value production only after our capture actually lands
  when estimating whether a newly captured planet can survive.

## Next Code Improvements

1. Add explicit candidate tags/features.
   - rulebase full action
   - single move
   - drop-one move
   - heuristic ROI
   - model proposal
   This will let the ranker learn when each generator is trustworthy.

2. Add a richer fleet/target relation feature.
   - For each candidate target, estimate friendly and enemy arrival buckets.
   - Include delayed-tail opportunity score and post-capture hold score.

3. Improve rollout credit assignment.
   - Add short-horizon deltas for score share, production share, captures, and
     source-loss events.
   - Keep final paired-margin value as the main signal, but use local deltas as
     per-row policy weights.

4. Add evaluation slices.
   - 2P regression vs regular.
   - 4P mixed champion pool.
   - Tail-capture opportunity seeds.
   - Early 4P source-guard seeds.

5. Package only after a gate passes.
   - Stage1 alignment recovered.
   - Stage1.5 eval beats regular on 4P mix.
   - No 2P regression versus regular over the fixed seed suite.
