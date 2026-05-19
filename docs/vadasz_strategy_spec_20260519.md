# Vadasz Replay Strategy Spec

Source: `data/myreplay/vadasz_replays`, 84 games, 52 wins and 32 losses.
This is a strategy-level read of `docs/vadasz_replays_deep_review.md`, focused on
how Vadasz builds advantage, turns deficits, and where the same style fails.

## Executive Read

Vadasz does not win because it simply launches more. It wins because early
launches form a production chain:

1. Take a high-production neutral or thin enemy point.
2. Immediately use support or follow-up waves so the new point is not an empty shell.
3. Treat the new point as a forward source within a few ticks.
4. Start hitting enemy prod3-prod5 points before all neutral space is consumed.

Local stats from the replay parser:

- Vadasz wins: 0-50 avg 24.8 launches; 51-120 avg 94.9 launches.
- Vadasz losses: 0-50 avg 17.9 launches; 51-120 avg 57.2 launches.
- Wins at 51-120: enemy-target launches avg 24.2, friendly support avg 46.6.
- Losses at 51-120: enemy-target launches avg 18.2, friendly support avg 26.1.
- All 52 wins attack at least one enemy prod>=3 point by t120. First such capture
  averages around t47.6; 50/52 happen by t80.

The important distinction: Vadasz support is usually not passive defense. It is
logistics for a forward source.

## Advantage Pattern

### Stage A: 0-20, High-ROI First Capture

The first real target is usually a neutral point with high production and modest
guard. In the explorer pass, 46/52 wins had a neutral first identified target,
with known first targets averaging prod 3.35 and guard about 14.3.

Representative games:

- `episode-76982204`: t2 `P19 -> P23`, neutral prod5 guard 11, captured at t4.
- `episode-76981696`: t7 captures neutral `P16 prod3`; t8 captures `P0 prod4`;
  t12 has three owned high-value points.
- `episode-77005601`: t10 sends to neutral `P14 prod5`; it becomes the local anchor.

Spec implication:

- Opening target score should favor `production / (guard + ETA cost)`.
- `prod>=4` with guard <=25 should receive a strong early bonus.
- Opening send ratios are often large, around 75-95%, with only a small reserve.

### Stage B: 10-40, New Point Becomes a Hub

Vadasz frequently supports a newly captured high-production point or immediately
uses it as a source. The replay analysis found 939 cases where a captured point
launched again within 25 ticks; median delay was 4 ticks, and 631 were within 5
ticks.

Examples:

- `76981696`: after taking `P16`, t11 `P24 -> P16`, t13 `P20 -> P16`, then
  t16 `P16 -> enemy P26 prod5`.
- `76982204`: after t4 `P23 prod5`, t4 sends more to `P23`; t7/t10 `P23 -> P15 prod4`;
  t17 `P23 -> enemy P17 prod4`.
- `77005601`: t22 and t26 reinforce `P14 prod5`; later this chain participates in
  taking enemy prod5 points.

Spec implication:

- A pure "support recent high-prod hub" rule was tested and lost. The missing part
  is that support must be tied to a next-hop plan.
- We need a `chain_source` rule, not a standalone hub-defense rule.

### Stage C: 30-120, Enemy High-Production Pressure

Vadasz does not wait for all neutral planets. Once it has a forward chain, it
starts taking enemy prod3-prod5 points.

Examples:

- `76981696`: t16 hits enemy `P26 prod5`; t33 hits enemy `P1 prod4` and `P27 prod5`.
- `76982204`: t17 hits enemy `P17 prod4`; t32/t35/t38 continues into enemy prod4/prod5.
- `76982683`: t25/t26/t31 repeatedly pressures enemy/front prod5 `P17`.

Spec implication:

- When own planet count >=3 or a recent forward prod>=3 point exists, evaluate
  enemy prod>=3 targets even before neutral phase is complete.
- Prefer enemy high-prod points with guard <=45, ETA <=12, or multiple friendly
  sources near the target.

## Comeback Pattern

Comeback wins are not conservative recoveries. They are high-production raids,
third-party captures, and heavy support into contested front points.

Important comeback games:

- `76987213`: t80 prod -9 and score -85; by t120 prod +9. Key gains: t65 `P12 prod4`,
  t94/t102 `P4 prod5`, t96 `P14 prod4`, t113/t116 `P15/P12 prod4`.
- `77011689`: t40 score -109/prod -3, t80 score -318/prod -3. It flips by taking
  t52 `P5 prod5`, t55 `P12 prod4`, t69 `P13 prod4`, t84 `P9 prod5`,
  t113 `P6 prod5`, t119 `P10 prod5`. This is the cleanest non-tail comeback.
- `77019883`: t80 score -129/prod -5; t120 score +206/prod +21. Between t80-t120
  it takes `P13 prod4`, `P6 prod3`, `P5 prod3`, `P12 prod4`.
- `77000515`: t80 score -110; t120 score +298/prod +16. It chains t80 `P14 prod4`,
  t82 `P13 prod4`, t83 `P17 prod4`, t88 `P10 prod5`, t89 `P18 prod4`,
  t106 `P12 prod4`.

Mechanisms:

- Third-party tail matters in 4P, but it is not the whole story.
- The stable comeback core is `enemy_or_recently_flipped prod>=3`.
- Support volume remains high during comebacks. Example: `77011689` t80-t120 has
  50 launches, 27 of them friendly support.
- Repeatedly contested high-prod points are worth fighting. Low-prod repeated
  points are the ones that need stop-loss.

Comeback mode spec:

- Trigger: `step >= 40` and either `score_delta < -50` or `prod_delta < -3`.
- Target priority:
  1. enemy or recently flipped prod>=3,
  2. enemy prod>=2 with low ships and close ETA,
  3. neutral prod>=3 low guard,
  4. low-production harassment only if it opens a chain.
- Add bonus if a non-self ownership flip occurred on the target within 12 ticks.
- Continue fighting prod>=3 front points if a follow-up wave can leave a real
  garrison; penalize prod<=1/prod2 feed points after repeated failed trades.
- Exit when score and production are non-negative for about 10 ticks.

## Failure Pattern

Vadasz losses look a lot like Solo regular losses: the bot can launch, but the
front production chain gets cut and then it feeds the same points.

Examples:

- `76988077`: t60 still strong, 12 stars / 30 prod / prod +7. Then t65 loses
  `P6 prod4`, t67 `P22 prod3`, t71 `P17 prod3`, t73 `P18 prod3`, t79 `P5 prod4`.
  By t80 prod is -17. Repeated feed points: `P17` 7 flips, `P13` 6, several 5.
- `76989413`: t40 prod +3, then t44 loses `P19 prod5`; t53 loses `P3 prod3`;
  t57/t58 loses multiple chain points. `P15` and `P19` flip 12 times each.
- `76991191`: takes several prod5 points early, but t53/t62/t74/t75/t78 loses
  prod5 anchors with huge enemy post-capture stacks. This needs pre-arrival wave
  preservation, not after-the-fact recapture.
- `77027248` in 2P: t120 production still tied, but t104/t105 loses `P14/P12 prod5`
  with large enemy landings, then t135 loses another prod5. This is midgame
  front-base attrition, not opening failure.

Spec implication:

- `recent_loss_recapture_min_production=4` misses prod2/prod3 connector planets.
- Existing value defense is too reactive for huge incoming waves.
- Candidate broadening globally was bad, but broadening in collapse/recapture mode
  may be different.

## Experiment Plan

### Plan 005: Enemy Wave Preserve Production

Goal: stop the "enemy lands with 90-190 ships on our prod4/prod5" failure.

Mechanism:

- For own prod>=4, inspect incoming enemy fleets and nearby enemy sendable mass.
- If projected enemy post-capture stack would exceed a threshold, preserve this
  planet before neutral expansion or enemy launch punish.
- Send from nearest non-critical own planets, but only if arrival beats the wave.

Initial variants:

- 4P: prod>=4, horizon 35, enemy_post_capture >=30, margin 10, max_send 48.
- 2P: prod>=3, horizon 40, enemy_post_capture >=45, margin 12, max_send 56.

### Plan 006: Dynamic Front-Base Recapture Value

Goal: make prod2/prod3 connector points matter when they are part of the chain.

Mechanism:

- Lower recapture min production from 4 to 2 only when:
  - 2P, or
  - planet count deficit >=3, or
  - lost point is near two own planets / near a recent high-prod anchor.
- Keep ordinary recapture bias unchanged for safe states.

Initial variants:

- 2P: recapture prod>=2, window 45, bonus 25, prod_weight 4.
- 4P collapse-only: prod>=2 if lost two planets within 20 ticks.

### Plan 007: Chain Source Next-Hop

Goal: model the successful Vadasz hub behavior without the failed standalone
support rule.

Mechanism:

- If a prod>=3 point was captured within 20 ticks, treat it as a preferred source
  for nearby prod>=3 neutral/enemy targets.
- If next-hop target can be captured but post-capture would be thin, schedule a
  second support/capture wave from a different source.
- Do not support the hub unless a next-hop target exists or an enemy wave is likely.

This directly fixes why `vadasz_plan004_high_prod_hub_support` failed: it moved
ships into hubs without requiring the hub to convert those ships into pressure.

### Plan 008: Contested Stop-Loss With High-Prod Exception

Goal: stop feeding low-value repeated points while still allowing Vadasz-style
high-prod front fights.

Mechanism:

- Track flips per planet in a rolling 40-tick window.
- If flips >=3 and production <=2, reduce target score unless one of these holds:
  - it is a connector to a prod>=4 own/enemy point,
  - our planned send leaves post-capture >= `max(8, prod*3)`,
  - comeback mode is active and the target is the only nearby enemy source.
- If production >=3, do not penalize by default; require holdability check instead.

## Recommended Order

1. Implement Plan 005 first. It directly targets the clearest loss examples and is
   less likely to disturb opening tempo.
2. Then Plan 006 for 2P and collapse-only 4P.
3. Then Plan 007, because it needs more careful source/target coordination.
4. Finally Plan 008 if replay tests show repeated low-prod feeding remains common.

### Plan 009: Mobile Relay ETA Shortcut

Goal: test whether far high-production targets should be attacked through fast
moving planets or comets as relays.

Result:

- Pure ETA-saving relay was not robust. The first small 4P run looked positive,
  but the larger confirmation regressed against regular.
- Interpretation: Vadasz is not mainly winning by route-shortening. The useful
  pattern is making a newly captured high-production moving planet become the
  next pressure source.

### Plan 010: Mobile High-Prod Forward Base

Goal: capture/use fast or moving prod>=4 planets as forward production bases,
then immediately let those new bases attack prod>=4 targets.

Promoted variant:

- `p4_mobile_base_prod4_goal4_moderate`
- Enabled only in 4P, steps 0-140.
- Requires source/relay production >=4 and goal production >=4.
- Allows the route even when ETA saving is not positive, because the strategic
  value is front-base production and next-hop pressure.
- Comets require at least 55 remaining ticks before being considered.

Validation:

- 2P focus confirmation: regular 64-56, candidate 64-56.
- 4P focus confirmation: regular 72-168, candidate 77-163.
- Promoted to `REGULAR_CONFIG` as
  `tail_m2_max14_net7_overpay4_p4lowhome_active4_p2trickle_s30_p4midborder_s40_p4chain_src4_p4mobilebase_regular`.
