# unread_260519 Ablation Goal

Source: `docs/unread_260519_deep_failure_review.md`

Goal: find variants that beat the current `REGULAR_CONFIG`, not just variants
that explain old losses. Current regular already includes the previous promoted
ideas:

- 4P low-home prod2 opening override.
- 2P high-production trickle.
- 4P mid-border source reserve.
- 4P recent-capture chain source.

So the next experiments should avoid broad repeats of those ideas. The new target
is to copy what the winners did against Solo:

1. Much higher 4P launch throughput before `t=80`.
2. Direct border pressure after initial expansion, especially `t=45..120`.
3. Better 2P high-production planet preservation against incoming waves.
4. More third-party/chaos harvesting in 4P.

## New Suites Added

### Plan032: 4P Throughput Counter

Suite: `unread260519_plan032_p4_throughput_counter`

Why: in the new replay batch, 4P winners averaged `82.6` launches before `t=80`,
while Solo averaged only `14.8`. The current regular may be over-filtering early
source sends or missing fallback attacks.

Variants:

- `p4_no_early_source_filter`
- `p4_short_source_filter_s35_60`
- `p4_no_attack_fallback_c6_s25_150`
- `p4_tail_inject_limit1_score52`
- `p4_throughput_tail_no_filter`

Primary command:

```bash
uv run python -m rulebase.kaggle_public_rl_informed_strategies.fast_multiplayer_eval \
  --suite unread260519_plan032_p4_throughput_counter \
  --games-per-seat 60 \
  --workers 64
```

Interpretation:

- Promote only if win rate and avg rank both improve over `regular`.
- If `no_early_source_filter` wins, current early source filter is too timid.
- If only `tail_inject` wins, the issue is not raw aggression but access to third-party tail targets.
- If `throughput_tail_no_filter` wins while individual pieces do not, the strategy needs a combined "chaos farmer" mode.

### Plan033: 2P Wave Hold Counter

Suite: `unread260519_plan033_p2_wave_hold_counter`

Why: several 2P losses are not bad openings. Solo often reaches competitive
production, then loses prod3-prod5 anchors to repeated direct pressure.

Variants:

- `p2_wave_prod3_h40_post45_m12`
- `p2_wave_prod4_h45_post60_m14`
- `p2_dynamic_recap_prod2_b35`
- `p2_wave_plus_recap_prod2`
- `p2_candidate3_wave`

Primary command:

```bash
uv run python -m rulebase.kaggle_public_rl_informed_strategies.fast_ablation_eval \
  --suite unread260519_plan033_p2_wave_hold_counter \
  --opponent regular \
  --games-per-seat 120 \
  --workers 64
```

Interpretation:

- If wave-only wins, preserve-prod should be promoted before adding recapture breadth.
- If wave+recap wins but wave-only is neutral, the key is "hold what can be held, immediately retake what lands thin."
- If `candidate3_wave` wins but others do not, the failure is target/candidate breadth under pressure.

### Plan034: Vadasz Imitation Combo

Suite: `unread260519_plan034_vadasz_imitation_combo`

Why: Vadasz wins by chaining new high-production sources into next-hop attacks,
not by passive support. The current regular already has a conservative
`src>=4` chain source. This suite tests whether a more opponent-like chain and
tail style can beat current regular.

Variants:

- `p4_chain_src3_relaxed`
- `p4_chain_mobile_relay`
- `p4_chain_tail_inject`
- `p4_vadasz_combo_relaxed_chain_tail`
- `p4_vadasz_combo_pressure`

Primary command:

```bash
uv run python -m rulebase.kaggle_public_rl_informed_strategies.fast_multiplayer_eval \
  --suite unread260519_plan034_vadasz_imitation_combo \
  --games-per-seat 60 \
  --workers 64
```

Interpretation:

- If `chain_src3_relaxed` wins, current `src>=4` chain is too narrow.
- If `chain_tail_inject` wins, current tail capture is good but not entering the candidate list often enough.
- If `vadasz_combo_pressure` wins, the right style is early local/leader pressure plus third-party capture, not just chain mechanics.

## Run Order

1. Run Plan032 4P first. The largest measured gap is 4P launch throughput.
2. Run Plan033 2P second. This protects against regressions where a 4P counter improves chaos play but weakens anchor preservation.
3. Run Plan034 third. It is the most strategically interesting but has more moving parts.
4. For any winner, immediately run the opposite-mode regression:
   - 4P winner -> run `fast_ablation_eval` against `regular`.
   - 2P winner -> run `fast_multiplayer_eval` against default opponents.

## Promotion Gate

A candidate is promotable only if:

- It beats `regular` in its focus suite.
- It is not worse than `regular` in the opposite-mode regression by more than
  roughly 1-2 percentage points.
- For 4P, avg rank improves or stays flat; win rate alone is not enough.
- For 2P, non-loss should not fall even if win rate rises.

Recommended confirmation size after a positive focus run:

```bash
# 4P confirmation
uv run python -m rulebase.kaggle_public_rl_informed_strategies.fast_multiplayer_eval \
  --suite <focus_suite> --games-per-seat 120 --workers 64

# 2P confirmation
uv run python -m rulebase.kaggle_public_rl_informed_strategies.fast_ablation_eval \
  --suite <focus_suite> --opponent regular --games-per-seat 200 --workers 64
```

## Expected Best Bets

My current ordering by probability:

1. `p4_tail_inject_limit1_score52`
2. `p4_no_attack_fallback_c6_s25_150`
3. `p4_vadasz_combo_pressure`
4. `p2_wave_plus_recap_prod2`
5. `p4_chain_src3_relaxed`

The most dangerous variants are the no-filter combos. They might beat current
regular in noisy 4P games but regress badly in 2P or against calmer opponents, so
they need confirmation before promotion.
