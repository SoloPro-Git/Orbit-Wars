# unread_260521/77273815 latest-regular fast ablation

Replay: `data/myreplay/unread_260521/77273815.json`

New baseline requested by submit package:

```bash
.kaggle/orbit_wars_tail_m2_max12_net7_overpay4_p4lowhome_active4_p4seed_prod4_slim_20260521.tar.gz
```

Local `REGULAR_CONFIG` matches this package:

- base: `TAIL_M2_MAX12_NET7_OVERPAY4_P4LOWHOME_ACTIVE4_P4SEED_PROD4_REGULAR_CONFIG`
- tail cap lowered to `third_party_tail_max_ships=12`
- 4P narrow prod4 capture seed enabled:
  - `capture_seed_min_active_players=4`
  - `capture_seed_max_active_players=4`
  - `capture_seed_min_target_production=4`
  - `capture_seed_max_step=60`
  - `capture_seed_max_send=10`

All runs used `training2.fast_orbit_wars.make_fast_orbit_wars` with numba.

## 2P Confirm Versus Latest Regular

Custom run:

- opponent: latest `regular`
- `games_per_seat=20`
- variants: previous 2P pressure/tempo candidates rebuilt from latest
  `REGULAR_CONFIG`
- artifact:
  `rulebase/kaggle_public_rl_informed_strategies/experiments/unread260521_latest_regular_p2_confirm_fast_ablation_20260521_151338.json`

| variant | result | win rate | non-loss | P0 | P1 |
|---|---:|---:|---:|---:|---:|
| `latest_regular` | 21-16-3 | 0.525 | 0.600 | 11-8-1 | 10-8-2 |
| `enemy_hp_prod3_s35_140` | 21-17-2 | 0.525 | 0.575 | 11-9-0 | 10-8-2 |
| `tempo_prod3_s45` | 19-18-3 | 0.475 | 0.550 | 8-11-1 | 11-7-2 |
| `enemy_hp_prod3_s25_110` | 19-19-2 | 0.475 | 0.525 | 9-11-0 | 10-8-2 |
| `hp_s35_plus_tempo_prod3` | 17-21-2 | 0.425 | 0.475 | 6-14-0 | 11-7-2 |
| `tempo_prod4_s55` | 16-21-3 | 0.400 | 0.475 | 10-9-1 | 6-12-2 |
| `tempo_no_attack_prod3_s65` | 14-23-3 | 0.350 | 0.425 | 6-13-1 | 8-10-2 |
| `hp_s35_plus_tempo_prod4` | 14-24-2 | 0.350 | 0.400 | 8-12-0 | 6-12-2 |
| `hp_s25_plus_tempo_prod4` | 11-27-2 | 0.275 | 0.325 | 5-15-0 | 6-12-2 |

Takeaway:

- Against latest regular, none of the previous pure-`StrategyConfig` 2P variants
  beat the baseline.
- `enemy_hp_prod3_s35_140` ties win rate but has lower non-loss.
- `tempo_prod3_s45` improves P1 slightly (`11-7-2` versus regular `10-8-2`)
  but hurts P0 badly (`8-11-1` versus `11-8-1`).
- Combining `hp` and `tempo` is negative.

## Seat-Conditional Probe

Because `tempo_prod3_s45` only looked useful for P1, I ran a probe where the
variant can choose one parameter set for `obs.player == 0` and another for
`obs.player == 1`. This is not directly expressible by current config; it is
only a probe for whether adding seat-aware logic is worth it.

Custom run:

- opponent: latest `regular`
- `games_per_seat=20`
- artifact:
  `rulebase/kaggle_public_rl_informed_strategies/experiments/unread260521_latest_regular_p2_seat_conditional_fast_ablation_20260521_152755.json`

| variant | result | win rate | non-loss | P0 | P1 |
|---|---:|---:|---:|---:|---:|
| `p0_hp_p1_tempo3` | 19-19-2 | 0.475 | 0.525 | 9-11-0 | 10-8-2 |
| `p0_hp_p1_hp_tempo3` | 19-19-2 | 0.475 | 0.525 | 9-11-0 | 10-8-2 |
| `p1_tempo3_only` | 17-20-3 | 0.425 | 0.500 | 7-12-1 | 10-8-2 |
| `p1_hp_tempo3_only` | 17-20-3 | 0.425 | 0.500 | 7-12-1 | 10-8-2 |
| `p1_tempo4_only` | 17-21-2 | 0.425 | 0.475 | 7-12-1 | 10-9-1 |
| `latest_regular` | 16-21-3 | 0.400 | 0.475 | 7-12-1 | 9-9-2 |

Takeaway:

- Seat-conditional variants have a small positive signal in this seed block,
  but not enough to justify implementation/promotion yet.
- The best conditional variants still lose too many P0 games.
- The P1 improvement is small and seed-sensitive.

## Current Recommendation

Do not promote any 2P change from this replay yet.

The latest regular is materially stronger than the earlier baseline used in the
first pass. After rebasing, the old promising changes mostly disappear:

- `enemy_high_prod_pressure` is no longer clearly better;
- `opening_tempo` is seat-sensitive and unstable;
- combined pressure + tempo variants are worse.

Next useful work should be metric-driven rather than adding another broad rule:

1. Add replay-style diagnostics to fast runs: `t=50` production, `t=80` launch
   count, first high-production planet loss.
2. Search for a narrower P1-only opening/tempo condition, but require it to
   improve a second seed block and not reduce P0.
3. Keep latest regular unchanged for now.
