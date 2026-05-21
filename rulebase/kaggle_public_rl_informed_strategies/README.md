# Kaggle Public RL-Informed Strategies

This directory starts as a copy of `rulebase/kaggle_public_strategies` and adds
deterministic signals inspired by the local RL pipeline under `training/`.

Main sources:

- `orbit-wars-agent-score-1049-2-highest-public.ipynb`
- `orbit-wars-rl-pipeline-public.ipynb`
- `orbit-wars-reinforcement-learning-tutorial.ipynb`

Included strategy ideas:

- Public production/distance/enemy-bonus target scoring.
- Enemy incoming detection by simulating fleet collision with planets.
- Defensive reinforcement planning for owned planets under attack.
- Moving planet trajectory prediction and aiming.
- Attack reservation through tracked fleet trajectories.
- Multi-source cooperative attacks against expensive targets.
- Optional shot validator feature encoder from the RL hybrid notebook.
- Optional shot validator move filter wrapper if `weights.npz` is available.
- Remaining-turn economic value from `training/core/reward.py`.
- Real comet lifetime / ROI gating when `obs["comets"]` is available.
- Contested-target and incoming-balance penalties from `training/core/feature_engineering.py`.
- Late-game arrival filtering.
- Rear-to-front support inspired by `training/expert/kaggle_expert.py`.

Entrypoint:

```python
from rulebase.kaggle_public_rl_informed_strategies import agent
```

Current best submission candidate:

- Config name:
  `tail_m2_max12_net7_overpay4_p4lowhome_active4_p4seed_prod4_regular`
- Alias: `regular`
- Packaged submission alias:
  `orbit_wars_tail_m2_max12_net7_overpay4_p4lowhome_active4_p4seed_prod4_slim_20260521`
- Base: `mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_posthold_recap_b35_p4src_w002_lead10_regular`
- Added regular feature: `enable_capture_hold_margin_gate=True`,
  `capture_hold_margin=4`, plus `enable_opening_neutral_territory_score=True`,
  `opening_territory_penalty=30.0`,
  `opening_territory_enemy_closer_margin=8.0`,
  `opening_territory_step_limit=35`, and a soft late home-anchor source
  reserve (`enable_home_anchor_source_reserve=True`,
  `home_anchor_step_min=60`, `home_anchor_min_after=6`,
  `home_anchor_prod_turns_after=1`), plus
  `capture_hold_use_post_capture_window=True`, and widened recent-loss
  recapture timing with a softer bonus:
  `recent_loss_recapture_min_step=0`,
  `recent_loss_recapture_max_step=500`,
  `recent_loss_recapture_window=50`,
  `recent_loss_recapture_bonus=35.0`, plus a 4P-only soft source-threat target
  penalty (`source_threat_send_min_active_players=4`,
  `source_threat_send_min_step=35`, `source_threat_send_max_step=160`,
  `source_threat_send_min_production=4.0`,
  `source_threat_target_penalty_weight=0.02`), plus a 4P-only moderate
  production-leader target pressure
  (`multiplayer_leader_prod_bonus=1.0`).
- Added 2026-05-18 feature: a 4P-only early source send filter for production
  `4+` sources at steps `35-80`, radius `42`, margin `5`, ROI multiplier
  `1.40`, net value `30.0`, and trade ratio `1.20`.
- Added 2026-05-18 tail-capture feature: conservative third-party tail capture
  with a watchlist, no candidate injection, `third_party_tail_margin=2`,
  `third_party_tail_max_ships=14`, and `third_party_tail_min_net_value=7.0`.
- Added 2026-05-18 tail overpay guard: if a tail-capture send is larger than
  the direct capture requirement, require at least `4` ships left after capture
  (`third_party_tail_overpay_min_post_capture=4`). This blocks thin overpay
  tail-captures like the `seed=9040` p0/p3 failures while preserving the
  positive `seed=9004` overpay cases.
- Added 2026-05-18 replay-derived 4P low-home opening feature: for exactly
  four active players, use the early-neutral multiplayer profile through step
  `50`, allow production `2+` neutral targets, use neutral bonus `6.0`, safe
  bonus `24.0`, and contested penalty `12.0`. This follows the strongest
  replay pattern where winners convert early production on low-home maps rather
  than waiting for perfect high-production targets.
- 2026-05-21 historical regular league: full champion pool, top-11 pool, and
  final top-5 mutual leagues made the tail + 4P low-home config the historical
  best regular. Later p2 trickle, p4 midgame border reserve, chain, and mobile
  relay variants are retained as named historical configs because the finalist
  mutual league favored the narrower version.
- 2026-05-21 Vadasz-inspired follow-up: the small positive confirmation suite
  `vadasz_historical_best_small_positive_confirm_fast_4p_ablation_20260521_143845_488612_74557c61`
  tested narrow refinements on that historical best. The combined stricter
  tail cap plus narrow 4P prod4 seed variant scored `77/320` wins, win rate
  `24.1%`, average rank `1.759`; the historical best baseline scored `70/320`,
  win rate `21.9%`, average rank `1.781`. `regular` now points at this
  follow-up config, while the historical-best package/config name remains
  available for rollback and comparison. 2P regression
  `vadasz_historical_best_promote_regression_fast_ablation_20260521_144857`
  was neutral against the historical best: both scored `98-94-8 / 200`.
- Why this matters: the previous recapture champion, source-risk champion, and
  leader-containment/source-protection champions are kept under their own
  historical names, while `regular` now uses the validated early 4P
  source-protection improvement, the positive tail-capture candidate, the
  replay-validated 4P low-home opening profile, the stricter max12 tail cap,
  and the narrow 4P prod4 seed follow-up. The positive-looking 2P low-home
  trickle and 4P midgame border reserve remain named experiments instead of
  live regular behavior.
- Validation:
  `myreplay_plan011_leader_bonus_focus_4p_ablation_20260517_144638_809435_ac08fd31`
- 4P result against recent champions: `356-844-0`, win rate `29.7%`,
  average rank `1.703`
- Previous `regular` in the same validation: `342-858-0`, win rate `28.5%`,
  average rank `1.715`
- Same-seed flips versus previous `regular`: `82` gains, `68` losses,
  net `+14` wins across `1200` games.
- 2P regression:
  `myreplay_plan011_leader_bonus_focus_ablation_20260517_152445`,
  exact neutral (`70-52-478` for all variants, net `0 / 600`).
- 2026-05-18 4P source-protection validation:
  `myreplay_plan014_early_4p_source_guard_focus_4p_ablation_20260518_120425_212961_5aa853d7`.
  New regular candidate: `383-817-0`, win rate `31.9%`, average rank `1.687`;
  previous `regular`: `358-842-0`, win rate `29.8%`, average rank `1.708`.
  Same-seed rank movement: `165` gains, `139` losses, net `+26 / 1200`.
- 2026-05-18 2P regression:
  `myreplay_plan014_early_4p_source_guard_focus_ablation_20260518_122659`,
  exact neutral (`70-52-478` for both variants, net `0 / 600`).

- Tail-capture search on 2026-05-18:
  `third_party_tail_tiny_top_confirm_4p_ablation_20260518_113722_728395_ac08fd31`
  against `regular recapture_s45_e180_w50_b40_regular regular`: `75-165-0`
  over `240` games, win rate `31.25%`; paired versus regular: `12` gains,
  `2` losses, net `+10` wins.
- Tail overpay guard check on 2026-05-18, current source-protection base,
  seeds `9000-9099` against pre-tail source-protection/recapture/pre-tail
  source-protection: plain tail `128-272-0`, paired net `+11` versus pre-tail;
  overpay guard `130-270-0`, paired net `+13` versus pre-tail and `+2` versus
  plain tail.
- Anti-tail hold gate check on 2026-05-18, current tail-overpay regular,
  seeds `9000-9039` against `regular/mp_soft_leader0/comet12_path4p`:
  broad nearby-planet anti-tail gate was harmful (`28-132-0`, paired
  `+5/-21` versus regular). Fleet-only anti-tail gate was near-neutral but
  still negative (`43-117-0`, paired `+0/-1`). Keep
  `enable_third_party_anti_tail_hold_gate` disabled unless a narrower replay
  shape is found.
- Replay-derived 4P low-home opening check on 2026-05-18 using
  `training2.fast_orbit_wars` with `use_numba=True`: against
  `regular/mp_soft_leader0/comet12_path4p`, the active-4 low-home candidate
  scored `196-444-0` over `640` games versus previous tail-overpay regular
  `187-453-0`; paired movement `+51/-42`, net `+9`, rank net `+9`. 2P
  regression was exact neutral over `1200` games (`264-317-19` for both,
  paired net `0`). Historical high-score pool
  `mp_soft_leader0/comet12_path4p/b35_recap` was exact neutral over `320`
  games (`81-239-0` for both, paired net `0`).
- Replay-derived 2P low-home trickle check on 2026-05-19 using
  `training2.fast_orbit_wars` with `use_numba=True`: confirm suite
  `myreplay_plan021_p2_low_home_trickle_confirm_fast_ablation_20260519_095619`
  scored `15-9` for `p2_low_home_trickle_s30_src1_tgt4_m5` versus `7-17`
  for previous regular over `24` games. Follow-up combo suite
  `myreplay_plan022_p2_trickle_highprod_combo_fast_ablation_20260519_100326`
  scored `24-16` for standalone `p2_trickle_s30` versus `13-26-1` for previous
  regular over `40` games; adding high-production source send-filter or recent
  reserve reduced performance to `21-19` and `20-20`, so only the opening
  trickle was promoted.

Historical best configs are intentionally kept in `HISTORICAL_BEST_VARIANTS`
and `CHAMPION_OPPONENT_VARIANTS`; do not delete named champion configs when
moving the `regular` alias forward.

The strategy package is designed for local testing and for selectively merging ideas
back into `rulebase/baseline`. If submitting to Kaggle as multiple files, package the
whole directory with `agent.py` as the entrypoint or inline the modules into `main.py`.
