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
  `tail_m2_max14_net7_regular_candidate`
- Alias: `regular`
- Packaged submission alias:
  `orbit_wars_regular_p4src_w002_lead10_p4earlysrc_s35_80_t120_tail_m2_max14_net7_slim_20260518`
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
- Why this matters: the previous recapture champion, source-risk champion, and
  leader-containment/source-protection champions are kept under their own
  historical names, while `regular` now uses the validated early 4P
  source-protection improvement plus the positive tail-capture candidate.
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

Historical best configs are intentionally kept in `HISTORICAL_BEST_VARIANTS`
and `CHAMPION_OPPONENT_VARIANTS`; do not delete named champion configs when
moving the `regular` alias forward.

The strategy package is designed for local testing and for selectively merging ideas
back into `rulebase/baseline`. If submitting to Kaggle as multiple files, package the
whole directory with `agent.py` as the entrypoint or inline the modules into `main.py`.
