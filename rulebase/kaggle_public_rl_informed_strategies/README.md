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

- Config name: `mp_local3_neu5_comet12_path4p_hold4_terr30_s35_home6_regular`
- Alias: `regular`
- Code commit: `48254b1`
- Base: `mp_local3_neu5_comet12_path4p_hold4_terr30_regular`
- Added regular feature: `enable_capture_hold_margin_gate=True`,
  `capture_hold_margin=4`, plus `enable_opening_neutral_territory_score=True`,
  `opening_territory_penalty=30.0`,
  `opening_territory_enemy_closer_margin=8.0`,
  `opening_territory_step_limit=35`, and a soft late home-anchor source
  reserve (`enable_home_anchor_source_reserve=True`,
  `home_anchor_step_min=60`, `home_anchor_min_after=6`,
  `home_anchor_prod_turns_after=1`)
- Validation: `myreplay_territory_top_validate_4p_ablation_20260515_185314_530542_e6a1823a`
- 4P result against recent champions: `54-106-0`, win rate `33.8%`,
  average rank `1.66`
- Previous `regular` in the same validation: `49-111-0`, win rate `30.6%`,
  average rank `1.69`
- 2P validation against `regular`: same non-loss as previous `regular`
  (`4-6-70`, non-loss `92.5%`), so no obvious 2P regression signal.

Historical best configs are intentionally kept in `HISTORICAL_BEST_VARIANTS`
and `CHAMPION_OPPONENT_VARIANTS`; do not delete named champion configs when
moving the `regular` alias forward.

The strategy package is designed for local testing and for selectively merging ideas
back into `rulebase/baseline`. If submitting to Kaggle as multiple files, package the
whole directory with `agent.py` as the entrypoint or inline the modules into `main.py`.
