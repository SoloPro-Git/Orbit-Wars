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

- Config name: `mp_local3_neu5_comet12_path4p_hold4_regular`
- Alias: `regular`
- Code commit: `8ba9ca5`
- Base: `mp_local3_neu5_comet12_path4p_regular`
- Added regular feature: `enable_capture_hold_margin_gate=True`,
  `capture_hold_margin=4`
- Validation: `recent_loss_validate_4p_ablation_20260515_181359_331711_0a162cd6`
- 4P result against recent champions: `23-57-0`, win rate `28.7%`,
  average rank `1.71`
- Previous `regular` in the same validation: `20-60-0`, win rate `25.0%`,
  average rank `1.75`
- 2P validation against `regular`: same non-loss as previous `regular`
  (`2-5-33`, non-loss `87.5%`), so no obvious 2P regression signal.

Historical best configs are intentionally kept in `HISTORICAL_BEST_VARIANTS`
and `CHAMPION_OPPONENT_VARIANTS`; do not delete named champion configs when
moving the `regular` alias forward.

The strategy package is designed for local testing and for selectively merging ideas
back into `rulebase/baseline`. If submitting to Kaggle as multiple files, package the
whole directory with `agent.py` as the entrypoint or inline the modules into `main.py`.
