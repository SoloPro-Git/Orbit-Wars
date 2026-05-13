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

Current smoke test notes:

- `RLInformedPublicRuleAgent` vs `baseline all-strategies`: 5-0 on seeds 42-46.
- `RLInformedPublicRuleAgent` vs original `PublicRuleAgent`: 1-4 on seeds 42-46.

So this package should be treated as an experiment bed for RL-derived rule ideas,
not as the current best submission candidate. The strongest submission candidate
remains `rulebase/kaggle_public_strategies` until ablation proves otherwise.

The strategy package is designed for local testing and for selectively merging ideas
back into `rulebase/baseline`. If submitting to Kaggle as multiple files, package the
whole directory with `agent.py` as the entrypoint or inline the modules into `main.py`.
