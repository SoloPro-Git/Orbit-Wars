# TinyPPO Phase 1 Status

Date: 2026-05-25

Goal: learn regular rulebase actions on the same observed states before PPO
self-play. Do not use winrate against regular as the BC stop condition.

Current best Phase 1 candidate:

```bash
checkpoint=tinyPPO/regular_bc_pure_targethead.pt
target_mask_mode=all_planets
launch_bias=-0.25
ship_bias=0.0
launch_temperature=1.0
```

Held-out regular-state eval, seed `992000`, 2P, 8 games, 12 rows/game:

```json
{
  "source_f1": 0.3764189514189513,
  "source_target_f1": 0.26956018518518515,
  "action_f1": 0.2556712962962963,
  "action_count_mae": 1.375,
  "model_actions_per_state": 1.2395833333333333,
  "regular_actions_per_state": 1.84375,
  "model_errors": 0.0
}
```

Useful comparison on the same seed:

- `targethead + candidate`: action F1 about `0.200`.
- `targethead + all_planets`: action F1 about `0.213`.
- `targethead + all_planets + launch_bias -0.25`: action F1 about `0.256`.
- Low-LR full-model fine-tune improved source recall but over-launched; action
  F1 fell back to about `0.200` without launch calibration.

Conclusion:

- `all_planets` runtime masking is better aligned with all-planets BC because
  the old candidate mask excludes too many regular targets before the target
  head can choose.
- Phase 1 is not complete yet. The model is moving toward regular behavior, but
  action F1 is still too low to call it "regular-like" or to start PPO self-play
  as the main run.
- Next BC work should improve action-count/source calibration and target choice
  without relying on online winrate against regular.

Suggested next eval command:

```bash
.venv/bin/python -m tinyPPO.eval_regular_imitation \
  --checkpoint tinyPPO/regular_bc_pure_targethead.pt \
  --players-list 2 --games-per-players 8 --rows-per-game 12 \
  --episode-steps 180 --seed 992000 --device cuda:7 \
  --diagnose-policy --target-mask-mode all_planets --launch-bias -0.25
```

When Phase 1 passes a stronger imitation gate, start PPO with:

```bash
--opponent-checkpoint <phase1-static-or-previous-ckpt> \
--start-phase latest \
--freeze-latest-opponent \
--promote-opponent-on-eval \
--promote-opponent-threshold 0.80 \
--target-mask-mode all_planets \
--launch-bias -0.25
```
