# TinyPPO Phase 1 Status

Date: 2026-05-25

Goal: learn regular rulebase actions on the same observed states before PPO
self-play. Phase 1 is not trying to beat or perturb regular; it is only trying
to make the model choose like regular when both face the same situation. Do not
use winrate against regular as the BC stop condition.

Scope:

- Phase 1 is pure regular behavior cloning. It should train from states produced
  by regular games and labels produced by regular on those same states.
- Phase 1 should answer only this question: when the model and regular see the
  same observation, does the model make a similar source/target/ship decision?
  It should not try to improve on regular, break regular's game plan, delete
  regular actions, or optimize for wins.
- Do not mix in DAgger/on-policy states, bridge policies, source deletion,
  reward learning, PPO, opponent promotion, regular-vs-model competition, or
  winrate gates for the Phase 1 checkpoint.
- The Phase 1 checkpoint is only an initialization for Phase 2. Beating regular
  belongs to Phase 2 PPO/self-play, after the model can imitate regular's
  source, target, and ship choices on held-out regular states. Phase 2 should
  then train against frozen self checkpoints and promote the frozen opponent
  only when the new policy reliably beats the old checkpoint.

Phase 1 acceptance should be based on held-out same-state imitation metrics:
action density close to regular, source F1, source-target/action F1, target
top-1/top-k, and ship bucket accuracy. Winrate versus regular is only a sanity
check because a BC policy and regular may drift into different states online.

Reusable gate command for Phase 1 candidates:

```bash
.venv/bin/python -m tinyPPO.phase1_gate \
  --checkpoint tinyPPO/<candidate>.pt \
  --players-list 2 --seeds 992000,993000,994000 \
  --games-per-players 8 --rows-per-game 12 \
  --episode-steps 180 --device cuda:7 \
  --target-mask-mode all_planets --target-pair-weight 1.0 \
  --launch-bias-grid=-0.35,-0.25,-0.15,-0.05,0.0,0.05,0.15 \
  --out tinyPPO/runs/<run>/phase1_gate.json
```

Treat `pass_phase1=true` as evidence that the BC checkpoint is regular-like
enough to consider Phase 2. Treat `pass_phase1=false` as "continue pure BC or
architecture/loss work"; do not jump to PPO just because online games look
interesting.

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
- Pure all-heads 2P, no source-target summary, `launch_pos_weight=2.4` reached
  validation source/action-count metrics that looked reasonable in-loader, but
  held-out regular-state action F1 did not beat the current best pure candidate:
  best sweep point was about `action_f1=0.229`, `micro_action_f1=0.168`.
- The older `targetaware_allplanets*` checkpoints should not be treated as
  Phase 1 pure-BC winners because their collect metadata includes DAgger rows.
- Current active experiment: pure 2P regular BC with `source_target_summary`,
  `launch_pos_weight=2.0`, no DAgger, run directory
  `tinyPPO/runs/regular_bc_pure2p_summary_pos20_20260525`. Early epoch 97
  loader metrics show promising launch-density calibration (`val_pred_rate`
  about `0.056` versus regular `0.058`), but target/action held-out eval has
  not yet been run.
- Follow-up: `regular_bc_pure2p_summary_pos20.pt` looked healthy in-loader
  through about epoch 400 (`val_pred_rate` close to regular and
  `val_target_acc` around `0.38`), but held-out regular-state action matching
  regressed: best 96-state sweep was only about `action_f1=0.198` to `0.202`
  depending on runtime mask, worse than the current pure target-head baseline.
  This run was stopped.
- Runtime `candidate` masks with `include_friendly_targets` and `top_k=12` did
  not rescue held-out action matching; they reduced micro source-target/action
  F1. The next useful delta is target-ranking quality itself, not more epochs
  or a narrower runtime mask.
- A direct coverage check showed `all_planets` covers the held-out regular
  labels on seed `992000` (`87/87` labelled actions). The failure is not target
  mask coverage; it is that target logits do not rank the regular target high
  enough globally.
- The target-margin continuation (`regular_bc_pure_target_margin.pt`) was
  stopped. Loader metrics were worse than the current best baseline, and a
  20-game 2P fast-env check versus regular at seed `993000` was only `1W-19L`
  for both `launch_bias=-0.35` and `-0.25`. Its conservative
  `launch_bias=-0.35` held-out macro action F1 looked slightly higher
  (`0.279`), but it averaged only `0.96` model actions per state versus
  regular's `1.84`, so it is under-launching rather than imitating regular's
  action density.
- Diagnostic eval for the current best baseline at `launch_bias=-0.25` on the
  same held-out states: labelled rows `96`, regular actions/state `1.84`,
  model actions/state `1.24`, action F1 `0.256`, source-target F1 `0.270`,
  label launch hit about `0.399`, target top-1 hit under all-planets runtime
  logits about `0.247`. The next single delta should improve global context and
  target/source decision quality, not move to PPO.

Conclusion:

- `all_planets` runtime masking is better aligned with all-planets BC because
  the old candidate mask excludes too many regular targets before the target
  head can choose.
- Phase 1 is not complete yet. The model is moving toward regular behavior, but
  action F1 is still too low to call it "regular-like" or to start PPO self-play
  as the main run.
- Next BC work should improve same-state imitation quality: action-count/source
  calibration, target ranking, and ship amount matching on held-out regular
  states. Do not switch Phase 1 into bridge/source-deletion/constraint gameplay;
  those belong to later analysis or Phase 2 if they are still needed.
- New architecture delta under test: enable planet self-attention only when
  `layers>1`; old `layers=1` checkpoints keep the previous behavior. This is
  intended to give BC access to regular's global board context without changing
  the action-space constraints. The first attention run with
  `launch_pos_weight=2.0` collapsed to no-launch in early epochs and was
  stopped. The active run is `regular_bc_attn2_pos8_2p_20260526`, using
  `hidden=128`, `layers=2`, `launch_pos_weight=8.0`, and the same pure 2P
  regular cache. This run was stopped after held-out eval showed it over-launches:
  even with `launch_bias=-0.8`, it averaged `3.19` model actions/state versus
  regular's `1.84`, and held-out action F1 was only about `0.202`, below the
  current best baseline's `0.256`. The next active attempt keeps the attention
  architecture but lowers `launch_pos_weight` to `4.0` to find a calibrated
  launch-density point before judging target quality.
- The first `launch_pos_weight=4.0` run found a useful early density window
  (`val_pred_rate` near regular around epoch 80-90), but later over-launched
  and the original best-checkpoint score selected an over-launching checkpoint.
  A 20-game fast-env sanity check versus regular at seed `993000` was `2W-18L`,
  so it is still not Phase-1 complete. The Ray BC imitation score now includes
  launch-density and action-count penalties so future `best_out` checkpoints do
  not prefer over-launching models.
- The short source-head-only continuation on 2026-05-25 under-launched
  (`pred_rate` about `0.041` versus regular `0.058`) and was stopped. It should
  not replace the current best Phase 1 candidate.
- 2026-05-26 correction: Phase 1 should not use bridge/source deletion,
  destructive perturbations, winrate pressure, or any other "damage regular"
  objective. It is only same-state regular behavior cloning: given the same
  board as regular, make the model pick similar source, target, and ship
  decisions. Beating regular belongs to Phase 2 PPO after the model can imitate
  regular.
- `regular_bc_attn2_pos4_score_2p.pt` selected a balanced loader checkpoint at
  epoch `90` (`val_pred_rate=0.0568` vs regular `0.0582`,
  `val_target_acc=0.388`), but held-out same-state eval still did not beat the
  old baseline: best action F1 was about `0.265` only when heavily
  under-launching (`0.41` actions/state); the density-matched points were below
  the baseline. The run later over-launched and was stopped.
- `regular_bc_attn2_pos4_slotset_2p.pt` added permutation-invariant same-source
  slot loss. It reached a clean loader density window at epoch `80`
  (`val_pred_rate=0.0580`, regular `0.0582`, score `0.278`), but held-out
  same-state eval with all-planets runtime mask was worse than the current
  baseline (`action_f1` about `0.25` only while under-launching, and about
  `0.18` near matched action density). This is not a Phase 1 winner. The
  useful lesson is that slot ordering was not the main blocker.
- `regular_bc_attn2_pos4_slotset_pair_2p.pt` kept the slot-set loss and added
  an auxiliary source-target binary loss so every regular source-target edge was
  supervised directly, not only through ordered active slots. It did not beat the
  current baseline on held-out same-state imitation. Near matched action density
  (`model_actions_per_state` about `1.80` versus regular `1.84`), action F1 was
  only about `0.19`; the train/candidate-style target hit was much higher than
  the all-planets runtime target hit, so the remaining blocker is global target
  ranking.
- `regular_bc_attn2_pairhead_2p.pt` added an independent source-target pair
  head. It improved source selection somewhat, but still did not pass Phase 1:
  near matched action density (`1.85` versus regular `1.84`) held-out action F1
  was about `0.21`, below `regular_bc_pure_targethead.pt` at about `0.256`.
  The run was stopped after it kept over-launching in later epochs
  (`val_pred_rate` around `0.12-0.15` versus regular `0.058`).
- 2026-05-26 Phase 1 stop/selection rule: do not optimize for damaging,
  perturbing, or beating regular. Select checkpoints by held-out same-state
  imitation only: action density close to regular, source F1, source-target F1,
  action F1, ship bucket accuracy, and all-planets target-pair top1 quality.
  Online winrate against regular may be logged as a sanity check, but must not
  be a Phase 1 stop condition.
- BC training now logs `target_pair_acc` / `val_target_pair_acc`, and Ray
  checkpoint selection records `val_target_quality=max(val_target_acc,
  val_target_pair_acc)`. This is still pure imitation; it only makes the
  validation score aware of the independent source-target head.
- `regular_bc_attn2_pairhead_quality_2p.pt` used the density-aware imitation
  score and did find a clean in-loader density window around epoch `60-70`
  (`val_launch_pred_rate` about `0.056-0.066` versus regular `0.0577`), but
  held-out same-state eval still did not beat the old baseline: near matched
  density (`model_actions_per_state=1.90`, regular `1.84`), action F1 was only
  about `0.195`. A direct held-out coverage check confirmed `all_planets`
  covers regular labels (`231/231` actions), so the issue is target ranking
  generalization, not runtime mask coverage.
- Current active pure-BC run: `tmux0:3 bc_regular_10k`, directory
  `tinyPPO/runs/regular_bc_attn2_pairhead_quality_2p_10k_20260526`, SwanLab
  experiment `tinyppo-regular-bc-pairhead-quality-2p-10k-20260526`. This keeps
  the same architecture/loss and increases pure 2P regular data from `2000` to
  `10000` games to test whether target/source-target imitation is data-limited.
  It is still Phase 1 only: no DAgger, no bridge, no PPO, no winrate stop.
- `eval_regular_imitation` diagnostic aggregation was corrected after this:
  label/action diagnostics such as `diag_label_target_in_runtime_mask` now use
  action-weighted averages instead of row-weighted averages, so noop rows no
  longer make all-planets coverage look artificially low.
- The 10k run finished data collection on 2026-05-26 and produced
  `tinyPPO/data/regular_bc_2p_10000g_rows16_20260526.pkl` (`42G`). Training
  found a loader density window around epoch `37-41`, then later epochs
  over-launched while target accuracy continued to rise.
- A quick Phase 1 gate preview on seed `992000`, 4 games, 8 rows/game showed the
  10k checkpoint is closer but still not Phase-1 accepted. With
  `target_pair_weight=1.0`, best density-matched metrics were
  `action_f1=0.271`, `source_target_f1=0.281`, density ratio `0.974`; below the
  configured `0.30` action/source-target thresholds.
- A decode sweep showed `target_pair_weight=0.0` is better for this checkpoint:
  density ratio `1.000`, `source_f1=0.477`, `source_target_f1=0.302`,
  `action_f1=0.293`. This is close but still below the `action_f1 >= 0.30`
  Phase 1 gate, so do not start PPO from it yet. If evaluating this checkpoint
  again, pass `--target-pair-weight 0.0`.
- The Phase 1 action-F1 metric was corrected to use the same relative
  `required_ships` bucket as BC training, instead of an absolute
  `log2(ships)` bucket. This only fixes the imitation ruler; it does not change
  the training target. With the corrected gate on three held-out seeds
  (`992000,993000,994000`), `checkpoint_e0040.pt` from
  `regular_bc_attn2_pairhead_snap_2p_10k_20260526` still does not pass Phase 1:
  density ratio `1.056`, `source_f1=0.425`,
  `source_target_f1=0.270`, `action_f1=0.248`. The source head is better than
  the old baseline, but target/ship action imitation is not regular-like
  enough yet.
- `regular_bc_attn2_pairsoftmax_2p_10k_20260526` added a pure-BC
  source-target pair softmax loss. This moved loader target-pair accuracy up
  substantially (`val_target_pair_acc` around `0.49` by epoch `180`) and gave
  promising small previews, but it still did not pass the full same-state gate.
  Best full-gate checks:
  - `checkpoint_e0080.pt`, three seeds, 8 games x 12 rows:
    density ratio `0.971`, `source_f1=0.416`,
    `source_target_f1=0.272`, `action_f1=0.258`.
  - `checkpoint_e0120.pt`, three seeds, 8 games x 12 rows:
    density ratio `1.002`, `source_f1=0.410`,
    `source_target_f1=0.285`, `action_f1=0.269`.
  This is progress over the old baseline but not enough to start PPO.
- Current active Phase 1 run:
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_20260526`, SwanLab
  experiment
  `tinyppo-regular-bc-rulefeat-pairsoftmax-2p-10k-20260526-task1c`.
  This run expands pair features from `16` to `24` dimensions with deterministic
  regular-style target-ranking signals: required-ships ratio, public target
  score, remaining economic value, late-arrival usefulness, comet life, behind
  production flag, and a strategic score. Because feature tensors changed, it
  recollects a fresh pure regular 2P cache at
  `tinyPPO/data/regular_bc_rulefeat_2p_10000g_rows16_20260526.pkl` before
  training. This is still Phase 1 only: no DAgger, no bridge, no PPO, no
  winrate stop. The first attempt used `--collect-games-per-task 16`, which
  made progress and partial-cache visibility too coarse. The second attempt
  used `--collect-games-per-task 1` and `--partial-cache-interval 500`, but
  accumulated every partial cache and filled `/data2` before training began.
  The current `task1c` run keeps only the newest partial cache and deletes all
  partials before the final full-cache save. This preserves visibility without
  letting partial checkpoints become the bottleneck. Its stop condition remains
  regular-like same-state imitation metrics, not winrate versus regular.
- `regular_bc_rulefeat_pairsoftmax_2p_10k_20260526/task1c` finished cleanly on
  2026-05-26. Collection produced
  `tinyPPO/data/regular_bc_rulefeat_2p_10000g_rows16_20260526.pkl`
  (`10000` games, `159954` samples, `352925` labelled actions), and training
  reached epoch `180` with exit code `0`. The final checkpoint improved the
  full same-state gate enough to pass the configured imitation thresholds:
  `regular_bc_ray_e0180.pt`, three held-out seeds, 8 games x 12 rows,
  `density_ratio=1.253`, `source_f1=0.449`, `source_target_f1=0.324`,
  `action_f1=0.301`. However, the online sanity eval versus regular remained
  decisively bad: epoch `40` was `1W-15L`, epoch `80` was `0W-16L`, epoch `120`
  was `1W-15L`, and epoch `160` was `0W-16L`; best nonloss was only `0.0625`.
  Therefore this is not an accepted Phase 1 checkpoint and Phase 2 self-play
  must not start from it. The useful diagnosis is that same-state source/target
  imitation is now barely crossing the local gate, but rollout behavior still
  does not preserve regular's stabilizing dynamics.
- During this run, `phase1_gate` was fixed to default to no baseline checkpoint.
  The old default `tinyPPO/regular_bc_pure_targethead.pt` has incompatible
  feature dimensions for rule-feature checkpoints (`208` vs `216`) and caused
  the watcher to fail/retry. `watch_phase1_gate` now supports an explicit
  `--baseline-checkpoint` and writes a failure JSON when a gate subprocess
  exits nonzero, so one bad checkpoint cannot loop forever.
- A follow-up offline-only continuation was launched on 2026-05-26 from
  `regular_bc_ray_e0180.pt` to test the hypothesis "same data, same training
  hyperparameters, just more BC epochs" before changing thresholds or decoding.
  Run directory:
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e360_20260526`.
  It reuses the same 10k regular cache (`cache_loaded=1`) and avoids GPU0 via
  explicit trainer/eval GPU IDs. This is still Phase 1 only: no DAgger, no
  bridge, no PPO, and no self-play. Early evidence through e0240 is negative
  for the online objective despite improving offline target metrics:
  stochastic eval at e0200 was `0W-16L`, stochastic eval at e0240 was
  `1W-15L`, stochastic eval at e0280 was `2W-14L`, deterministic manual eval at
  e0200 was `5W-11L`, and deterministic manual eval at e0220 was `4W-12L` using
  `target_mask_mode=all_planets`,
  `target_pair_weight=1.0`, `launch_bias=-0.25`, 16 games, 180 steps, seed
  `260526`. This continuation does not yet produce a regular-like checkpoint
  and must not be used to start Phase 2 unless later checkpoints show a clear
  online recovery.
- `imitation_regular_ray.py` now supports `--trainer-gpu-ids` and
  `--eval-gpu-ids-manual` for explicit physical CUDA placement when Ray's GPU
  scheduler would otherwise collide with another job. This is infrastructure
  only; it does not change the BC objective, model, labels, or decoding.

Phase split, current decision:

- Phase 1 stop condition is same-state imitation only. Do not require the model
  to beat regular, and do not train it to sabotage or prune regular's choices.
- Phase 2 starts only after Phase 1 has a verified regular-like checkpoint.
  Then PPO should train against frozen self checkpoints and promote the frozen
  opponent when the live policy reliably beats the old checkpoint.

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
source .venv/bin/activate
PYTHONUNBUFFERED=1 python -m tinyPPO.train_ray \
  --ray-address 10.0.104.198:6380 \
  --out-dir tinyPPO/runs/selfplay_from_regular_bc_<date> \
  --resume-checkpoint <phase1_regular_like_ckpt.pt> \
  --opponent-checkpoint <phase1_regular_like_ckpt.pt> \
  --start-phase latest \
  --opponent-mode self \
  --freeze-latest-opponent \
  --promote-opponent-on-eval \
  --promote-opponent-threshold 0.80 \
  --promote-opponent-metric eval_stochastic_vs_opponent \
  --stop-metric eval_stochastic_vs_regular \
  --stop-winrate 0.70 \
  --updates 100000 \
  --episodes-per-update 42 \
  --workers-per-gpu 2 \
  --gpu-ids 2,3,4,5,6,7 \
  --eval-workers 16 \
  --eval-gpu-ids 0,1 \
  --gpus-per-eval-worker 0.125 \
  --eval-opponents nearest,regular,opponent \
  --eval-interval 40 \
  --eval-games 40 \
  --checkpoint-interval 10 \
  --hidden 128 --heads 4 --layers 2 \
  --source-target-summary --target-pair-head \
  --action-slots 3 --ship-buckets 5 \
  --target-mask-mode all_planets \
  --max-actions-per-source-safety 3 \
  --launch-bias -0.25 \
  --eval-stochastic \
  --latest-opponent-stochastic \
  --opponent-eval-stochastic \
  --top-k-checkpoints 5 \
  --allow-no-swanlab
```

The Phase 2 command deliberately uses the same Phase 1 checkpoint for both the
learner and the initial frozen opponent. The frozen opponent should only be
replaced by promotion after eval shows the live policy beats that old checkpoint
by the configured threshold. Regular remains an eval opponent and stop metric,
not a training opponent.

2026-05-26 long offline continuation:

- User hypothesis: online eval may be seed-sensitive, so continue pure offline
  regular BC much longer before tuning hyperparameters.
- First attempt:
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e10000_20260526`.
  It resumes from
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e360_20260526/latest.pt`,
  which had reached `resume_update=330` before that previous continuation
  stopped on a Ray/system actor error.
- This is still Phase 1 pure BC on the existing regular cache
  `tinyPPO/data/regular_bc_rulefeat_2p_10000g_rows16_20260526.pkl`; no DAgger,
  no bridge policy, no PPO, no self-play, and no online-winrate stop.
- The run uses `--epochs 10000`, `--checkpoint-interval 100`, and
  `--eval-interval 100`, so checkpoints should be saved every 100 epochs/iters
  for later review.
- Resource layout was changed to match observed capacity: `--trainers 14` with
  manual trainer GPU IDs `1,1,2,2,3,3,4,4,5,5,6,6,7,7`, i.e. two training
  actors per GPU on GPUs 1-7. Eval actors are lightweight and manually placed on
  GPUs 1-7. GPU0 is intentionally left free for monitoring/manual eval.
- Startup verification: the log reports `cache_loaded=1`, `resume_update=330`,
  and the manual GPU ID map above. `nvidia-smi` showed two
  `BCTrainEvalActor.train_epoch` processes on each GPU 1-7, using about 5.8GB
  each, plus one small `BCEvalActor` per GPU.
- SwanLab run:
  `https://swanlab.cn/@Solo/orbit-wars/runs/jv3eh3n28drylll8ywyuf`.
- This 14-trainer attempt was stopped by a host system-memory OOM at about
  2026-05-26 10:59:59. The OOM killer selected a `BCTrainEvalActor`; no e400
  checkpoint had been written, so the usable resume point stayed at e330.
- Restarted safer continuation:
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e10000_safe7_20260526`.
  This resumes from the same e330 `latest.pt`, keeps the same data, model,
  losses, LR, batch size, `--epochs 10000`, `--checkpoint-interval 100`, and
  `--eval-interval 100`, but changes resource layout to `--trainers 7` with
  manual trainer GPU IDs `1,2,3,4,5,6,7`.
- Safe7 startup verification: log reports `cache_loaded=1`,
  `resume_update=330`, and manual GPU IDs `[1,2,3,4,5,6,7]`. `nvidia-smi`
  showed one `BCTrainEvalActor` and one small `BCEvalActor` on each GPU 1-7.
  Epoch 331 completed, so training has resumed.
- Safe7 SwanLab run:
  `https://swanlab.cn/@Solo/orbit-wars/runs/bymvi3jb745c2zoreevmq`.
- Per user correction, safe7 was too conservative for throughput and was
  stopped around epoch 342 before any new 100-epoch checkpoint was written.
  Active replacement:
  `tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e10000_2w_eval64_20260526`.
  It again uses 14 trainers with manual GPU IDs
  `1,1,2,2,3,3,4,4,5,5,6,6,7,7` so each GPU 1-7 carries two training workers.
  Eval remains every 100 epochs/iters but was increased to `--eval-games 64`.
  Startup verification showed `cache_loaded=1`, `resume_update=330`, and two
  `BCTrainEvalActor.train_epoch` processes plus one small `BCEvalActor` on each
  GPU 1-7. SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/nba0yp1j0z6o9bqwdql04`.

Optional checkpoint watcher once Phase 1 training starts producing checkpoints:

2026-05-26 DAgger/offline expansion phase:

- The long pure regular-vs-regular continuation was stopped around epoch 2218
  after clear overfitting: train target accuracy kept rising while validation
  target accuracy and validation loss worsened. The saved checkpoints through
  e2100 remain available, but latest should not be treated as the default Phase
  1 candidate.
- Next-stage objective is to reduce BC distribution drift, not to optimize
  online winrate directly and not to start PPO/self-play. The model should see
  states caused by its own rollout, but the labels must still come from
  `regular` on those same observations.
- `tinyPPO/imitation_regular_ray.py` now supports multi-checkpoint DAgger via
  `--dagger-checkpoints` and can collect only model-controlled seats via
  `--dagger-model-seat-only`. This lets the expansion focus on model-induced
  off-distribution states while regular opponents still act in the environment.
- Active mixed-BC run:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_20260526`.
  It resumes from the less-overfit e900 checkpoint and uses e400/e600/e900/e1100/e1300
  as rollout policies for DAgger data collection.
- DAgger collection target: 3000 two-player games, 5 collector actors, about
  600 games per rollout checkpoint, `rows_per_game=16`, stochastic model policy,
  `launch_bias=-0.25`, and `--dagger-model-seat-only`. Expected scale is roughly
  20-30% extra rows relative to the 159,954-row regular cache.
- Training mix: original cache
  `tinyPPO/data/regular_bc_rulefeat_2p_10000g_rows16_20260526.pkl` plus DAgger
  cache
  `tinyPPO/data/regular_bc_dagger_modelseat_5ckpt_2p_3000g_rows16_20260526.pkl`.
  The training objective/model/losses/LR remain the same as the pure BC run.
- This run still belongs to Phase 1. Success requires same-state imitation not
  to regress and online vs regular to stop being consistently near `1W-63L`;
  provisional online target is multi-seed nonloss above about `0.15` before
  considering Phase 2.
- SwanLab run:
  `https://swanlab.cn/@Solo/orbit-wars/runs/4e1fcscsxazx6yyakfuvt`.
- The first DAgger collection attempts were stopped before writing cache because
  actor placement was inefficient: 5 large shards had poor progress visibility,
  then 25 fractional-GPU actors were packed mostly onto GPU0-2 by Ray. DAgger
  collection now supports `--dagger-gpu-ids-manual` so collector actors can use
  explicit physical GPU IDs, matching the manual trainer/eval placement pattern.
- Active replacement collection/training run:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_allgpu_20260526`.
  It keeps the same 3000-game, five-checkpoint, model-seat-only DAgger target,
  but uses 32 collector actors with manual GPU IDs `0,1,2,3,4,5,6,7`, giving
  four DAgger collectors per GPU during collection. SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/36d0zgeqdlgip244d68by`.
- The 32-collector run also made progress but still had large shards and weak
  visibility. DAgger collection now prints `dagger_actor_progress` every 10
  games from each actor. Active replacement:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_64w_20260526`.
  It keeps the same data target but uses 64 collector actors with manual GPU IDs
  `0,1,2,3,4,5,6,7`, i.e. about eight collectors per GPU and about 46-47 games
  per actor. Startup verification showed all 64 actors running, GPUs 0-7 around
  4GB each, and heartbeat rows such as `done=10,total=47,rows=160`.
  SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/etjf3v0s3a23lt5qhw3na`.
- The 64-collector DAgger stage completed and wrote
  `tinyPPO/data/regular_bc_dagger_modelseat_5ckpt_2p_3000g_rows16_20260526.pkl`
  (about 19GB): 3000 games, 47,940 DAgger rows, 288,099 labelled actions,
  `model_seat_only=1`. The mixed training set is now 207,894 rows
  (159,954 original rows plus 47,940 DAgger rows).
- Mixed BC training resumed from e900 and saved
  `regular_bc_ray_e1000.pt`. The e1000 64-game online eval was still bad:
  1 win, 63 losses, 0 draws, winrate/nonloss 0.015625, mean reward -0.96875.
  Continue observing e1100/e1200 before changing another delta; do not treat the
  improving offline imitation metrics as sufficient Phase 1 success.
- The run was stopped manually after e1254 because e1200 online regressed and
  validation target metrics kept degrading while train target metrics improved.
  Online evals were: e1000 = 1W/63L, e1100 = 6W/58L, e1200 = 0W/64L. The best
  online checkpoint for this attempt is e1100:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_64w_20260526/online_best.pt`
  and `regular_bc_ray_best_e1100.pt`. This DAgger batch helped slightly at
  e1100 but did not reach the provisional nonloss target of 0.15 and still does
  not satisfy Phase 1.
- Follow-up diagnosis showed the e1000/e1100/e1200 checkpoints were
  over-launching on both original regular states and DAgger model-seat states.
  A launch-bias sweep improved offline action-count density but did not improve
  online enough: 64-game/bias evals were `-0.25=1W`, `-0.65=1W`,
  `-0.85=1W`, `-1.05=2W`. So inference bias alone is not the next useful
  delta.
- `tinyPPO/diagnose_bc_cache.py` was added for offline cache diagnostics. It
  reports predicted-vs-label action density, launch precision/recall/F1, target
  accuracy/top-k, and ship accuracy for BC checkpoints on encoded row caches.
- The active next run is:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_countcal_eval64x4_20260526`.
  It uses the same original+DAgger data and resumes from e900, but adds one
  focused training objective: `--launch-count-loss-weight 0.1`, a SmoothL1 loss
  matching the predicted launch-probability count per row to the regular label
  action count. No other BC objective hyperparameters were changed.
- The active run saves checkpoints every 100 epochs and evals every 100 epochs.
  Eval uses 64 actors, manual GPU IDs `0,1,2,3,4,5,6,7`, 256 total games per
  eval, i.e. 8 eval workers/GPU and 4 games/worker. Training uses 16 trainers,
  2 trainers/GPU. It is running detached under PID recorded in
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_countcal_eval64x4_20260526/train.pid`.
- Startup gotchas for this run: Ray runtime packaging initially tried to upload
  the 61GB and 19GB cache files, so `imitation_regular_ray.py` now excludes
  large data/run artifacts from `ray.init(runtime_env=...)`. Long detached runs
  should be started with `setsid -f` and `.venv/bin/python`, not plain `nohup uv
  run`, because the tool session can reap background children and Ray workers
  may otherwise create separate uv environments.
- Early count-calibrated metrics around e901-e919 show validation launch
  density much closer to labels than before: `val_launch_true_rate ~= 0.0799`,
  `val_launch_pred_rate` mostly `0.086-0.097`, and
  `val_launch_density_alignment ~= 0.94-0.97`. Let it reach e1000 and inspect
  the 256-game online eval before changing another delta.
- e1000 finished its first 256-game async online eval: `16W/240L/0D`,
  nonloss/winrate `0.0625`, mean reward `-0.875`. This is better than the
  repeated `1W/64L` collapse but still below the provisional `>0.15` nonloss
  target and not clearly better than the previous smaller e1100 result
  (`6W/64`). Do not call Phase 1 complete. Continue this same single-delta run
  to e1100 before changing the recipe.
- e1000 offline metrics remained calibrated enough to keep running:
  `val_launch_pred_rate=0.0971` vs `val_launch_true_rate=0.0799`,
  `val_launch_density_alignment=0.9348`, `val_target_acc=0.5383`,
  `val_ship_acc=0.9118`. e1001 was even more density-aligned
  (`val_launch_pred_rate=0.0859`, alignment `0.9756`), so there may be a later
  better checkpoint before overfit.
- e1100 was worse online: `13W/243L/0D` over 256 games, nonloss/winrate
  `0.05078125`, mean reward `-0.8984375`. Its offline metrics also showed no
  useful validation gain over e1000 (`val_target_acc=0.5303`, density alignment
  `0.9488`). The count-calibrated run was stopped after e1100. Best online
  checkpoint for this run remains e1000:
  `tinyPPO/runs/regular_bc_dagger_mix_e900_5ckpt_3k_countcal_eval64x4_20260526/online_best.pt`
  and `regular_bc_ray_best_e1000.pt`.
- Same-state Phase 1 gate for the count-calibrated online-best/e1000 checkpoint
  passed and did not regress versus the pure e0180 baseline. Gate output:
  `phase1_gate_online_best_vs_e0180.json`. Candidate summary:
  `density_ratio=0.96795`, `action_f1=0.32563`,
  `source_target_f1=0.34040`, `source_f1=0.46393`,
  `model_actions_per_state=2.097` versus regular `2.167`,
  `pass_phase1=true`. Baseline e0180 on the same gate was
  `density_ratio=1.3766`, `action_f1=0.29803`,
  `source_target_f1=0.31832`, `source_f1=0.45271`. This means the new loss and
  DAgger mix improved same-state imitation/density, but online rollout
  robustness is still below target.
- Interpretation after this run: count calibration fixed much of the
  over-launching and preserved/improved same-state regular imitation, but the
  rollout distribution is still too narrow. The next aligned delta should be a
  second DAgger/offline-expansion pass using the current count-calibrated
  e1000/e1100-style policy states as rollout sources, labelled by regular,
  while keeping the mix near 70-80% original regular and 20-30% current-model
  regular-labelled states. Do not start PPO yet.
- The second DAgger/offline-expansion run is active at
  `tinyPPO/runs/regular_bc_dagger2_countcal_e1000_3k_eval64x4_20260526`.
  It resumes from the count-calibrated e1000 `online_best.pt`, rolls out the
  count-calibrated e1000/e1100 policies for a new current-policy DAgger cache,
  then trains original regular data plus this new regular-labelled rollout
  data with the same count-calibrated BC objective.
- This run uses all GPUs for collection and eval via manual GPU IDs
  `0,1,2,3,4,5,6,7`. DAgger collection uses 64 actors, i.e. 8 workers/GPU.
  Online eval is configured as 64 eval actors with 256 total games, i.e.
  8 workers/GPU and 4 games/worker, evaluated every 100 epochs. Checkpoints are
  saved every 100 epochs.
- Current status: DAgger2 collection is running and has started reporting
  `dagger_actor_progress` (`done=10/47` on several actors, about 160 rows per
  actor at that point). GPU memory is stable around 4.1GB/card during
  collection, and system memory/disk headroom are healthy. No OOM signs so far.
- DAgger2 collection finished and wrote
  `tinyPPO/data/regular_bc_dagger2_countcal_e1000e1100_2p_3000g_rows16_20260526.pkl`
  (about 19GB). Collection summary: 3000 model-rollout games, 47,919 samples,
  277,928 labelled actions, 16,755 skipped actions, 294,683 model-seat regular
  label actions, 64 actors, two rollout checkpoints, model-seat-only enabled.
  The training process then resumed from the count-calibrated e1000
  `online_best.pt` and started BC training from `resume_update=1000` with
  16 train actors and 64 eval actors. Next important evidence is e1100 offline
  metrics plus the 256-game online eval.

```bash
source .venv/bin/activate
PYTHONUNBUFFERED=1 python -m tinyPPO.watch_phase1_gate \
  --run-dir tinyPPO/runs/regular_bc_rulefeat_pairsoftmax_2p_10k_20260526 \
  --pattern 'regular_bc_ray_e*.pt' \
  --device cuda:0 \
  --target-mask-mode all_planets \
  --target-pair-weight 1.0
```

Only start the watcher on a genuinely spare GPU. It runs same-state imitation
gates for newly saved BC checkpoints and writes `phase1_gate_*.json` into the
run directory. It is intentionally separate from the trainer so Phase 1 data
collection/training is not blocked by eval.

## 2026-05-26 DAgger2 follow-up

- The full DAgger2 mix run
  `tinyPPO/runs/regular_bc_dagger2_countcal_e1000_3k_eval64x4_20260526`
  was stopped after e1100. Its 256-game online eval was worse than the previous
  count-calibrated checkpoint: `8W/248L/0D`, nonloss/winrate `0.03125`, mean
  reward `-0.9375`.
- Offline diagnosis showed the e1100 model did learn the DAgger2 rows better,
  but shifted the original regular distribution too conservative. On DAgger2
  cache it improved target/ship metrics, while on original regular cache its
  predicted action density fell to about `0.897x` label density. The likely
  issue is mix balance: DAgger2 rows are action-dense, so full-row mixing gives
  them too much effective action weight.
- `imitation_regular_ray.py` now has `--dagger-max-rows` to deterministically
  subsample cached DAgger rows after load/collection. The active run is
  `tinyPPO/runs/regular_bc_dagger2_countcal_e1000_cap16k_eval64x4_20260526`.
  It resumes from the count-calibrated e1000 `online_best.pt`, uses the same
  objective, but caps DAgger2 to `16000` rows: collect summary confirmed
  `159954` original rows plus `16000` DAgger rows, with
  `samples_before_subsample=47919` and `subsampled=1`.
- This active cap16k run uses 64 train actors and manual GPU IDs
  `0,1,2,3,4,5,6,7`, i.e. 8 training workers/GPU. Eval uses 64 actors and
  `256` total games, i.e. 8 eval workers/GPU and 4 games/worker, every 100
  epochs. Early e1001-e1006 loader metrics are stable; GPUs are fully utilized.
  Next stop/check point is e1100 online eval. If e1100 is still around
  `0.03-0.06` nonloss or original-cache density worsens, stop and rethink data
  selection/weighting rather than increasing epochs.
- Result: cap16k e1100 did not improve online robustness. The scripted final
  async eval was starved because 64 train actors held the Ray CPU resources, so
  `imitation_regular_ray.py` now kills train actors before draining pending
  final evals. A manual 64-process eval of
  `regular_bc_ray_e1100.pt` over the same 256-game shape produced
  `16W/240L/0D`, nonloss/winrate `0.0625`, mean reward `-0.875`.
  The best-imitation checkpoint from this run was e1009, but its manual 256-game
  eval was worse: `13W/243L/0D`, nonloss `0.05078125`.
- Same-state Phase1 gate for cap16k e1100 still passed:
  `phase1_gate_e1100.json` reported `action_f1=0.32859`,
  `source_target_f1=0.34058`, `source_f1=0.45635`,
  `model_actions_per_state=1.9097` versus regular `2.1667`, density ratio
  `0.88141`, and `pass_phase1=true`. This means the checkpoint is still
  regular-like on held-out same states, but it is more conservative than the
  count-calibrated e1000 online-best checkpoint, whose gate density ratio was
  about `0.968`.
- Cache diagnosis explains the online stall. On a 12k-row original regular
  sample, cap16k e1100 under-launches badly:
  `density_ratio=0.8276`, `pred_actions_per_row=1.839` versus labels `2.222`,
  while target/ship metrics are fine (`target_acc=0.5960`, `ship_acc=0.9220`).
  On a 12k-row DAgger2 sample it is slightly action-dense:
  `density_ratio=1.097`, `pred_actions_per_row=6.334` versus labels `5.774`.
  So the issue is not merely target accuracy; the mixed dataset is teaching a
  different action rhythm on original regular states while still not improving
  online distribution drift.
- Decision: do not continue this cap16k run and do not advance to PPO. The
  current best online BC remains the count-calibrated e1000 checkpoint
  (`regular_bc_dagger_mix_e900_5ckpt_3k_countcal_eval64x4_20260526/online_best.pt`),
  but it is still far below the `>0.15` provisional online nonloss target. The
  next useful DAgger delta should change data selection/weighting rather than
  adding epochs: keep original regular dominant, validate original and DAgger
  densities separately, and avoid letting action-dense model-state rows distort
  original-state launch cadence.

## 2026-05-26 5ckpt DAgger loss-weight follow-up

- Added `--dagger-loss-weight` to `imitation_regular_ray.py` and threaded
  per-row sample weights through `stack_rows`/`bc_loss`. This lets cached DAgger
  rows contribute less loss without discarding their state coverage.
- Ran
  `tinyPPO/runs/regular_bc_dagger_5ckpt_lossw05_e900_eval64x4_20260526` from
  the original regular e900 checkpoint with the exact 5-checkpoint DAgger cache
  from e400/e600/e900/e1100/e1300 and `--dagger-loss-weight 0.5`.
  Collection used `159954` original rows plus `47940` DAgger rows; weighted
  row mean was about `0.885`, roughly keeping original regular dominant.
- Eval was configured as requested with `64` eval actors over GPUs
  `0,1,2,3,4,5,6,7`: 8 eval workers/GPU and `256` total games, so 4 games per
  worker. The final e1000 async eval completed successfully.
- Result: e1000 online eval versus regular was worse than the earlier
  count-calibrated best: `10W/246L/0D`, nonloss/winrate `0.0390625`, mean
  reward `-0.921875`. Offline validation still over-launched on held-out mixed
  rows (`val_launch_pred_rate=0.09435` versus `0.07801`, density alignment
  `0.9366`) and target quality stayed around `0.531`.
- Decision: do not continue this loss-weight-0.5 5ckpt run. It confirms that
  simply downweighting the full DAgger cache is not enough. The next delta
  should be more selective about model-state rows, or diagnose per-cache action
  density before another long run.

## 2026-05-27 5ckpt DAgger action-count filtering

- Added DAgger row filters to `imitation_regular_ray.py`:
  `--dagger-min-actions-per-row` and `--dagger-max-actions-per-row`. These only
  filter model-rollout/regular-labelled DAgger rows after cache load or
  collection; original regular rows are untouched.
- The 5ckpt DAgger cache action-count distribution is very long-tailed:
  `47940` rows, `288099` labelled actions, mean `6.01` actions/row, max `38`.
  Keeping rows with `action_count <= 8` leaves `36332` rows and `105274`
  labelled actions, mean `2.90` actions/row. Mixed with the original regular
  cache (`352925` labelled actions), this gives roughly `77/23`
  original-vs-DAgger action mass without extra loss weighting.
- Ran
  `tinyPPO/runs/regular_bc_dagger_5ckpt_actioncap8_e900_eval64x4_20260527`
  from the original regular e900 checkpoint, same 5 ckpts
  e400/e600/e900/e1100/e1300, same count-calibrated objective, and only the
  DAgger action-count cap as the intended data-selection delta. Eval used
  `64` actors and `256` games, i.e. 8 eval workers/GPU and 4 games/worker.
- Collection summary for the run: total `196286` rows, consisting of `159954`
  original rows plus `36332` filtered DAgger rows. DAgger metrics recorded
  `samples_before_action_filter=47940`, `action_filter_keep_frac=0.75786`,
  and `action_filter_mean_actions=2.8976`.
- Offline cadence was better than the full/weighted DAgger branches but still
  not enough: e1000 had `val_launch_pred_rate=0.07284` versus true
  `0.06242`, density alignment `0.94855`, `val_target_acc=0.50736`, and
  `val_imitation_score=0.49071`.
- Result: e1000 online eval versus regular was still poor:
  `11W/245L/0D`, nonloss/winrate `0.04296875`, mean reward `-0.9140625`.
- Decision: do not continue this action-cap8 run. Filtering out high-action
  DAgger rows improves launch cadence but does not solve online distribution
  drift. The next useful step should diagnose which model-rollout states are
  missing or mislabeled for online recovery, rather than merely changing the
  global DAgger row/action ratio.

## 2026-05-27 5ckpt DAgger action-weight cap

- Added `--dagger-action-weight-cap` to `imitation_regular_ray.py`. Unlike
  `--dagger-max-actions-per-row`, this keeps every DAgger row but scales rows
  whose regular label has more than N launch actions by `N/action_count`. The
  intent is to keep high-action recovery states visible while preventing them
  from dominating active-action loss.
- Cache diagnostics showed why this was worth testing. The actioncap8 e1000
  checkpoint matched the cap8 DAgger subset (`density_ratio=1.034`) but
  under-launched on original regular rows (`0.888`) and badly under-launched on
  the full DAgger cache (`0.635`). The full loss-weight-0.5 branch preserved
  the full DAgger cache but over-launched there (`1.157`) while still
  under-launching original rows (`0.902`).
- Ran
  `tinyPPO/runs/regular_bc_dagger_5ckpt_actionwcap35_e900_eval64x4_20260527`
  from original regular e900 with the same 5 DAgger checkpoints and
  `--dagger-action-weight-cap 3.5`. Collection loaded all `47940` DAgger rows,
  with `dagger_weight_mean=0.72376`, weighted DAgger labelled actions
  `117138`, and an effective original-vs-DAgger action mass around `75/25`.
- Offline validation looked cleaner than the failed row-cap branch: e1000 had
  `val_launch_pred_rate=0.08551` versus true `0.07801`, density alignment
  `0.96939`, `val_target_acc=0.52816`, and best imitation score `0.52393`.
- Result: online eval still did not improve:
  `10W/246L/0D`, nonloss/winrate `0.0390625`, mean reward `-0.921875`.
- Decision: do not continue this action-weight-cap run. Existing 5ckpt DAgger
  cache has now failed under full mixing, row subsampling/capping, constant
  downweighting, and action-count-capped weighting. The next delta should not
  be another global row/action weighting rule on the same cache.
- `DaggerCollectActor` now attaches metadata to newly generated DAgger rows:
  checkpoint path, seed, player count, model seat, player id, sampled raw index,
  observed step (when present), model action count, and regular-label action
  count. A 4-game smoke collect verified these fields are present. New DAgger
  caches should use this metadata to select states by origin, timestep, model
  action/regular-label mismatch, and likely failure/recovery segments rather
  than only global action-count distribution.

## 2026-05-27 64-trainer continuation and DAgger metadata fix

- Started
  `tinyPPO/runs/regular_bc_dagger_mix_e1000_5ckpt_3k_countcal_64tr_eval4pg_20260527`
  from the current best DAgger-mix online checkpoint
  `regular_bc_dagger_mix_e900_5ckpt_3k_countcal_eval64x4_20260526/online_best.pt`
  (e1000). This is a resource/utilization continuation only: same original
  regular cache, same 5ckpt DAgger cache, same count-calibrated objective.
- Parallelism was increased to `64` train actors and `64` eval actors with
  manual GPU IDs `0..7`, i.e. 8 workers/GPU. Online eval remains every `100`
  epochs with `256` games total, so each eval actor runs 4 games. Checkpoints
  are still saved every `100` epochs. The run is healthy as of e1037, with all
  8 GPUs at full utilization and no OOM.
- Fixed DAgger metadata for future caches. The previous smoke test showed
  `dagger_obs_step=60`, which was actually `remainingOverageTime` on nonzero
  player observations, not the real turn. `DaggerCollectActor` now tracks the
  current turn from player-0 observations and stores `dagger_turn_index` /
  `dagger_obs_step` from that value.
- Added per-row final-outcome metadata for future DAgger caches:
  `dagger_game_length`, `dagger_model_final_reward`, and
  `dagger_model_final_status`. A smoke collect with one 80-step game verified
  realistic metadata: e.g. `dagger_turn_index=4`, `dagger_game_length=79`,
  and `dagger_model_final_reward=-1.0`.
- Added `tinyPPO/summarize_dagger_cache.py` to summarize newly generated DAgger
  caches by checkpoint, turn bucket, final reward bucket, and
  regular-label-vs-model action-count gap. This should be used before the next
  long DAgger training run so the next data expansion targets failure and
  mismatch states instead of applying another global mix/weight rule.
- Added metadata-aware DAgger training filters for future runs:
  `--dagger-min-turn`, `--dagger-max-turn`,
  `--dagger-min-final-reward`, `--dagger-max-final-reward`, and
  `--dagger-min-abs-action-gap`. Defaults are disabled. These let the next
  BC/DAgger run focus on model-created losing states or action-mismatch states
  while still mixing them with the original regular cache.
- The 64-trainer continuation reached e1100 and was stopped after online eval:
  `15W/241L/0D`, nonloss/winrate `0.05859375`. This is effectively unchanged
  from the previous best e1000 (`16W/240L/0D`) and far below the `>0.15`
  distribution-drift target. Do not keep blind-training this old cache toward
  e10000.
- Added `tinyPPO/collect_dagger_cache.py`, a standalone DAgger cache collector
  that does not load the 61GB original BC cache or start training. Started a
  fresh metadata cache collection:
  `tinyPPO/data/regular_bc_dagger_metadata_5ckpt_2p_3000g_rows16_20260527.pkl`
  using the same 5 checkpoints, 3000 2P games, 64 actors, model-seat-only
  rows, and 16 rows/game. This cache should replace the old metadata-poor
  5ckpt DAgger cache for the next selective BC run.

## 2026-05-27 selective metadata DAgger plan

- The metadata DAgger cache completed successfully:
  `tinyPPO/data/regular_bc_dagger_metadata_5ckpt_2p_3000g_rows16_20260527.pkl`
  with `3000` games, `47974` sampled model-seat rows, `349218`
  regular-labelled actions, and `333709` usable labelled actions after skip
  handling.
- Summary by turn showed the main distribution gap is action under-launching
  after the first ~100 turns. Mean regular-label minus model action count is
  about `1.44` in turns `50-99`, `6.79` in turns `100-149`, and `8.73` in
  turns `150-199`; very late rows have even larger gaps but risk dominating
  the loss with recovery states.
- Next run should use the new metadata cache with
  `--dagger-min-turn 50 --dagger-max-turn 199 --dagger-action-weight-cap 5`.
  This keeps the early-mid model-state drift visible while capping high-action
  rows, giving roughly a 74/26 original-vs-DAgger effective action mass.
- Training/eval resource target for this run: `64` train actors and `64` eval
  actors across 8 GPUs, i.e. 8 workers/GPU. Online eval stays every `100`
  epochs; `--eval-games 256` means each eval actor runs 4 games.
- Started
  `tinyPPO/runs/regular_bc_dagger_metadata_turn50_199_cap5_e1000_eval4pg_20260527`
  from the previous online-best e1000 checkpoint. The run uses 64 trainers and
  64 eval actors with manual GPU ids `0..7`; all 8 GPUs reached full
  utilization during training.
- Early epochs after resume are healthy on same-state imitation. Around
  e1012-e1033, validation launch density alignment stayed near `0.96-0.97`
  and did not show the immediate Phase1 collapse that would indicate the
  selective DAgger mix is too strong.
- `tinyPPO/watch_phase1_gate.py` now uses `sys.executable` for its subprocess
  call so watcher-launched gates run under the same virtualenv. A watcher is
  running for this run and will automatically gate `regular_bc_ray_e*.pt`
  checkpoints against the previous online-best baseline as they appear.
- The `turn50-199 cap5` run reached e1100 and failed the online distribution
  drift target: `16W/240L/0D` over 256 games, nonloss/winrate `0.0625`, mean
  reward `-0.875`. This is the same as the previous best e1000 and does not
  show improvement toward the `>0.15` nonloss target.
- The e1100 same-state validation did not collapse before stop
  (`val_launch_density_alignment=0.96824`, `val_imitation_score=0.62114`), so
  the failure is not an obvious Phase1 imitation collapse; the selected
  model-state rows were just not sufficient to improve online behavior.
- Stopped that branch after e1100 rather than continuing toward e1300. Next
  delta is a more focused metadata filter: `--dagger-min-turn 100
  --dagger-max-turn 249 --dagger-action-weight-cap 7`, still from the same
  e1000 online-best warm start and with the same 64 trainer / 64 eval actor
  setup.
- Started
  `tinyPPO/runs/regular_bc_dagger_metadata_turn100_249_cap7_e1000_eval4pg_20260527`.
  The loaded DAgger subset has `18864` rows, `180829` unweighted labelled
  actions, and `95018` weighted labelled actions, giving an effective DAgger
  action mass of about `21%` against the original `352925` regular labelled
  actions. This is a cleaner test of the midgame under-launch gap than the
  previous broader `50-199 cap5` branch.
- The `turn100-249 cap7` branch reached e1100 and also failed online:
  `13W/243L/0D` over 256 games, nonloss/winrate `0.05078125`, mean reward
  `-0.8984375`. Same-state validation still looked stable
  (`val_launch_density_alignment=0.9757`, `val_imitation_score=0.61909`), so
  the branch did not fail because it forgot regular on the held-out cache.
- Offline diagnosis at e1100 showed the model was already over-launching on
  both the original regular cache and the selected DAgger subset:
  original `density_ratio=1.0569`, DAgger turn100-249
  `density_ratio=1.1087`. The old e1000 warm start was also over-dense on the
  same diagnostics, especially on the selected DAgger subset
  (`density_ratio=1.1890`). This argues against spending more time only
  pulling launch-count density on these static rows.
- Added explicit eval fanout support:
  `--eval-games-per-actor`. When this is set, the trainer uses all configured
  eval actors and submits that many games to each actor. This avoids accidental
  under-use when we want fixed games/worker.
- Continued from the e1100 checkpoint in
  `tinyPPO/runs/regular_bc_dagger_metadata_turn100_249_cap7_e1100_eval8pgpu4g_20260527`
  with 64 trainers and 64 eval actors across 8 GPUs, using
  `--eval-games-per-actor 4`. The e1200 eval submission confirmed
  `parts=64`, `games=256`, and `games_per_actor=4`; all GPUs were saturated
  during training.
- e1200 online result worsened to `7W/249L/0D` over 256 games,
  nonloss/winrate `0.02734375`, mean reward `-0.9453125`. Stop continuing this
  static metadata-cache branch. The next useful delta should diagnose/collect
  on actual failed rollouts from the current policy instead of adding more
  epochs on the same DAgger rows.
- Added `tinyPPO/collect_failed_rollout_dagger_cache.py` for the next DAgger
  data step. It runs a checkpoint against regular, logs only the model-seat
  observations, calls regular on those same observations for labels, stores
  `BCRow`s with model-vs-regular action counts, score-based outcome metadata,
  turn buckets, and action-gap buckets. It uses the fast simulator and supports
  Ray actors plus manual GPU ids.
- Important correction: do not use fast-env `reward` alone to identify failed
  model rollouts. The online eval code defines win/loss by comparing
  `score(obs, model_seat)` against the opponent score, so the new collector
  supports `--keep-outcomes loss,draw` and records
  `dagger_model_outcome`. It also supports seed-band collection to reproduce
  async eval seeds via `--seed-base-start`, `--seed-base-count`,
  `--seed-base-stride`, and `--games-per-seed-base`.
- Smoke collection on e1200:
  `tinyPPO/data/failed_rollout_dagger_e1200_smoke_2p_16g_rows8_20260527.pkl`
  produced 40 rows from failed model rollouts. On those rows, regular averaged
  `2.5` actions/row while the model averaged `1.375`, confirming that actual
  failed rollout states still include an under-action gap.
- Diagnostic collection on the e1200 eval seed band:
  `tinyPPO/data/failed_rollout_dagger_e1200_evalseed_scoreloss_2p_256g_rows12_20260527.pkl`
  used `--seed-base-start 1470525 --seed-base-count 64
  --seed-base-stride 100 --games-per-seed-base 4 --min-turn 50`. It produced
  648 rows from score-loss outcomes. Regular averaged `2.171` actions/row,
  model averaged `0.998`, mean action gap `1.173`; rows concentrated in turns
  `50-99` and `100-149`.
- A second eval-seed collection without the `min-turn` filter:
  `tinyPPO/data/failed_rollout_dagger_e1200_evalseed_scoreloss_2p_256g_rows12_allturn_20260527.pkl`
  produced 696 rows. It shows a substantial early-failure component:
  213 rows in turns `0-49`, 203 in `50-99`, and 100 in `100-149`.
  Regular averaged `2.138` actions/row, model averaged `1.312`, mean action
  gap `0.826`.
- Next training delta should not use only the old 5ckpt static metadata cache.
  Mix the original regular cache with one or more current-policy failed-rollout
  caches, with low effective DAgger mass first because the new cache is small
  but high signal. Keep the stop rule strict: same-state gate must not drop,
  and online nonloss must move above the current 0.03-0.06 band before spending
  more GPU time.
- Built the first mixed-current-policy cache:
  `tinyPPO/data/regular_bc_dagger_mix_5ckpt50_199_50ka_e1200loss_allturn_w30_20260527.pkl`.
  It combines 50k labelled actions from the old 5-checkpoint turn 50-199
  metadata cache at weight 1.0 with 561 rows / 1315 labelled actions from the
  e1200 eval-seed failed-rollout all-turn cache at weight 30.0. Effective
  DAgger action mass is about 89.5k weighted actions versus 352.9k original
  regular actions, roughly a 20% DAgger fraction.
- Ran
  `tinyPPO/runs/regular_bc_dagger_mix_old5_50ka_e1200loss_w30_e1000_eval4pg_20260527`
  from the old e1000 online-best warm start with 64 trainer workers and 64 eval
  actors across 8 GPUs. Eval used `--eval-games-per-actor 4`, so the e1100
  submission was exactly 64 actors x 4 games = 256 games. GPU utilization was
  saturated during training, around 36 GB per GPU.
- e1100 online result was `14W/242L/0D` over 256 games, nonloss/winrate
  `0.0546875`, mean reward `-0.890625`. This is still in the same 0.03-0.06
  band and below the old e1000 best `16W/240L` result, so this mixed-cache
  branch is not a breakthrough. Same-state metrics also showed late overfit
  pressure: train target/pair continued improving while validation pair
  accuracy drifted down into the low `0.53` range by e1100.
- Stop this branch here. The failed-rollout cache is useful diagnostically, but
  simply upweighting this tiny current-policy cache does not fix online play.
  Next delta should either collect a much broader current-policy failed-rollout
  cache over more seed bands/checkpoints or change the supervised target
  formulation; do not spend more epochs on this exact mixture.

## 2026-05-27 broader failed-rollout DAgger cache

- Collected broader current-policy failed-rollout DAgger data from five original
  long-run checkpoints: e0400/e0600/e0900/e1100/e1300 from
  `regular_bc_rulefeat_pairsoftmax_2p_10k_continue_e10000_2w_eval64_20260526`.
  Each checkpoint was evaluated for 256 games against regular using the fast
  simulator, keeping only score-loss/draw model-seat states and labelling those
  same observations with regular.
- Collection used 64 actors across GPUs `0..7`, i.e. 8 workers/GPU. The final
  caches were:
  - `failed_rollout_dagger_5ckpt_e0400_2p_256g_rows16_lossdraw_20260527.pkl`:
    53 kept loss games, 836 rows, 1358 labelled actions.
  - `failed_rollout_dagger_5ckpt_e0600_2p_256g_rows16_lossdraw_20260527.pkl`:
    52 kept loss games, 831 rows, 1478 labelled actions.
  - `failed_rollout_dagger_5ckpt_e0900_2p_256g_rows16_lossdraw_20260527.pkl`:
    43 kept loss games, 677 rows, 1408 labelled actions.
  - `failed_rollout_dagger_5ckpt_e1100_2p_256g_rows16_lossdraw_20260527.pkl`:
    48 kept loss games, 767 rows, 1643 labelled actions.
  - `failed_rollout_dagger_5ckpt_e1300_2p_256g_rows16_lossdraw_20260527.pkl`:
    64 kept loss games, 1023 rows, 1750 labelled actions.
- Across these failed rollout states, regular consistently wanted more actions
  than the model. Mean regular-label action counts were about `1.88`, `2.08`,
  `2.35`, `2.39`, and `1.94` actions/row, while model action counts were about
  `0.65`, `0.64`, `0.96`, `0.99`, and `0.87`. Many failures already appear in
  turns `0-49` and `50-99`, so the drift is not only a late-game recovery
  issue.
- Built
  `tinyPPO/data/regular_bc_dagger_mix_failed5ckpt_lossdraw_w15_20260527.pkl`
  from the five failed-rollout caches with per-source weight `15.0`. The mix
  contains 3370 rows, 7637 labelled actions, and 114555 weighted labelled
  actions. Against the original regular cache's 352925 labelled actions this is
  roughly a 24.5% effective DAgger action mass.
- Ran
  `tinyPPO/runs/regular_bc_dagger_failed5ckpt_lossdraw_w15_e1000_eval4pg_20260527`
  from the previous online-best e1000 checkpoint. Training used 64 train actors
  across GPUs `0..7`; eval used 64 eval actors with
  `--eval-games-per-actor 4`, i.e. exactly 8 eval workers/GPU and 256 total
  games per eval. GPUs reached full utilization during training.
- Result at e1100: online eval versus regular was `17W/239L/0D` over 256
  games, nonloss/winrate `0.06640625`, mean reward `-0.8671875`. This is a
  small improvement over the previous `16W/240L` online-best but still far
  below the provisional `>0.15` nonloss target.
- Same-state validation did not collapse, but showed some late DAgger-bias
  pressure by e1100: `val_launch_density_alignment=0.96783`,
  `val_target_acc=0.95282`, `val_ship_acc=0.91497`, while
  `val_target_pair_acc` drifted down to about `0.5300`.
- Decision: this broader failed-state cache is the best online result so far
  but not a breakthrough. Do not start PPO or call Phase 1 done. The next data
  delta should broaden current-policy state coverage beyond only losing
  rollouts: collect model-rollout states from all outcomes or balance
  win/loss/draw and turn buckets, then mix at lower or balanced effective
  weight so original regular cadence is preserved.

## 2026-05-27 all-outcome rollout DAgger check

- Tested the broader-coverage hypothesis by collecting all score outcomes,
  not only loss/draw, from the same five original long-run checkpoints
  e0400/e0600/e0900/e1100/e1300. This used
  `--keep-outcomes loss,draw,win`, 128 games per checkpoint, 64 actors across
  GPUs `0..7`, i.e. 8 rollout workers/GPU.
- Resulting all-outcome caches:
  - e0400: 128 games, 2038 rows, 12788 labelled actions,
    outcome games `102W/26L`, regular/model action means `6.53/2.89`.
  - e0600: 128 games, 2037 rows, 13185 labelled actions,
    outcome games `103W/25L`, regular/model action means `6.77/2.68`.
  - e0900: 128 games, 2038 rows, 12691 labelled actions,
    outcome games `97W/31L`, regular/model action means `6.51/2.53`.
  - e1100: 128 games, 2037 rows, 13196 labelled actions,
    outcome games `105W/23L`, regular/model action means `6.80/2.61`.
  - e1300: 128 games, 2040 rows, 14044 labelled actions,
    outcome games `98W/30L`, regular/model action means `7.19/2.51`.
- Even in model-winning games, the model-created states are not regular-like:
  regular labels ask for roughly 3.6-4.7 more actions/row than the model emits.
  The distribution gap is therefore broader than only "failed recovery" states.
- Built
  `tinyPPO/data/regular_bc_dagger_mix_alloutcome5ckpt_w175_20260527.pkl`
  using weight `1.75` for each source. The mix has 9310 rows, 65904 labelled
  actions, and 115332 weighted labelled actions, giving about the same
  effective DAgger mass as the previous failed-only run.
- Ran
  `tinyPPO/runs/regular_bc_dagger_alloutcome5ckpt_w175_e1000_eval4pg_20260527`
  from the previous online-best e1000 checkpoint. Training used 64 train actors
  and eval used 64 eval actors with `--eval-games-per-actor 4`, so e1100 eval
  was exactly 64 actors x 4 games = 256 games.
- Result at e1100: online eval versus regular regressed to `12W/244L/0D`,
  nonloss/winrate `0.046875`, mean reward `-0.90625`. This is worse than the
  failed-only broader cache (`17W/239L`) and worse than the old online-best
  checkpoint (`16W/240L`).
- Same-state validation stayed superficially healthy but showed the familiar
  late static-cache pressure: e1100 `val_launch_density_alignment=0.95638`,
  `val_target_acc=0.95710`, `val_ship_acc=0.91398`, while
  `val_target_pair_acc` drifted down to `0.53915`.
- Decision: do not continue this all-outcome static mix. The collection is
  diagnostically useful because it proves drift exists even in model-winning
  states, but naive all-outcome mixing at 20-30% effective action mass hurts
  online play. The next data delta should balance outcome and turn buckets, or
  lower/bucket-cap the all-outcome DAgger weight, rather than simply training
  longer on this mix.

## 2026-05-27 bucket-balanced all-outcome DAgger check

- Added `tinyPPO/build_dagger_bucket_mix_cache.py`. It pools multiple rollout
  caches, assigns rows to explicit `outcome|min_turn|max_turn` buckets, samples
  up to a per-bucket labelled-action cap, and writes
  `dagger_loss_weight_override` so the selected rows can be mixed into BC
  without changing the trainer.
- Built a first conservative bucket-balanced cache from the all-outcome
  e0400/e0600/e0900/e1100/e1300 rollout caches:
  `tinyPPO/data/regular_bc_dagger_mix_alloutcome5ckpt_bucketbal_w50_20260527.pkl`.
  Buckets kept all loss rows and capped win rows by turn:
  loss 0-49/50-99/100-149/150-199/200-499 plus win
  0-49/50-99/100-149/150-199/200-499. The result has 4373 rows, 20724 raw
  labelled actions, and 103620 weighted labelled actions at row weight `5.0`,
  about a 22.7% effective DAgger action mass versus the original 352925 regular
  labelled actions.
- Bucket content showed the collected loss states are sparse in action mass:
  all loss buckets together contain only 3724 raw labelled actions. Most of the
  balanced cache's action mass still comes from capped win early/midgame rows.
- Ran
  `tinyPPO/runs/regular_bc_dagger_alloutcome5ckpt_bucketbal_w50_e1000_eval4pg_20260527`
  from the previous online-best e1000 checkpoint. Training used 64 train actors;
  eval used 64 eval actors with `--eval-games-per-actor 4`, so e1100 eval was
  again exactly 256 games with 8 eval workers/GPU.
- Result at e1100: online eval versus regular was again `12W/244L/0D`,
  nonloss/winrate `0.046875`, mean reward `-0.90625`. This matches the naive
  all-outcome branch and is worse than the failed-only cache's `17W/239L`.
- Same-state validation was density-stable but target-pair generalization kept
  drifting down: e1100 `val_launch_density_alignment=0.96589`,
  `val_target_acc=0.95402`, `val_ship_acc=0.91851`, but
  `val_target_pair_acc=0.53254`.
- Decision: do not continue this branch. Outcome/turn bucketing avoided the
  worst density distortion but did not improve online distribution drift. The
  next aligned step should either collect fresh states from the current best
  failed-only/e1000-style policy rather than old pure-BC checkpoints, or change
  the supervised target formulation so regular-labelled model states teach
  action rhythm and target ranking without forcing every high-action regular
  rescue row as a static BC target.

## 2026-05-27 current-best failed-state DAgger check

- Collected fresh failed rollout states from the current best online checkpoint
  `regular_bc_dagger_failed5ckpt_lossdraw_w15_e1000_eval4pg_20260527/online_best.pt`
  instead of older pure-BC checkpoints. The collection used 64 rollout actors
  across GPUs `0..7` (8 workers/GPU), 512 games, stochastic actions,
  `launch_bias=-0.25`, `rows_per_game=16`, and kept only loss/draw games.
- Saved
  `tinyPPO/data/failed_rollout_dagger_currentbest_2p_512g_rows16_lossdraw_20260527.pkl`.
  It kept 86 losing games, 1358 sampled rows, 2340 labelled launch actions,
  298 skipped/noop label actions, and 1731 model actions. The average row still
  had regular asking for more actions than the model emitted
  (`1.94` regular actions/row vs `1.27` model actions/row).
- Built
  `tinyPPO/data/regular_bc_dagger_mix_currentbest_lossdraw512_w40_20260527.pkl`
  from the action-bearing rows only: 1009 rows, 2340 raw labelled actions, and
  93600 weighted labelled actions with row weight `40.0`. This is about 21%
  effective DAgger action mass versus the original 352925 regular labelled
  actions.
- Ran
  `tinyPPO/runs/regular_bc_dagger_currentbest_lossdraw512_w40_e1100to1200_eval4pg_20260527`
  from the current best checkpoint. Training used 64 train actors and eval used
  64 eval actors with `--eval-games-per-actor 4`, so e1200 eval was exactly
  256 games with 8 workers/GPU.
- Result at e1200: online eval versus regular regressed badly to `7W/249L/0D`,
  nonloss/winrate `0.02734375`, mean reward `-0.9453125`. This is far below
  the current best branch's `17W/239L`.
- Same-state metrics also showed the failure mode: density stayed roughly
  acceptable (`val_launch_density_alignment=0.96055`) but target-pair
  validation drifted down to `0.53657`, while train target-pair kept improving.
- Decision: do not continue this branch and do not promote its checkpoint. A
  tiny set of current-policy failure rows weighted to 20% action mass is too
  sharp and overfits target ranking. The next supervised delta should either
  lower the weight substantially and bucket-cap by turn, or change the target
  formulation so failure-state labels teach launch rhythm without overpowering
  pair ranking.

## 2026-05-27 DAgger row-weight plumbing fix and launch-only weighting check

- Found and fixed an important training plumbing bug in
  `tinyPPO/imitation_regular_ray.py`: cache rows with
  `dagger_loss_weight_override` were ignored unless command-line
  `--dagger-loss-weight != 1` or `--dagger-action-weight-cap > 0`. That meant
  cache-built weighted mixes could report `weighted_labelled_actions` in their
  cache metrics while training with `sample_weight_mean=1.0`. The trainer now
  enables sample weights whenever any DAgger row has a non-1 override or
  `sample_weight`.
- Added per-component sample-weight scales in `tinyPPO/imitation_regular.py`
  and `tinyPPO/imitation_regular_ray.py`:
  `--sample-weight-launch-scale`, `--sample-weight-count-scale`,
  `--sample-weight-target-scale`, `--sample-weight-ship-scale`, and
  `--sample-weight-pair-scale`. Defaults preserve historical behavior. This
  lets DAgger rows teach launch/count rhythm without multiplying target/pair
  ranking losses by the same factor.
- Re-ran the current-best failed-state cache with fixed row-weight plumbing:
  `tinyPPO/runs/regular_bc_dagger_currentbest_lossdraw512_w40_launchonly_fixedweights_e1100to1200_eval4pg_20260527`.
  It used the same cache
  `regular_bc_dagger_mix_currentbest_lossdraw512_w40_20260527.pkl`, but with
  target/ship/pair sample-weight scales set to `0.0` and launch/count scales
  left at `1.0`.
- The fixed run confirmed weights were active:
  `dagger_weight_mean=40.0`; train `launch_sample_weight_mean` and
  `count_sample_weight_mean` were about `1.2457`, while target/ship/pair weight
  means stayed at `1.0`.
- Result at e1200 was still bad: online eval versus regular was
  `7W/249L/0D`, nonloss/winrate `0.02734375`, mean reward `-0.9453125`.
  Same-state validation showed launch cadence and target-pair drift:
  `val_launch_density_alignment=0.95963`,
  `val_target_pair_acc=0.53764`.
- Decision: fixed row weighting proves the high-weight hypothesis is actively
  harmful, not just noisy. The next DAgger data delta should use much lower
  effective launch/count mass or a larger/better-balanced rollout cache; do not
  promote this checkpoint.

## 2026-05-27 conservative launch/count DAgger weighting check

- Re-ran the same current-best failed-state cache with fixed row-weight
  plumbing, but reduced launch/count amplification to `0.25` while keeping
  target/ship/pair amplification at `0.0`:
  `tinyPPO/runs/regular_bc_dagger_currentbest_lossdraw512_w40_launchcount025_fixedweights_e1100to1200_eval4pg_20260527`.
  With cache row weight `40.0`, this makes DAgger rows count as `10.75x` only
  for launch/count losses and `1.0x` for target/ship/pair losses.
- The run confirmed active but conservative component weights:
  train `launch_sample_weight_mean` and `count_sample_weight_mean` were about
  `1.0614`; target/ship/pair sample-weight means stayed at `1.0`.
- Result at e1200 was still below the current best: online eval versus regular
  was `9W/247L/0D`, nonloss/winrate `0.03515625`, mean reward `-0.9296875`.
  Same-state validation ended at
  `val_launch_density_alignment=0.94607`,
  `val_target_pair_acc=0.53732`,
  `val_target_acc=0.95563`, and `val_ship_acc=0.91475`.
- Decision: even conservative weighting on this tiny current-best failure cache
  hurts online play. Do not promote this checkpoint and do not spend more GPU
  time sweeping weights on this exact cache. The next useful delta should be
  data quality/coverage: collect a larger, turn-balanced rollout-state cache
  from the current best (or multiple nearby checkpoints) and keep the true
  component weights low, with online eval as the stop criterion.

## 2026-05-27 fixed-weight 5-ckpt all-outcome DAgger check

- Re-ran the existing e400/e600/e900/e1100/e1300 all-outcome bucket-balanced
  cache after fixing row-weight plumbing:
  `tinyPPO/runs/regular_bc_dagger_alloutcome5ckpt_bucketbal_w50_fixedweights_e1100to1200_eval4pg_20260527`.
  This used
  `tinyPPO/data/regular_bc_dagger_mix_alloutcome5ckpt_bucketbal_w50_20260527.pkl`,
  resumed from the current best
  `regular_bc_dagger_failed5ckpt_lossdraw_w15_e1000_eval4pg_20260527/online_best.pt`,
  and kept the intended full-component DAgger row weight `5.0`.
- The cache mix matches the requested 20-30% DAgger action-mass regime:
  20724 raw DAgger labelled actions, 103620 weighted labelled actions, versus
  352925 original regular labelled actions. Runtime confirmed the weight was
  active with `dagger_weight_mean=5.0` and train/val
  `sample_weight_mean` about `1.106`/`1.111`.
- Same-state validation did not satisfy the "Phase1 gate should not drop"
  criterion. Launch density stayed recoverable, but target-pair generalization
  split badly: train `target_pair_acc` rose to `0.60018` by e1200 while
  `val_target_pair_acc` fell to `0.53772` and `val_target_acc` ended at
  `0.95714`.
- Online eval at e1200 versus regular was `7W/249L/0D`,
  nonloss/winrate `0.02734375`, mean reward `-0.9453125`. This is much worse
  than the current best `17W/239L/0D`.
- Decision: do not promote this checkpoint and do not continue full-component
  20-30% all-outcome DAgger mixing as-is. The useful next delta is not "more
  of this"; target/pair labels from model-rollout states need separate
  protection. Keep DAgger expansion, but either amplify only launch/count with
  much lower effective weight, or rebuild the cache with stricter turn/outcome
  caps before touching target/pair losses again.

## 2026-05-27 fixed-weight 5-ckpt launch/count-only DAgger check

- Ran the same e400/e600/e900/e1100/e1300 all-outcome bucket-balanced cache,
  but protected target/ship/pair losses from DAgger row amplification:
  `tinyPPO/runs/regular_bc_dagger_alloutcome5ckpt_bucketbal_w50_launchcount_fixedweights_e1100to1200_eval4pg_20260527`.
  Settings kept cache row weight `5.0`, with
  `--sample-weight-launch-scale 1.0`,
  `--sample-weight-count-scale 1.0`, and target/ship/pair scales `0.0`.
- Runtime confirmed the intended component weighting: train/val
  `launch_sample_weight_mean` and `count_sample_weight_mean` were about
  `1.106`/`1.111`, while target/ship/pair component sample weights stayed at
  `1.0`.
- This protected target/pair early, but did not hold through e1200. Same-state
  validation ended at `val_launch_density_alignment=0.95143`,
  `val_target_acc=0.95720`, `val_target_pair_acc=0.53558`,
  and `val_ship_acc=0.91961`. The shared representation still drifted even
  without amplifying target/ship/pair losses.
- Online eval at e1200 versus regular was only `8W/248L/0D`,
  nonloss/winrate `0.03125`, mean reward `-0.9375`, still worse than the
  current best `17W/239L/0D`.
- Decision: do not promote this checkpoint. The core issue is no longer just
  component weighting; this 5-ckpt all-outcome cache is either too broad/noisy
  or its target labels conflict with stable same-state generalization. Next
  DAgger step should use a smaller, explicitly gated cache: earlier stopping
  around the best same-state epoch, stricter turn/outcome caps, or freeze/
  distill target-pair behavior while adapting only launch/count.

## 2026-05-27 source-head-only DAgger freeze check

- Ran the same e400/e600/e900/e1100/e1300 all-outcome bucket-balanced cache
  with only `source_head` trainable:
  `tinyPPO/runs/regular_bc_dagger_alloutcome5ckpt_bucketbal_w50_sourcehead_fixedweights_e1100to1200_eval4pg_20260527`.
  This kept the cache row weight at `5.0`, amplified only launch/count losses,
  froze target/ship/pair modules, and used 64 trainers / 64 eval actors across
  8 GPUs (`8` workers per GPU). Online eval used `4` games per actor for
  `256` games total.
- Runtime confirmed the intended protected update: train/val
  `launch_sample_weight_mean` and `count_sample_weight_mean` stayed about
  `1.106`/`1.111`, while target/ship/pair component sample weights stayed at
  `1.0`. Same-state target/pair was stable by construction:
  e1200 `val_target_pair_acc=0.59651`, `val_target_acc=0.96586`,
  `val_ship_acc=0.92546`.
- The source/count adaptation still hurt online play. E1200 online eval versus
  regular was `5W/251L/0D`, nonloss/winrate `0.01953125`, mean reward
  `-0.9609375`. This is below both the current best `17W/239L/0D` and the
  previous all-component/launch-count DAgger checks.
- Decision: do not promote. Freezing target/pair prevents same-state metric
  drift, but does not make this all-outcome cache useful. The failure is now
  clearly data-distribution/selection rather than only head interference. Next
  DAgger work should rebuild a stricter cache instead of sweeping weights:
  fewer early-turn rows, explicit losing-state focus, outcome/turn caps, and a
  same-state plus online gate before any long run.

## 2026-05-27 strict current-best losing-state DAgger check

- Collected a stricter current-best DAgger cache from
  `regular_bc_dagger_failed5ckpt_lossdraw_w15_e1000_eval4pg_20260527/online_best.pt`:
  `tinyPPO/data/failed_rollout_dagger_currentbest_strictloss_2p_1024g_rows8_t50_gap1_20260527.pkl`.
  Collection used 1024 two-player games, 64 actors, score-outcome `loss` only,
  turns `50..300`, at most 8 sampled rows per kept game, and
  `min_abs_action_gap=1`.
- Cache stats: 1024 games produced 192 score losses; 186 loss games survived
  row filtering. The final cache has 1455 rows, 3233 labelled launch actions
  from `launch_mask`, 3680 raw regular actions, and 1839 model actions. The
  mean action gap is positive (`+1.265`), so the selected failures mostly show
  the current best under-launching relative to regular. Turn distribution is
  still early-heavy but no longer includes turn 0-49:
  `50-99:824`, `100-149:333`, `150-199:163`, `200-249:72`, `250-299:63`.
- Trained a narrow source-head-only correction run:
  `tinyPPO/runs/regular_bc_dagger_currentbest_strictloss_w5_sourcehead_e1100to1200_eval4pg_20260527`.
  It resumed the current best at e1100, used the strict cache directly with
  `--dagger-loss-weight 5.0`, and kept DAgger amplification only on
  launch/count losses. This is a small correction: 3233 raw labelled actions,
  16165 weighted labelled actions, about 4.6% of the original BC action mass.
- Offline gates looked healthier than the all-outcome runs. E1200 ended with
  `val_target_pair_acc=0.60221`, `val_target_acc=0.96698`,
  `val_ship_acc=0.92380`, `val_launch_density_alignment=0.96387`, and
  `val_action_count_mae=1.47100`. The target/pair behavior stayed frozen and
  action-count error improved somewhat.
- Online eval still failed: e1200 versus regular was `5W/251L/0D`,
  nonloss/winrate `0.01953125`, mean reward `-0.9609375`, far below the
  current best `17W/239L/0D`.
- Decision: do not promote. This rules out "small BC correction on
  under-launching failure states" as sufficient. The next stage needs a
  rollout-level objective or policy-selection change: evaluate action-count
  calibration in actual rollouts, inspect losing game traces, and consider
  training/evaluating candidate policies with a direct rollout metric before
  adding more DAgger rows.

## 2026-05-27 rollout behavior diagnosis after strict-loss DAgger

- Added and ran `tinyPPO/diagnose_rollout_behavior.py` to compare model actions
  against regular actions on the exact same model-controlled rollout states.
  Full output:
  `tinyPPO/runs/rollout_behavior_current_vs_strictloss_128g_20260527.json`.
- The comparison used 128 two-player games per checkpoint, 64 actors, the same
  seed shards, and the fast simulator. Because this seed set is much more
  favorable than the standard online eval, use it for behavior comparison only,
  not as a promotion score.
- Current best behavior on this diagnostic set:
  `95W/33L/0D`, nonloss `0.74219`. Overall, the model produced `1.74`
  actions/state while regular would produce `4.39` on those same observations.
  The gap is concentrated after turn 100: turn `100-149` has model `2.19`
  versus regular `6.52`; turn `150-199` has model `1.40` versus regular
  `6.00`. In winning trajectories the gap is even clearer: model `2.39`
  versus regular `6.59`.
- The strict-loss source-head checkpoint looked better on these particular
  seeds (`102W/26L/0D`, nonloss `0.79688`), but it did not fix the imitation
  discrepancy. Overall action gap grew to `3.13`, and the midgame gap remained:
  turn `100-149` model `2.19` versus regular `7.14`; turn `150-199` model
  `1.22` versus regular `6.42`. In winning trajectories it was still model
  `2.26` versus regular `6.67`.
- Crucial conclusion: the previous strict-loss cache targeted the wrong part of
  the rollout distribution. Losing-game states are mostly dead/low-action
  states, where regular and model are already close (`loss` rows are about
  `0.67` model actions versus `0.68-0.78` regular actions). The real drift is
  healthy midgame and winning/nonterminal states where regular keeps launching
  several fleets and the model becomes too quiet.
- Next stage target: reduce rollout action-density drift on healthy midgame
  model states while preserving same-state target/ship behavior. The next data
  delta should collect a small DAgger cache gated by state/action discrepancy,
  not final outcome alone: roughly turns `100..250`, model still healthy
  (`owned_planets`/ships above a threshold or not catastrophically behind), and
  signed action gap `regular_action_count - model_action_count >= 2`, with caps
  by turn bucket and outcome. Train this as a small source/count correction
  first, keep target/ship/pair DAgger amplification off, and require both
  same-state gates and standard multi-seed online eval to beat the current best
  before promotion.

## 2026-05-27 healthy-midgame action-gap DAgger check

- Added health/action-gap filters to
  `tinyPPO/collect_failed_rollout_dagger_cache.py`:
  `--min-action-gap`, `--max-action-gap`, `--min-owned-planets`,
  `--min-owned-ships`, and `--min-score-gap`. Defaults preserve the old
  behavior.
- Collected targeted DAgger caches from the five requested checkpoints
  `e0400/e0600/e0900/e1100/e1300` using 256 two-player games each, 64 actors,
  turns `100..250`, `regular_action_count - model_action_count >= 2`,
  `owned_planets >= 8`, and `score_gap >= -2000`. Each cache had about
  `733..762` rows and `9092..9564` labelled actions. The selected rows were
  the intended high-gap states: regular averaged about `12.5..12.9` actions per
  row while the model averaged only `1.6..1.9`.
- Built the mixed cache
  `tinyPPO/data/regular_bc_dagger_mix_healthy_midgame_gap5ckpt_w2_45ka_20260527.pkl`.
  It has `3548` rows, `43958` raw labelled actions, and `87916` weighted
  labelled actions with per-row weight `2.0`, about 25% of the original
  regular cache action mass (`352925` actions). Bucket caps kept the mix focused
  on `win/loss` rows in turns `100..250`, with most weight in healthy winning
  midgame states.
- Trained the single-delta source/count correction run
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_sourcehead_e1100to1200_eval4pg_20260527`.
  It resumed the current best, used the new healthy-midgame cache, kept
  `target/ship/pair` DAgger amplification off, and trained only
  `source_head`. Runtime confirmed DAgger weighting:
  `dagger_unweighted_labelled_actions=43958`,
  `dagger_weighted_labelled_actions=87916`, and DAgger row weight `2.0`.
- Same-state validation stayed stable by construction:
  e1200 `val_target_pair_acc=0.59917`, `val_target_acc=0.96725`,
  `val_ship_acc=0.92072`. Best loader imitation score during this continuation
  was `0.61897` at e1190.
- Standard online eval still failed. E1200 versus regular was
  `5W/251L/0D`, nonloss/winrate `0.01953125`, mean reward `-0.9609375`, below
  the current best `17W/239L/0D`.
- Follow-up rollout behavior diagnosis:
  `tinyPPO/runs/rollout_behavior_current_vs_healthy_gap_sourcehead_128g_20260527.json`.
  On the diagnostic seed set, current best was `95W/33L`, and the healthy-gap
  source-head checkpoint was `96W/32L`; this seed set remains behavior-only and
  is not a promotion gate. The action-density gap improved only slightly
  overall (`2.65` to `2.52`) and remained large in the exact target region:
  turn `100-149` got worse (`4.33` to `4.44` regular-minus-model actions),
  turn `150-199` improved (`4.60` to `3.90`), and turn `200-249` worsened
  (`3.77` to `4.62`). In winning trajectories, the gap only moved from `4.20`
  to `4.02`.
- Decision: do not promote. The new targeted data is correctly aimed at the
  rollout drift, but source-head-only training cannot absorb it enough. The
  next single delta should be about allowing the regular labels to affect the
  coupled action decision, not collecting more of the same data. Candidate next
  checks: unfreeze target/ship/pair with a small DAgger component scale instead
  of zero, or train all heads from the same healthy-midgame cache with a lower
  total DAgger action mass. Keep the same standard online gate; do not proceed
  to PPO/self-play yet.

## 2026-05-27 low-scale all-heads healthy-midgame DAgger early stop

- Tested the next single delta in
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_allheads_tsp025_e1100to1200_eval4pg_20260527`.
  This used the same healthy-midgame action-gap cache and same resume point as
  the source-head run, but trained all heads. DAgger amplification stayed full
  for launch/count and was reduced to `0.25` scale for target/ship/pair, so the
  observed validation sample weights were about `1.021` for launch/count and
  `1.005` for target/ship/pair.
- The run was stopped early around e1115. Even that small target/ship/pair
  amplification caused same-state target-pair accuracy to drop from the stable
  source-head/current-best region around `0.599` to about `0.561..0.563`
  (`val_target_acc` around `0.962..0.963`, `val_ship_acc` around
  `0.915..0.918`). That is the same failure shape as earlier all-outcome
  all-heads DAgger runs.
- No online promotion eval was run, because the same-state gate had already
  failed. Decision: do not promote, and do not spend more GPU on this exact
  setting.
- Updated stage conclusion: the healthy-midgame rows are aimed at the right
  rollout states, but directly amplifying target/ship/pair on those rows is too
  easy to destabilize. The next conservative check, if continuing this line, is
  to keep the same cache and all-heads training but set target/ship/pair DAgger
  amplification to `0.0`, so only launch/count get the row boost while the
  other heads see those added states at ordinary weight. Stop immediately if
  `val_target_pair_acc` falls below roughly `0.59`.

## 2026-05-27 all-heads healthy-midgame DAgger without target/pair amplification

- Ran the conservative follow-up
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_allheads_tsp0_e1100to1200_eval4pg_20260527`.
  It kept the same healthy-midgame DAgger cache and all-heads training, but set
  `--sample-weight-target-scale 0.0`, `--sample-weight-ship-scale 0.0`, and
  `--sample-weight-pair-scale 0.0`. Runtime metrics confirmed that only
  launch/count were upweighted: validation launch/count sample weight was about
  `1.021`, while target/ship/pair sample weight was exactly `1.0`.
- This still failed the early same-state gate. The run was stopped at e1104
  before online eval. Validation target-pair accuracy stayed in the bad range:
  e1101 `0.56314`, e1102 `0.56822`, e1103 `0.56526`, e1104 `0.56209`.
  These are far below the stable source-head/current-best region around
  `0.599`.
- Decision: do not promote and do not continue all-heads training on this cache.
  The failure now appears to come from allowing target/ship/pair parameters to
  adapt on the model-rollout state distribution at all, not just from giving
  those components extra sample weight.
- Updated next-step constraint: target/ship/pair should stay frozen for this
  DAgger phase. Further progress needs either a better source/count-only
  rollout objective, a cleaner source/count target derived from regular action
  density, or an online selection/bridging check that changes only which
  regular-like sources are launched while preserving regular target/ship logic.

## 2026-05-27 source-head healthy-midgame DAgger continuation

- Ran a source/count-only continuation in
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_sourcehead_e1200to1600_eval4pg_20260527`,
  resuming from the stable source-head e1200 checkpoint and keeping the same
  healthy-midgame DAgger cache, row weights, eval cadence, and frozen
  target/ship/pair heads.
- Same-state Phase1 gates stayed stable. From e1201 through e1400,
  validation target-pair accuracy stayed fixed around `0.59917`,
  `val_target_acc` stayed around `0.96725`, and `val_ship_acc` around
  `0.9207..0.9208`. This confirms that source-head-only DAgger is safe for the
  frozen target/ship/pair behavior.
- Extra offline training did not produce the needed online improvement. E1300
  online eval versus regular was `13W/243L/0D`, nonloss/winrate `0.05078125`,
  mean reward `-0.8984375`. This is still below the current best
  `17W/239L/0D` (`0.06640625` nonloss) and far below the next-stage target of
  stable nonloss above `0.15`.
- The continuation was stopped after the e1300 eval result instead of spending
  the rest of e1400..e1600. Decision: do not promote.
- Updated conclusion: the current healthy-midgame DAgger cache plus
  source-head-only BC can preserve Phase1 but does not meaningfully reduce
  rollout drift. All-heads variants reduce same-state target/pair quality.
  The next useful delta should therefore be behavioral rather than "more of the
  same": diagnose which regular source launches are missing in healthy midgame,
  train or bridge only that source-selection decision, and keep regular
  target/ship assignment intact until source density improves online.

## 2026-05-27 healthy-midgame source-gap diagnosis

- Added `tinyPPO/diagnose_dagger_source_gaps.py` to inspect DAgger cache rows
  without rerunning rollouts. It reads a regular-labelled DAgger cache, runs a
  checkpoint forward on the encoded model-rollout states, and reports
  source-level precision/recall plus grouped profiles for regular sources that
  the model scores below a launch threshold.
- Ran it on
  `tinyPPO/data/regular_bc_dagger_mix_healthy_midgame_gap5ckpt_w2_45ka_20260527.pkl`
  with the current best checkpoint. Report:
  `tinyPPO/runs/dagger_source_gap_current_best_healthy_midgame_20260527.json`.
  At threshold `0.5`, current best has source recall `0.7901`, precision
  `0.8641`, and `70.0%` of DAgger rows still miss at least one regular source.
  Missed regular sources are not marginal threshold cases: their mean source
  probability is only `0.272`, with mean ships about `75.7`, production `2.53`,
  incoming enemy `6.45`, and score gap still positive (`2050`). The largest
  missed groups are normal healthy winning midgame actions, especially enemy
  target launches (`5086` missed), friendly support/redistribution (`2642`
  missed), and neutral expansion (`1498` missed).
- The source-head e1300 continuation improves this offline source recall but
  does not solve online play. Report:
  `tinyPPO/runs/dagger_source_gap_sourcehead_e1300_healthy_midgame_20260527.json`.
  At threshold `0.5`, source recall improves to `0.8793`, but precision drops
  to `0.8261` and `56.0%` of rows still miss at least one regular source.
  This matches the online result: source probabilities were pushed up on the
  cache, but the full standalone policy still did not improve beyond the
  current best.
- A threshold-only diagnostic for current best at `0.25` shows recall can be
  forced to `0.9095`, but precision falls to `0.7955` and `44.8%` of rows still
  miss at least one source. Report:
  `tinyPPO/runs/dagger_source_gap_current_best_thr025_healthy_midgame_20260527.json`.
  So the problem is not just a single global launch threshold.
- Ran a small source-only bridge rollout check using the source-head e1300
  checkpoint while preserving regular target/ship actions:
  `tinyPPO/runs/eval_bridge_sourcehead_e1300_64g_cpu_20260527.log`.
  On the same 64-game seed set, pure regular anchor was `31W/33L/0D`.
  A conservative bridge with `threshold=0.25` and `max_source_drops=1` was
  `34W/30L/0D`, keeping `93.3%` of regular actions and attempting on `12.0%`
  of calls. Less conservative variants failed: `threshold=0.5,max_drop=1`
  was `21W/43L`, and unlimited drops kept only `53..60%` of actions and
  collapsed to `17..19W`.
- Updated next-step recommendation: do not spend more GPU on all-heads DAgger
  or longer source-head BC with this cache. The useful source signal is a
  conservative edit signal over regular actions, not an unconstrained action
  generator yet. The next single delta should train or evaluate a constrained
  source gate that can make at most one regular-action source edit per decision,
  with regular target/ship assignment frozen. Stop if it drops below regular
  anchor on same-seed bridge eval or if standalone Phase1 target/pair gates
  degrade.

## 2026-05-27 constrained source-drop BC gate

- Added `tinyPPO/train_bridge_drop_bc.py`. This is an offline BC/distillation
  path, not PPO. It loads a model-rollout DAgger cache, builds examples from
  regular-labelled actions, and trains a categorical gate over: keep all regular
  actions, or drop exactly one regular action. Target and ship assignment remain
  the regular rulebase's output at runtime.
- The teacher is the conservative source-prob bridge that had rollout signs of
  life: for rows with at least `--min-anchor-actions-to-filter` regular actions,
  compute source probabilities from the init checkpoint, drop the lowest-prob
  regular action only if it is below `--teacher-threshold`, otherwise keep all.
  This keeps the next delta focused on the source edit signal and avoids
  changing target/ship/pair heads.
- Extended `tinyPPO/eval_bridge.py` and `tinyPPO/eval_bridge_ray.py` so the
  trained `RegularSourceDropGateAgent` can be evaluated alongside the regular
  anchor with parallel Ray workers. This addresses the previous slow single
  process validation path.
- Smoke command passed:

```bash
.venv/bin/python -m tinyPPO.train_bridge_drop_bc \
  --dagger-cache tinyPPO/data/regular_bc_dagger_mix_healthy_midgame_gap5ckpt_w2_45ka_20260527.pkl \
  --init-checkpoint tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_sourcehead_e1200to1600_eval4pg_20260527/regular_bc_ray_e1300.pt \
  --out tinyPPO/runs/bridge_drop_bc_smoke_20260527/drop_gate_smoke.pt \
  --device cpu --teacher-threshold 0.25 --no-drop-bias 2.0 \
  --max-rows 128 --epochs 1 --batch-size 32
```

- Smoke build metrics: `108` gate examples from `128` rows, teacher drop rate
  `0.2407`. A 4-game/80-step Ray eval smoke also ran successfully with
  `--include-drop-gate`. The smoke is only a wiring check; it is not a promoted
  result.
- Fixed `tinyPPO/eval_bridge_ray.py` so variant evals default to the same seed
  set (`--variant-seed-stride 0`) and stream one completed part at a time. The
  old Ray evaluator offset seeds by variant, which was not a fair same-seed
  bridge comparison.
- Trained a full-cache source-head-only gate:
  `tinyPPO/runs/bridge_drop_bc_healthy_midgame_sourcehead_e1300_thr025_20260527/drop_gate.pt`.
  It used the source-head e1300 checkpoint as initialization, teacher threshold
  `0.25`, no-drop bias `2.0`, and the healthy-midgame 5-ckpt cache. Build
  metrics: `2969` examples from `3548` rows, teacher drop rate `0.2984`.
  Final validation: loss `0.7867`, accuracy `0.7910`, predicted drop rate
  `0.4427` versus teacher drop rate `0.3146`.
- Same-seed 16-game deterministic Ray eval:
  `tinyPPO/runs/bridge_drop_bc_healthy_midgame_sourcehead_e1300_thr025_20260527/eval_drop_gate_16g_same_seed_bias234.log`.
  Regular anchor was `6W/10L/0D`. Drop gate with bias `2.0` was also
  `6W/10L/0D`, keeping `97.4%` of regular actions; bias `3.0` was also
  `6W/10L/0D`, keeping `98.9%`; bias `4.0` was worse at `4W/12L/0D`.
  This version is safe-ish at low bias but shows no online improvement, so do
  not promote it.
- Updated conclusion: the conservative edit interface is useful for isolating
  source decisions, but this particular offline threshold-distilled drop gate
  does not yet reduce rollout drift. The next aligned delta should train the
  gate against an outcome-aware or counterfactual source label, or improve the
  DAgger source target, rather than continuing this exact threshold-distillation
  run longer.

## 2026-05-27 standalone decode sweep for source-head DAgger

- Added `tinyPPO/eval_policy_sweep_ray.py`, a small Ray evaluator for
  same-seed online decode sweeps. It evaluates `TinyPPOAgent` against regular
  while sweeping launch bias, launch temperature, target-pair weight, target
  mask mode, and ship bias. This is for cheap diagnosis only; it is not a new
  training objective.
- Ran a first source-head e1300 sweep on the hard seed band `950000`:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_sourcehead_e1200to1600_eval4pg_20260527/eval_decode_sweep_e1300_16g_20260527.log`.
  The `target_pair_weight=0.0` configurations at launch bias `-0.5` and
  `-0.25` were both `0W/16L/0D`, so this checkpoint still needs the pair head
  and does not benefit from removing it.
- Reran a focused positive launch-bias sweep with `target_pair_weight=1.0`:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_sourcehead_e1200to1600_eval4pg_20260527/eval_decode_sweep_e1300_pair1_posbias_16g_20260527.log`.
  Results on the same 16-game seed set:
  - `launch_bias=0.0`: `0W/16L/0D`.
  - `launch_bias=0.25`: `1W/15L/0D`.
  - `launch_bias=0.5`: `1W/15L/0D`.
  - `launch_bias=0.75`: `0W/16L/0D`.
  - `launch_bias=1.0`: `0W/16L/0D`.
- Conclusion: the source-head DAgger failure is not just a missed launch-bias
  calibration. Small positive bias gives only a one-game flicker on this hard
  band and larger bias makes it worse. This reinforces the previous conclusion:
  the model-rollout DAgger rows are exposing real source/target/ship decision
  drift, not merely a global under-launch threshold issue.

## 2026-05-27 DAgger full-action diagnosis and target-pair-only delta

- Added `tinyPPO/diagnose_dagger_action_imitation.py`. It reads regular-labelled
  DAgger `BCRow` caches and compares a checkpoint's decoded source,
  source-target, and source-target-ship bucket decisions directly on those
  model-rollout states. It reports both macro row F1 and micro F1, plus grouped
  metrics by DAgger outcome / turn bucket / source checkpoint.
- On the healthy-midgame cache with the dataset target mask, source-head e1300
  looks better than the current best:
  - Current best:
    `source_f1=0.7209`, `source_target_f1=0.6997`, `action_f1=0.6194`,
    `micro_action_f1=0.7679`, model/regular actions `10.35/12.39`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_current_best_healthy_midgame_20260527.json`.
  - Source-head e1300:
    `source_f1=0.7974`, `source_target_f1=0.7739`, `action_f1=0.6810`,
    `micro_action_f1=0.8018`, model/regular actions `11.75/12.39`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_sourcehead_e1300_healthy_midgame_20260527.json`.
- The same diagnosis under an all-planets target mask exposes the real target
  ranking problem:
  - Current best all-planets:
    `source_target_f1=0.2848`, `action_f1=0.2523`,
    `micro_source_target_f1=0.3339`, `micro_action_f1=0.3060`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_current_best_healthy_midgame_allplanets_20260527.json`.
  - Source-head e1300 all-planets:
    `source_target_f1=0.3061`, `action_f1=0.2686`,
    `micro_source_target_f1=0.3370`, `micro_action_f1=0.3078`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_sourcehead_e1300_healthy_midgame_allplanets_20260527.json`.
- Interpretation: source-head-only DAgger genuinely improves source density and
  cache action matching when the label target is included in the supervised
  dataset mask, but it barely improves global target ranking. That explains why
  cache metrics and online play disagree. The next single delta should target
  global source-target ranking on DAgger states without touching the main
  source/target/ship heads.
- Added `target_pair_head` to `--trainable-modules` in both
  `tinyPPO/imitation_regular.py` and `tinyPPO/imitation_regular_ray.py`. This
  freezes the whole model except `model.target_pair_head`.
- Smoke-tested the new trainable module by resuming source-head e1300, using a
  tiny regular sample plus `64` healthy-midgame DAgger rows, and running one
  epoch through Ray. The command saved
  `tinyPPO/runs/target_pair_head_dagger_smoke_20260527/regular_bc_ray.pt` and
  completed at epoch `1301`, verifying that target-pair-only BC trains and
  checkpoints correctly.
- Added the next real-run script:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_targetpair_e1300to1400_eval4pg_20260527/run.sh`.
  It resumes source-head e1300, mixes the original 10k regular cache with the
  healthy-midgame DAgger cache, trains only `target_pair_head`, keeps DAgger
  amplification only on the pair loss, and evaluates every 100 epochs. Stop if
  all-planets DAgger target ranking does not improve, if Phase1 same-state
  target/pair gates fall, or if online nonloss remains below the current best.

## 2026-05-27 target-pair-only DAgger result

- Ran the target-pair-only DAgger delta through epoch `1400`:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_targetpair_e1300to1400_eval4pg_20260527/regular_bc_ray_e1400.pt`.
- Training behaved as intended for a frozen-backbone pair-head-only run:
  launch/count/target/ship metrics stayed effectively fixed while
  `val_target_pair_softmax_loss` moved only slightly, from about `1.2041` at
  epoch `1301` to `1.2017` at epoch `1400`. `val_target_pair_acc` remained near
  `0.599`, so the offline pair-head signal did not translate into a clear
  accuracy gain.
- Online evaluation against regular at epoch `1400` was worse than both the
  current best and source-head e1300: `7W/249L/0D` over `256` games,
  `nonloss=0.0273`.
- Same-state DAgger action diagnosis confirms that this did not fix global
  target ranking:
  - Dataset target mask:
    `source_f1=0.7974`, `source_target_f1=0.7739`, `action_f1=0.6812`,
    `micro_action_f1=0.8018`, model/regular actions `11.75/12.39`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_targetpair_e1400_healthy_midgame_20260527.json`.
  - All-planets target mask:
    `source_target_f1=0.3068`, `action_f1=0.2692`,
    `micro_source_target_f1=0.3385`, `micro_action_f1=0.3091`,
    model/regular actions `13.19/12.39`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_targetpair_e1400_healthy_midgame_allplanets_20260527.json`.
- Decision: do not promote this branch. The source-head DAgger checkpoint
  exposed the right failure mode, but training only the target-pair head is too
  weak to repair it. The next delta should broaden the target-ranking update
  enough to affect shared source-target representations while preserving the
  BC baseline gates and using same-state all-planets DAgger imitation as the
  early stop signal before spending more online eval.

## 2026-05-27 target-ranking and pair-adapter follow-up

- Added `target_ranking` to `--trainable-modules` in both BC trainers. It
  unfreezes `model.edge` plus `model.target_pair_head`, leaving the other heads
  frozen. A smoke test passed, so the path was runnable.
- Started the real `target_ranking` run:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_targetranking_e1300to1400_eval4pg_20260527/run.sh`.
  It was stopped manually at epoch `1308` before online eval because the
  same-state source/launch gate degraded immediately. `val_launch_pred_rate`
  jumped to about `0.116-0.120` against regular's `0.063`, and
  `val_action_count_mae` rose to about `2.4`. This happened because
  `source_target_summary` feeds the shared edge representation into the source
  head, so updating `edge` for target ranking also perturbs source decisions.
  Do not promote this branch or continue it.
- Added an isolated `target_pair_adapter` architecture option. When
  `--target-pair-adapter` is set, target-pair logits use a separate
  `target_pair_edge` MLP. `--resume-compatible` initializes this adapter by
  copying the old `edge` tensors, so pair-ranking can move without changing the
  source/target/ship heads. Added `target_pair_adapter` as a trainable module
  that unfreezes only `target_pair_edge` plus `target_pair_head`.
- Smoke-tested adapter resume/training:
  `tinyPPO/runs/target_pair_adapter_dagger_smoke_20260527/regular_bc_ray.pt`.
  The log confirmed `target_pair_edge.*` tensors were copied from `edge.*`.
- Ran a short full-data adapter stage without online eval:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_e1300_e20_noeval_20260527/regular_bc_ray_e0020.pt`.
  This preserved the source/launch gate: `val_launch_pred_rate=0.0721` versus
  regular `0.0632`, and `val_action_count_mae=1.516`, matching source-head
  e1300 rather than the failed shared-edge run. Pair loss moved from about
  `1.2191` to `1.2125`.
- However, same-state DAgger action diagnosis did not improve global target
  ranking, so no online eval was run:
  - Dataset target mask:
    `source_f1=0.7974`, `source_target_f1=0.7738`, `action_f1=0.6811`,
    `micro_action_f1=0.8020`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_pairadapter_e0020_healthy_midgame_20260527.json`.
  - All-planets target mask:
    `source_target_f1=0.3054`, `action_f1=0.2682`,
    `micro_source_target_f1=0.3367`, `micro_action_f1=0.3075`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_pairadapter_e0020_healthy_midgame_allplanets_20260527.json`.
- Decision: the adapter is useful infrastructure because it lets target-ranking
  training avoid source drift, but this 20-epoch DAgger pair-softmax objective
  still does not improve the all-planets target-ranking metric. The next
  useful delta should change the target-ranking supervision itself, not spend
  more online eval on this adapter checkpoint.

## 2026-05-27 DAgger target-rank diagnosis

- Added `tinyPPO/diagnose_dagger_target_ranks.py`, a cheap same-state DAgger
  diagnostic that ranks the regular-labelled target under a checkpoint's
  all-planets target logits. It reports top-k coverage, mean/percentile rank,
  MRR, owner buckets, turn buckets, and the top-1 margin over the regular label.
- Ran it on the healthy-midgame five-checkpoint DAgger mix:
  `tinyPPO/data/regular_bc_dagger_mix_healthy_midgame_gap5ckpt_w2_45ka_20260527.pkl`.
- Source-head e1300 all-planets ranks:
  `top1=0.3825`, `top3=0.7059`, `top5=0.8440`, `top10=0.9550`,
  `mean_rank=3.2620`, `median_rank=2`, `p90_rank=7`, `mrr=0.5738`,
  mean top-1-over-label margin `2.0075`.
  Report:
  `tinyPPO/runs/dagger_target_ranks_sourcehead_e1300_healthy_midgame_allplanets_20260527.json`.
- Pair-adapter e0020 all-planets ranks:
  `top1=0.3825`, `top3=0.7088`, `top5=0.8472`, `top10=0.9567`,
  `mean_rank=3.2322`, `median_rank=2`, `p90_rank=7`, `mrr=0.5747`,
  mean top-1-over-label margin `1.8820`.
  Report:
  `tinyPPO/runs/dagger_target_ranks_pairadapter_e0020_healthy_midgame_allplanets_20260527.json`.
- Interpretation: the regular target is usually near the top of the all-planets
  ranking but is top-1 only about 38% of the time. The adapter slightly reduces
  margin/rank but does not change the decision. This supports changing the
  target-ranking supervision to an all-planets hard-negative or margin/top-k
  objective, while keeping the adapter isolation to protect source/launch gates.
  It does not support more online eval or just extending the current pair-softmax
  run.
- Added `--target-pair-margin-loss-weight` to both BC trainers. This applies a
  hard-negative margin loss directly to `target_pair_logits` using the same
  active regular labels and all valid source-target pairs. It is separate from
  the older `target_margin_loss`, which acts on the slot target head and does
  not train the isolated adapter path.
- Added the next runnable single-delta script:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_margin_e1300_e20_noeval_20260527/run.sh`.
  It resumes source-head e1300 with `--target-pair-adapter`, trains only the
  adapter plus pair head, uses `target_pair_softmax_loss_weight=0.5` and
  `target_pair_margin_loss_weight=1.0`, and keeps online eval disabled until
  all-planets same-state target ranks improve.
- Smoke-tested the Ray path with 1 regular game plus 64 DAgger rows:
  `tinyPPO/runs/target_pair_adapter_margin_dagger_smoke_20260527/regular_bc_ray.pt`.
  The smoke completed, copied adapter tensors from `edge.*`, logged nonzero
  `train_target_pair_margin_loss=0.4081` and
  `val_target_pair_margin_loss=0.6222`, and saved checkpoints. The tiny-smoke
  launch metrics are not meaningful; this only verifies the training path.
- Ran the full 20-epoch adapter+margin stage:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_margin_e1300_e20_noeval_20260527/regular_bc_ray_e0020.pt`.
  SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/mj4pnqtm5ayldot9wrt1v`.
  Source/launch gate stayed stable (`val_launch_pred_rate=0.0721` versus
  regular `0.0632`, `val_action_count_mae=1.516`), and the margin objective
  decreased (`val_target_pair_margin_loss` about `0.3456 -> 0.2991`), but the
  actual all-planets target ranking got worse.
- Full DAgger all-planets rank/action diagnosis:
  - e0020 rank:
    `top1=0.3431`, `top3=0.6624`, `top5=0.8092`, `top10=0.9426`,
    `mean_rank=3.6061`, `mrr=0.5379`.
    Report:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_margin_e0020_healthy_midgame_allplanets_20260527.json`.
  - e0020 action imitation:
    `source_target_f1=0.2736`, `action_f1=0.2407`,
    `micro_source_target_f1=0.3023`, `micro_action_f1=0.2775`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_pairadapter_margin_e0020_healthy_midgame_allplanets_20260527.json`.
  - best-imitation checkpoint was also worse than the non-margin adapter:
    rank `top1=0.3606`, `top5=0.8230`, `mean_rank=3.4639`; action
    `source_target_f1=0.2889`, `action_f1=0.2535`.
    Reports:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_margin_best_healthy_midgame_allplanets_20260527.json`,
    `tinyPPO/runs/dagger_action_imitation_pairadapter_margin_best_healthy_midgame_allplanets_20260527.json`.
- Decision: do not promote or online-eval the margin branch. It optimizes the
  local margin number while making the global all-planets ranking worse, partly
  by skewing top-1 targets toward own planets. The next delta should not be a
  stronger generic margin; it should add structure to target supervision, for
  example owner-conditioned negatives / target-owner calibration, or a residual
  adapter constrained not to move the old ranking unless the regular target is
  near-top-k and owner-compatible.

## 2026-05-27 target-owner calibration setup

- After the failed generic-margin branch, the next hypothesis is that target
  ranking needs to separate strategic target type from within-type planet
  ranking. The diagnostic evidence is owner-distribution skew: regular-labelled
  actions in the DAgger cache are mostly enemy/neutral/own
  `25893/6806/11259`, while the failed margin branch's model top-1 targets
  were skewed toward own planets `16432/2027/25499`.
- Added `--target-pair-owner-loss-weight` to both BC trainers. For each active
  regular action, it aggregates `target_pair_logits` by target owner
  (`own/neutral/enemy`) using logsumexp over valid targets for the same source,
  then applies a weighted three-class CE against the regular target's owner.
  This is intentionally weaker than planet-level margin: it should correct the
  target-type distribution without directly forcing a particular planet to top-1.
- Smoke-tested the Ray path with 1 regular game plus 64 DAgger rows:
  `tinyPPO/runs/target_pair_adapter_owner_dagger_smoke_20260527/regular_bc_ray.pt`.
  The smoke completed with nonzero owner metrics:
  `train_target_pair_owner_loss=0.5340`,
  `train_target_pair_owner_acc=0.7496`,
  `val_target_pair_owner_loss=0.6479`,
  `val_target_pair_owner_acc=0.7015`.
- Added the next short-run script but did not start the full run yet:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_owner_e1300_e20_noeval_20260527/run.sh`.
  It resumes source-head e1300, uses `--target-pair-adapter`, trains only the
  adapter plus pair head, combines pair-softmax `0.5` with target-owner CE `1.0`,
  keeps source/target/ship row-weight amplification off, and disables online eval.
- Decision gate before spending online eval: compare the resulting checkpoint
  against source-head e1300 and non-owner pairadapter e0020 on the same DAgger
  all-planets diagnostics. It must preserve source/launch gate and improve
  target-owner distribution plus `target_top1/top3/action_f1`. If owner acc
  improves but planet-level ranking does not, the next step should be a
  within-owner ranking head/loss rather than another global loss.
- Ran the full owner-calibration 20-epoch adapter stage:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_owner_e1300_e20_noeval_20260527/regular_bc_ray_e0020.pt`.
  SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/m3zfbvzkrdex7zmqfgtw6`.
  Source/launch gate stayed stable (`val_launch_pred_rate=0.0721` versus
  regular `0.0632`, `val_action_count_mae=1.516`). Loader owner metrics moved
  only slightly (`val_target_pair_owner_acc` around `0.798 -> 0.802`, with
  `val_target_pair_owner_loss` about `0.454 -> 0.448`).
- Full DAgger all-planets diagnostics show no real improvement over the
  non-owner pairadapter:
  - e0020 rank:
    `top1=0.3799`, `top3=0.7056`, `top5=0.8447`, `top10=0.9561`,
    `mean_rank=3.2547`, `mrr=0.5724`.
    Report:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_owner_e0020_healthy_midgame_allplanets_20260527.json`.
  - e0020 action imitation:
    `source_target_f1=0.3045`, `action_f1=0.2674`,
    `micro_source_target_f1=0.3344`, `micro_action_f1=0.3054`.
    Report:
    `tinyPPO/runs/dagger_action_imitation_pairadapter_owner_e0020_healthy_midgame_allplanets_20260527.json`.
  - best-imitation checkpoint was slightly better than e0020 but still not a
    real improvement over sourcehead/pairadapter baselines:
    rank `top1=0.3828`, `top5=0.8457`, `mean_rank=3.2499`; action
    `source_target_f1=0.3054`, `action_f1=0.2679`,
    `micro_action_f1=0.3078`.
    Reports:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_owner_best_healthy_midgame_allplanets_20260527.json`,
    `tinyPPO/runs/dagger_action_imitation_pairadapter_owner_best_healthy_midgame_allplanets_20260527.json`.
- Decision: do not promote or online-eval the owner-calibration branch. It is
  safer than generic margin and does not harm source/launch gates, but by
  itself it does not improve planet-level target choice. The next useful delta
  should use the owner signal as structure, not as the only objective: train
  within-owner target ranking, or construct hard-case DAgger rows where regular
  chooses enemy/neutral and the model top-1 chooses own.

## 2026-05-27 target-owner conditioned decoder setup

- Model-structure conclusion: do not enlarge or replace the whole policy yet.
  The source/launch path is reasonably fragile but usable; the persistent
  blocker is the target decoder. The next single architecture delta is therefore
  an optional target-owner-conditioned bias on `target_pair_logits`, behind
  `--target-pair-owner-head`.
- Implementation detail: `TinyPolicyValueNet` now supports
  `target_pair_owner_head`, a zero-initialized `source_context -> 3 owner
  logits` head. For each source-target edge it gathers the bias corresponding
  to the target planet's owner and adds it to the existing target-pair score.
  Because it is zero-initialized, enabling it with `--resume-compatible` leaves
  the old checkpoint's initial logits unchanged.
- The owner-conditioned head is included in `target_pair_head`,
  `target_ranking`, and `target_pair_adapter` trainable-module modes when
  present. This keeps the experiment isolated to the target-pair branch instead
  of perturbing source/launch/ship heads.
- Smoke checks passed:
  - `py_compile` for `tinyPPO/model.py`, `tinyPPO/imitation_regular.py`, and
    `tinyPPO/imitation_regular_ray.py`.
  - A direct forward comparison between old pairadapter config and
    owner-head-enabled config after copying matching weights had
    `max_abs_diff=0.0` on finite `target_pair_logits`, confirming zero-init
    compatibility.
  - Ray smoke:
    `tinyPPO/runs/target_pair_adapter_ownerhead_dagger_smoke_20260527/regular_bc_ray.pt`.
    It resumed sourcehead e1300 compatibly, loaded 41 tensors, copied adapter
    tensors from `edge.*`, and logged nonzero
    `train_target_pair_owner_loss=0.5214`,
    `val_target_pair_owner_loss=0.6010`.
- Added the runnable short-stage script:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_ownerhead_e1300_e20_noeval_20260527/run.sh`.
  It resumes sourcehead e1300, enables `--target-pair-adapter` plus
  `--target-pair-owner-head`, trains only `target_pair_adapter`, keeps
  pair-softmax `0.5` and owner CE `1.0`, and disables online eval.
- Stop/gate before any online eval: compare the resulting checkpoint against
  sourcehead e1300, pairadapter e0020, and owner-calibration best on the same
  DAgger all-planets rank/action diagnostics. It must preserve source/launch
  density and improve at least one real target-quality metric
  (`top1/top3/mean_rank/action_f1`) without worsening owner skew. If not, do
  not spend online-eval GPU on it; move to hardcase data or within-owner target
  ranking.

## 2026-05-27 owner-conditioned DAgger short-run result

- Ran the 20-epoch owner-conditioned adapter stage:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_ownerhead_e1300_e20_noeval_20260527/regular_bc_ray_e0020.pt`.
  SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/ps4b6s5u2l8spaojo4vat`.
  The run used the intended DAgger mixture: cached pure regular data
  (`163502` rows, `352925` labelled actions) plus the 5-checkpoint model-rollout
  DAgger cache (`3548` rows, `43958` labelled actions, weighted to `87916`).
- Loader source/launch metrics stayed stable, which means the isolated
  adapter/head path did not break the known gate:
  `val_launch_pred_rate=0.0721` versus true `0.0632`, and
  `val_action_count_mae=1.516`. However, loader pair/owner metrics barely moved
  after the first few epochs; final e0020 had
  `val_target_pair_acc=0.5907` and `val_target_pair_owner_acc=0.8006`.
- Full DAgger all-planets diagnosis:
  - e0020 was worse than the previous pairadapter/sourcehead baselines:
    rank `top1=0.3789`, `top3=0.7049`, `top5=0.8441`,
    `mean_rank=3.2596`, action `source_target_f1=0.3032`,
    `action_f1=0.2662`, `micro_action_f1=0.3044`.
    Reports:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_ownerhead_e0020_healthy_midgame_allplanets_20260527.json`,
    `tinyPPO/runs/dagger_action_imitation_pairadapter_ownerhead_e0020_healthy_midgame_allplanets_20260527.json`.
  - best-imitation was slightly better than sourcehead e1300 but only by a very
    small margin:
    rank `top1=0.3848`, `top3=0.7084`, `top5=0.8476`,
    `mean_rank=3.2324`, action `source_target_f1=0.3069`,
    `action_f1=0.2694`, `micro_action_f1=0.3097`.
    Reports:
    `tinyPPO/runs/dagger_target_ranks_pairadapter_ownerhead_best_healthy_midgame_allplanets_20260527.json`,
    `tinyPPO/runs/dagger_action_imitation_pairadapter_ownerhead_best_healthy_midgame_allplanets_20260527.json`.
- Same-state Phase1 gate was parallelized across six CPU processes instead of
  running one serial gate: 3 seeds for candidate and 3 seeds for sourcehead e1300.
  Aggregate result is in
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_ownerhead_e1300_e20_noeval_20260527/phase1_parallel/aggregate.json`.
  Candidate and baseline were effectively tied: candidate action F1 `0.2697`
  versus baseline `0.2687`, source-target F1 `0.2806` versus `0.2795`,
  density ratio `0.875` for both, and micro action F1 slightly worse
  (`0.2405` versus `0.2422`).
- Online sanity was run with 64 Ray CPU tasks per variant, one game per task,
  using fast env versus regular:
  - `launch_bias=-0.05`, all-planets, target-pair weight `1.0`:
    `1W/63L/0D`, nonloss `0.016`.
  - `launch_bias=0.15`, all-planets, target-pair weight `1.0`:
    `2W/62L/0D`, nonloss `0.031`.
- Decision: do not promote ownerhead and do not spend more online eval on this
  branch. It preserves same-state imitation but does not solve rollout
  distribution shift. The next aligned step is data-centric: expand or rebalance
  regular-labelled model-rollout states, especially hard cases where regular
  chooses enemy/neutral targets and the model ranks own targets first, rather
  than adding another small head or sweeping launch hyperparameters.

## 2026-05-27 hardcase DAgger cache for next data-centric stage

- Fixed `tinyPPO/build_dagger_hardcase_cache.py` so row-level reason counts are
  counted once per selected row and `selected_model_top1_owner_counts` is
  populated. The script compiles.
- Built a sourcehead-e1300 hardcase cache from the existing 5-checkpoint
  DAgger model-rollout cache:
  `tinyPPO/data/regular_bc_dagger_hardcase_sourcehead_e1300_rank3_margin0_20ka_20260527.pkl`.
  It selects rows where sourcehead e1300 misranks regular-labelled targets under
  all-planets target-pair logits, with `max_rank=3`, `min_margin=0.0`,
  `max_actions=20000`, and row weight `2.0`.
- Hardcase selection summary:
  - Input: `3548` rows, `43958` regular-labelled actions.
  - Candidate selected before cap: `3395` rows, `43470` actions.
  - Final capped cache: `1553` rows, `20000` actions.
  - Primary reasons: `enemy_neutral_to_own=2782`,
    `owner_mismatch=429`, `rank_gt_max=117`, `margin_gt_min=67`.
  - Selected regular target owners: enemy `11810`, neutral `3156`, own `5034`.
  - Selected model top-1 owners: enemy `8090`, neutral `1183`, own `10727`.
  - Source checkpoints remain balanced enough for this stage:
    e400/e600/e900/e1100/e1300 rows `312/326/313/299/303`.
- Next run should be a data-only delta against the current DAgger baseline: mix
  the original pure regular cache, the existing healthy-midgame DAgger cache,
  and this hardcase cache. Keep the architecture fixed; keep online eval off
  until all-planets DAgger rank/action metrics move by more than the tiny
  ownerhead change, and same-state Phase1 gate stays flat.

## 2026-05-27 hardcase data-only adapter result

- Added a guarded training fix in `tinyPPO/imitation_regular.py` and
  `tinyPPO/imitation_regular_ray.py`: when an adapter-only batch has no active
  supervised loss requiring gradients, count it as
  `train_skipped_no_grad_samples` and skip backward/step. This avoids crashing
  frozen-head adapter stages on batches without target-pair supervision while
  leaving normal batches unchanged.
- Ran the intended data-only delta:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_hardcase20ka_e1300_e20_noeval_20260527/regular_bc_ray_best_imitation.pt`.
  SwanLab:
  `https://swanlab.cn/@Solo/orbit-wars/runs/eto0cwqpys3ey3re6c80v`.
  The run resumed sourcehead e1300 with the existing target-pair adapter
  structure, did not enable ownerhead, and mixed pure regular data plus both
  DAgger caches. Effective DAgger actions were `87916 + 40000 = 127916`
  against `352925` pure-regular actions, about `26.6%` of the weighted action
  mix, which matches the intended 20-30% DAgger range.
- Loader metrics at e0020 stayed sane:
  `val_target_pair_acc=0.6000`, `val_launch_pred_rate=0.0737` versus true
  `0.0643`, and `val_action_count_mae=1.5163`. The no-grad skip metric was
  `0.0` on the successful full run.
- DAgger all-planets diagnostics on the original healthy-midgame rollout cache:
  - e0020 rank: `top1=0.3828`, `top3=0.7089`, `top5=0.8475`,
    `top10=0.9567`, `mean_rank=3.2289`, `mrr=0.5750`.
  - best-imitation rank: `top1=0.3854`, `top3=0.7086`, `top5=0.8476`,
    `top10=0.9560`, `mean_rank=3.2315`, `mrr=0.5765`.
  - e0020 action: `source_target_f1=0.3061`, `action_f1=0.2686`,
    `micro_source_target_f1=0.3369`, `micro_action_f1=0.3077`,
    `action_count_mae=2.6559`.
  - best-imitation action: `source_target_f1=0.3074`, `action_f1=0.2699`,
    `micro_source_target_f1=0.3396`, `micro_action_f1=0.3101`,
    `action_count_mae=2.6559`.
  This is a small same-state improvement over sourcehead e1300 and the previous
  pairadapter run, but it is still a very small movement, not a behavioral
  breakthrough.
- Same-state Phase1 gate was run in parallel. Aggregate:
  `tinyPPO/runs/regular_bc_dagger_healthy_midgame_gap5ckpt_w2_pairadapter_hardcase20ka_e1300_e20_noeval_20260527/phase1_parallel/aggregate.json`.
  Candidate best-imitation beat sourcehead e1300 slightly on same-state metrics:
  action F1 `0.2704` versus `0.2687`, source-target F1 `0.2813` versus
  `0.2795`, micro action F1 `0.2427` versus `0.2422`, selected score
  `0.2157` versus `0.2146`, with density ratio unchanged at `0.875`.
- Online eval used 64 Ray CPU tasks per variant, one game per task, fast env
  versus regular. Results:
  - `launch_bias=-0.05`: `3W/61L/0D`, nonloss `0.047`.
  - `launch_bias=0.0`: `0W/64L/0D`, nonloss `0.000`.
  - `launch_bias=0.15`: `0W/64L/0D`, nonloss `0.000`.
- Decision: do not promote this checkpoint as the next baseline. It is the best
  same-state adapter result so far, but online behavior is still far below the
  Phase1 acceptance target (`nonloss > 0.15`) and worse than a useful regular
  clone should be. The next step should stay data-centric but needs broader
  rollout coverage, not another tiny target-head tweak: generate more
  regular-labelled states from the model's own rollouts across more seeds,
  include early-collapse and midgame recovery states, and explicitly audit
  where source selection, launch amount, and target owner/rank diverge before
  another training spend.

## 2026-05-27 broad model-seat DAgger all-heads result

- Collected a broader model-seat DAgger cache from five earlier checkpoints
  (`e0400/e0600/e0900/e1100/e1300`) to label states caused by model rollouts
  with regular's action choices:
  `tinyPPO/data/regular_bc_dagger_broad5ckpt_modelseat_2p_5120g_rows8_s4_20260527.pkl`.
  It contains `5120` games, `40792` sampled rows, and `294941`
  regular-labelled actions. Checkpoint coverage was balanced enough
  (`7650`-`8300` rows per source checkpoint), and with
  `dagger_loss_weight=0.4` the effective weighted DAgger share was about `25%`
  against the original 10k regular cache.
- Trained an all-heads continuation from sourcehead `e1300` for 80 epochs:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_modelseat_w04_allheads_e1300_e80_noeval_20260527`.
  Checkpoints were saved at `e0020/e0040/e0060/e0080` plus
  `regular_bc_ray_best_imitation.pt`. Train loss kept decreasing, but validation
  imitation peaked earlier and was lower again by epoch 80, so this fixed-loader
  view shows mild overfit rather than a clean late improvement.
- Same-state Phase1 gate versus sourcehead `e1300` did not improve. Baseline
  summary: `action_f1=0.269`, `source_target_f1=0.279`,
  `source_f1=0.445`, `density_ratio=0.875`, `selected_score=0.215`.
  The broad all-heads `best_imitation` checkpoint reached only
  `action_f1=0.253`, `source_target_f1=0.265`, `source_f1=0.442`,
  `density_ratio=0.837`, `selected_score=0.198`; `e0080` was lower at
  `action_f1=0.244`, `source_target_f1=0.255`, `selected_score=0.193`.
  Both failed due to action/source-target F1 and negative score delta versus
  baseline.
- Online sanity also had no sign of life: for `best_imitation`, 64 games each
  at `launch_bias=-0.05`, `0.0`, and `0.15` all returned `0W/64L/0D`.
- Decision: do not continue this all-heads broad DAgger line by simply adding
  epochs. The larger rollout data did expose distribution-shift states, but
  all-heads training on the current architecture/loss moved same-state imitation
  backward. The next step should be diagnostic/modeling work around
  source-target ranking and action construction, not more training time on this
  exact recipe.

## 2026-05-27 broad hardcase target-pair adapter result

- Ran broad-cache diagnostics before the next delta. On the full broad
  model-seat cache, sourcehead `e1300` still has high regular-source recall
  (`0.885` at threshold `0.5`), so source selection is not the main failure.
  Target ranking is the clearer bottleneck: sourcehead `e1300` gets broad-cache
  target `top1=0.3710`, `top3=0.6725`, `top5=0.8044`,
  `mean_rank=3.7326`, and `mrr=0.5547`. The previous small hardcase-best is
  only slightly better (`top1=0.3717`, `top3=0.6746`, `mean_rank=3.7102`),
  while the broad all-heads run is worse (`top1=0.3678`,
  `mean_rank=3.7817`).
- Built a larger broad hardcase cache from sourcehead `e1300` target-rank
  mistakes:
  `tinyPPO/data/regular_bc_dagger_broad5ckpt_hardcase_sourcehead_e1300_rank3_margin0_60ka_w2_20260527.pkl`.
  It selected `7251` rows and exactly `60000` regular-labelled actions from
  the broad rollout cache, with row weight `2.0`. Mixed with the 10k regular
  cache, the weighted DAgger action share is about `120000 / (352925+120000)`,
  or `25.4%`, matching the intended 20-30% DAgger exposure.
- Trained only the target-pair adapter from sourcehead `e1300` for 80 epochs:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_hardcase60ka_pairadapter_e1300_e80_noeval_20260527`.
  This deliberately did not train source/launch/ship heads; their loader
  metrics stayed fixed. The pair loss fit the training side
  (`train_target_pair_softmax_loss` fell from about `1.270` to `1.221`), but
  validation pair accuracy did not improve and drifted down by epoch 80
  (`val_target_pair_acc` about `0.589`).
- External broad-cache diagnostics confirmed no improvement:
  - e0020 target rank: `top1=0.3653`, `top3=0.6664`,
    `top5=0.8008`, `mean_rank=3.7756`, `mrr=0.5497`.
  - e0080 target rank: `top1=0.3675`, `top3=0.6690`,
    `top5=0.8025`, `mean_rank=3.7576`, `mrr=0.5517`.
  - e0020 action: `action_f1=0.2563`, `source_target_f1=0.2776`,
    `micro_action_f1=0.2844`.
  - e0080 action: `action_f1=0.2567`, `source_target_f1=0.2780`,
    `micro_action_f1=0.2862`.
  These are below sourcehead `e1300` on the same broad cache
  (`action_f1=0.2576`, `source_target_f1=0.2792`,
  `micro_action_f1=0.2893`) and below the previous hardcase-best.
- Same-state Phase1 gate for e0080 versus sourcehead `e1300` was effectively
  flat, not a pass: candidate action F1 `0.2698` versus baseline `0.2687`,
  source-target F1 `0.2806` versus `0.2795`, selected score `0.2156` versus
  `0.2146`, and density ratio `0.875` for both. It failed configured Phase1
  thresholds due to action/source-target F1 below `0.30` and score delta only
  `+0.0011`, far below the required `+0.02`.
- Decision: do not run online eval and do not promote this checkpoint. This
  confirms that simply upweighting target-rank hardcases in the current
  pair-adapter path mostly overfits the hardcase subset and does not improve
  target ranking on the broader model-rollout distribution. The next aligned
  work should change the representation/objective for target construction, not
  only resample more hardcases.

## 2026-05-27 within-owner target-pair objective result

- Added a new optional loss, `--target-pair-within-owner-loss-weight`, to both
  BC trainers. For each active regular-labelled source-target pair, it masks the
  target-pair logits to planets with the same owner class as the regular target
  and applies weighted CE there. Default weight is `0.0`, so existing runs are
  unchanged. The intent was to test the previous hypothesis that target
  construction needs within-owner ranking structure, not only global target CE
  or owner calibration.
- Smoke-tested the Ray path on the broad DAgger smoke cache with a compatible
  sourcehead `e1300` resume. The first attempt caught an active-action
  broadcasting bug in the new mask; after fixing it, the smoke loaded all
  `41` compatible tensors, copied adapter tensors from `edge.*`, produced
  nonzero `train_target_pair_within_owner_loss`, and completed one epoch.
- Ran the full broad model-seat DAgger setup from sourcehead `e1300`, training
  only the target-pair adapter with pair softmax `0.25` plus within-owner CE
  `1.0`:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_withinowner_pairadapter_e1300_e80_noeval_20260527`.
  The run used the 10k pure regular cache plus the broad 5-checkpoint
  model-seat DAgger cache at `dagger_loss_weight=0.4` (`117976` weighted DAgger
  labelled actions versus `352925` pure-regular actions, about `25%` DAgger
  exposure). It used `64` trainers across manual GPU ids `0-7`; GPU utilization
  was near `99-100%`.
- Stopped the run after epoch 49, with checkpoints saved at `e0020` and
  `e0040`. This was not a shallow step-level stop: the large-cache run had
  already trained deeply enough to show the trend. Training-side pair/within
  owner losses kept decreasing, but validation pair metrics only oscillated:
  `val_target_pair_acc` stayed around `0.578-0.581`, and
  `val_target_pair_within_owner_acc` stayed around `0.720-0.721`.
- External 10k-row broad-cache target-rank diagnostics showed the new objective
  moved backward versus sourcehead `e1300` on the same sampled rows:
  - sourcehead `e1300`: `top1=0.3694`, `top3=0.6739`,
    `top5=0.8050`, `mean_rank=3.7427`, `mrr=0.5540`.
  - within-owner `e0020`: `top1=0.3647`, `top3=0.6690`,
    `top5=0.8027`, `mean_rank=3.7778`, `mrr=0.5500`.
  - within-owner `e0040`: `top1=0.3646`, `top3=0.6696`,
    `top5=0.8031`, `mean_rank=3.7751`, `mrr=0.5502`.
- Decision: do not promote, do not run online eval, and do not continue this
  exact objective to epoch 80. Within-owner CE can be optimized locally, but it
  does not improve the broad rollout target ranking that matters for reducing
  distribution shift. The next modeling step should move beyond additive losses
  on the same logits: inspect action construction failures at the row/action
  level and consider changing the decoder representation or candidate/action
  factorization itself, while keeping the current source/launch baseline
  protected.

## 2026-05-27 action-construction audit and data cleanup

- Added `diagnose_dagger_action_errors.py` to compare model-decoded
  source-target multisets against regular labels on regular-labelled DAgger
  rows. This is deliberately action-level: target-rank loss alone was not
  explaining why online play remained brittle.
- On the broad 5-checkpoint model-seat DAgger cache, sourcehead `e1300` with
  all-planet target decoding had weak action construction despite reasonable
  source overlap: `source_f1=0.6758`, `source_target_f1=0.2808`,
  `micro_source_target_f1=0.3161`, `regular_actions/state=7.12`,
  `model_actions/state=8.45`, and `rows_pred_gt_true=0.553`. It also produced
  many extra own-target actions, indicating unreliable unconstrained target
  logits.
- The hardcase pair-adapter best checkpoint was effectively unchanged on the
  same audit (`source_target_f1=0.2810`,
  `micro_source_target_f1=0.3168`), so the previous hardcase adapter did not
  solve action construction.
- Reusing the dataset/candidate target mask changed the picture sharply for
  sourcehead `e1300`: `source_f1=0.7137`, `source_target_f1=0.6579`,
  `micro_source_target_f1=0.8124`, `model_actions/state=7.69`, and extra
  own-target actions disappeared. Lowering launch bias from `-0.05` to `-0.20`
  under all-planet decoding only moved `micro_source_target_f1` to `0.3174`,
  so the main lever is the target candidate/safety mask, not launch threshold.
- Online candidate-mask evals still did not pass the DAgger-stage target:
  sourcehead `e1300` reached `6/64` wins versus regular
  (`nonloss=0.09375`), and the hardcase adapter reached `3/64`
  (`nonloss=0.046875`). This is better than the stable 1/63 regime but still
  below the desired multi-seed nonloss target of about `>0.15`. Candidate
  masking should remain the online/deployment default, but it is not sufficient
  by itself.
- Cleaned regenerated/failed intermediate data caches from `tinyPPO/data`,
  reducing it from about `157G` to `77G` and the whole project from about
  `252G` to `172G`. Kept the two important sources for the current DAgger
  phase: the 10k pure regular cache
  `regular_bc_rulefeat_2p_10000g_rows16_20260526.pkl` and the current broad
  model-seat regular-labelled cache
  `regular_bc_dagger_broad5ckpt_modelseat_2p_5120g_rows8_s4_20260527.pkl`.
- Next phase remains regular-labelled DAgger/offline expansion: roll out p0
  with model checkpoints (`e400/e600/e900/e1100/e1300`) against regular p1,
  record states from the model-induced distribution, relabel each p0
  observation with `regularp0`, then train BC on a 70-80% pure-regular /
  20-30% model-state regular-label mix. Do not use the model action as the
  label, and do not switch to PPO/self-play until the model can approach
  regular in online evals.

## 2026-05-27 broad DAgger source-head-only result

- Confirmed the existing broad cache is the intended DAgger form: model controls
  the p0/model seat during rollout, regular controls p1, and each model-seat
  observation is labelled by calling `regular` on the same p0 observation. The
  cache covers the requested training-stage checkpoints
  `e400/e600/e900/e1100/e1300`, has `5120` games, `40792` model-seat rows, and
  `294941` labelled regular actions.
- Ran a conservative single-delta training job:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_modelseat_w03_sourcehead_e1300_e160_noeval_20260527`.
  It used the 10k pure regular cache plus the broad DAgger cache with
  `dagger_loss_weight=0.3`, which gives about `88482 / (352925+88482) = 20.1%`
  weighted DAgger labelled-action mass. It trained only `source_head` from
  sourcehead `e1300` for the full `160` epochs using `64` trainers manually
  spread across GPUs `0-7`; GPU utilization was about `99-100%`.
- Training reached a plateau rather than a late improvement. Best validation
  imitation score was `0.6449`; final epoch had
  `val_imitation_score=0.6424`, `val_launch_f1=0.6704`, and
  `val_action_count_mae=1.6192`.
- Same-state Phase1 gate versus sourcehead `e1300` did not pass:
  - all-planets decoding: candidate `selected_score=0.1765` versus baseline
    `0.1805`, `density_ratio=0.703`, `action_f1=0.263`,
    `source_target_f1=0.274`; failed score delta and F1 thresholds.
  - candidate-mask decoding: candidate `selected_score=0.1280` versus baseline
    `0.1295`, `density_ratio=0.591`, `action_f1=0.230`,
    `source_target_f1=0.242`; failed density, score delta, and F1 thresholds.
- Online candidate-mask eval also regressed: `1/64` versus regular
  (`nonloss=0.0156`), worse than sourcehead `e1300` under the same
  candidate-mask setting (`6/64` in the earlier audit).
- Decision: do not promote this checkpoint. The DAgger data construction is
  correct, but source/launch-only adaptation is not enough and can reduce action
  density. The next aligned delta should address action construction as a
  coupled decoder problem: target candidate/factorization plus ship/action
  amount, while preserving the candidate/safety mask that offline diagnostics
  showed to be essential. Continuing to train the same source-head-only setup is
  not expected to fix the distribution shift.

## 2026-05-27 broad DAgger all-heads result

- Ran the paired all-heads control:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_modelseat_w03_allheads_e1300_e160_noeval_20260527`.
  It used the same pure regular cache, same broad model-seat regular-labelled
  DAgger cache, same sourcehead `e1300` resume checkpoint, and the same
  `dagger_loss_weight=0.3` mix as the source-head-only run. The only intended
  delta was `--trainable-modules all`, so target/ship/count heads could move
  together with source selection. The job trained the full `160` epochs with
  `64` trainers across GPUs `0-7`.
- Training did not show a late rescue from deeper optimization. Train loss kept
  decreasing, but validation loss rose from about `1.81` early to `1.8508` at
  epoch `160`. Best validation imitation score was `0.64267`, and the final
  epoch had `val_imitation_score=0.63822`, `val_launch_f1=0.6675`,
  `val_action_count_mae=1.6452`, and `val_target_pair_acc=0.5771`. This looks
  like train-distribution fitting rather than improved deployable behavior.
- Same-state Phase1 gate versus sourcehead `e1300` did not pass:
  - all-planets decoding: candidate `selected_score=0.1756` versus baseline
    `0.1805`, `density_ratio=0.727`, `action_f1=0.258`,
    `source_target_f1=0.268`; failed score delta and F1 thresholds.
  - candidate-mask decoding: candidate `selected_score=0.1354` versus baseline
    `0.1295`, but `density_ratio=0.599`, `action_f1=0.241`, and
    `source_target_f1=0.253`; failed density and F1 thresholds despite the
    small score increase.
- Online candidate-mask eval was again `1/64` versus regular
  (`nonloss=0.0156`) at `launch_bias=-0.05`, `ship_bias=0.0`,
  `target_pair_weight=1.0`, `target_top_k=6`.
- Decision: do not promote this checkpoint. The failure mode is now replicated
  with both source-head-only and all-heads DAgger at the intended 20% DAgger mix:
  more epochs and simply unfreezing all heads do not lower distribution drift.
  The next useful experiment should be a small, controlled capacity/decoder
  check rather than another long replay of the same setup: for example compare
  `hidden=256` or `layers=3` for a short run under the same data mix, and in
  parallel work on action construction/factorization so candidate-safe target
  choices, source choice, ship amount, and action count are decoded as a
  coherent regular action set.

## 2026-05-27 layers=3 capacity check

- Ran a controlled capacity check:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_modelseat_w03_layers3_e1300_e80_noeval_20260527`.
  The data, DAgger weight, resume checkpoint, optimizer settings, and all-heads
  training setup matched the previous all-heads run; the intended architecture
  delta was `layers=2 -> 3` with `hidden=128`. `resume-compatible` loaded `41`
  tensors and skipped none, so this was a warm-started continuation rather than
  a random restart.
- Loader metrics did not indicate a capacity breakthrough. Best validation
  imitation score was `0.64204`, slightly below the layers=2 all-heads run
  (`0.64267`), and final validation loss rose to `1.8460`. Train loss kept
  falling, so the pattern remains train-distribution fitting without a matching
  validation/deployment gain.
- Same-state Phase1 gate versus sourcehead `e1300` did not pass:
  - all-planets decoding: candidate `selected_score=0.1862` versus baseline
    `0.1805`, `density_ratio=0.724`, `action_f1=0.268`,
    `source_target_f1=0.276`; the small score lift did not clear the required
    `+0.02` margin and F1 thresholds.
  - candidate-mask decoding: candidate `selected_score=0.1344` versus baseline
    `0.1295`, but `density_ratio=0.573`, `action_f1=0.243`, and
    `source_target_f1=0.255`; it is still strongly under-launching.
- Online candidate-mask eval remained in the bad regime: `1/64` versus regular
  (`nonloss=0.0156`) at `launch_bias=-0.05`, `ship_bias=0.0`,
  `target_pair_weight=1.0`, `target_top_k=6`.
- Decision: do not promote this checkpoint. The evidence argues against
  "slightly deeper model" as the main blocker. A larger model may still be worth
  testing later, but the next aligned work should prioritize the action decoder:
  candidate-safe target factorization, action-count calibration, and ship
  amount/source-target coupling under the regular-labelled DAgger distribution.

## 2026-05-27 safe-mask decoder check

- Tested whether the online collapse was mainly caused by the runtime
  `candidate` target mask excluding regular labels. On a shared 96-state
  held-out regular sample using the layers=3 best-imitation checkpoint,
  `candidate` mode covered only `0.174` of regular label targets
  (`0.279` with friendly targets enabled), while the training safety mask hit
  `0.921`. The broader `safe` runtime mask covered `0.784`; `all_planets`
  covered `1.0` but still dropped actions later because unsafe argmax targets
  fail the final path check.
- Same sample, `launch_bias=-0.05`, target-pair weight `1.0`:
  - `candidate`: `model_actions_per_state=1.177` versus regular `1.979`,
    `source_target_f1=0.192`, `action_f1=0.183`.
  - `candidate + friendly`: density improved to `1.802`, but
    `source_target_f1=0.203`, `action_f1=0.193`; coverage alone was not enough.
  - `safe`: density `1.781`, `source_target_f1=0.244`, `action_f1=0.237`,
    and runtime label-target coverage `0.784`. This confirms the narrow
    candidate set is a real same-state bottleneck, but target ranking remains
    weak even when labels are mostly available.
- Ran a 64-task Ray online check, `4` games per task (`256` games per variant),
  versus regular using the layers=3 best-imitation checkpoint and `safe` mask:
  - `launch_bias=-0.25`: `1W/255L/0D`, nonloss `0.0039`.
  - `launch_bias=-0.05`: `4W/252L/0D`, nonloss `0.0156`.
  - `launch_bias=0.15`: `1W/255L/0D`, nonloss `0.0039`.
- Decision: do not promote `safe` decoding as the next baseline. It improves
  same-state target availability, but online behavior is still in the stable
  bad regime. The next model-side delta should train or decode target choice
  against the runtime-safe candidate space directly, rather than relying on a
  heuristic top-k candidate mask or only expanding the mask at inference time.
- Tooling: `tinyPPO.phase1_gate` now accepts `--workers` to parallelize
  seed/checkpoint evaluations. A CPU smoke check with `--workers 2` completed,
  so broader same-state gates no longer need to run as one slow process.

## 2026-05-27 safe-mask data collection support

- Added `--row-target-mask-mode {candidate,safe,all_planets}` to the regular BC
  and DAgger collection paths. The default stays `candidate` for compatibility,
  but new caches can now store the same broader `safe` target mask used by the
  runtime decoder instead of only storing `candidate + forced label target`.
- This is a data-construction delta, not a DAgger semantic change: model
  rollouts still provide the visited states, and labels still come from
  `regular` on the same observation/player perspective.
- Smoke checks:
  - local BC: `tinyPPO.imitation_regular` with `--row-target-mask-mode safe`,
    `1` game, `4` rows, `1` CPU epoch completed and saved
    `/tmp/orbit_safe_mask_smoke.pt`; collect metrics reported
    `row_target_mask_mode=safe`.
  - Ray DAgger: `tinyPPO.collect_dagger_cache` with `1` actor, `1` model-seat
    game, checkpoint `regular_bc_ray_e0400.pt`, and
    `--row-target-mask-mode safe` completed and saved
    `/tmp/orbit_safe_dagger_smoke.pkl`; metrics reported `3` rows, `16`
    labelled actions, and `row_target_mask_mode=safe`.
- Next experiment should regenerate a safe-mask pure regular cache and a
  safe-mask broad checkpoint DAgger cache, then repeat the intended 70-80% pure
  / 20-30% DAgger BC mix. This directly tests whether training target ranking
  against the runtime-safe negative set fixes the same-state/online gap that
  inference-only `safe` decoding did not fix.

## 2026-05-28 safe-mask regular plus broad DAgger rerun

- Collected a new pure regular safe-mask cache:
  `tinyPPO/data/regular_bc_rulefeat_2p_10000g_rows16_safe_20260527.pkl`.
  It used `10000` 2P games, `rows_per_game=16`, and
  `--row-target-mask-mode safe`. The loader reported `200838` samples and
  `352925` labelled actions.
- Collected the model-seat DAgger data in 8 shards to avoid Ray object-store
  OOMs:
  `tinyPPO/data/regular_bc_dagger_broad5ckpt_modelseat_2p_0640g_rows8_safe_20260528_shard*.pkl`.
  Total metrics were `5120` games, `40884` samples, `286907`
  regular-labelled actions, and `300006` model-seat label actions. Rollouts
  used checkpoints `e0400/e0600/e0900/e1100/e1300`, model p0 versus regular p1,
  and labels came from running `regular` on the model-seat observation.
- Trained:
  `tinyPPO/runs/regular_bc_dagger_broad5ckpt_modelseat_safemask_w03_allheads_e1300_e160_noeval_20260528`.
  It resumed sourcehead `e1300`, used `dagger_loss_weight=0.3`, trained all
  heads for `160` epochs, and disabled online eval during training. Weighted
  action mass was about `80.4%` pure regular / `19.6%` DAgger.
- Offline training showed mild overfit after the early/mid epochs. Best
  imitation score was `0.54904` at epoch `158`, but validation target/pair
  accuracy mostly plateaued after about epoch `40` while train metrics kept
  improving. Final epoch `160` had `val_imitation_score=0.54764`,
  `val_target_acc=0.64282`, `val_target_pair_acc=0.55873`, and
  `val_action_count_mae=1.7894`.
- Same-state Phase1 gate with `safe` decoding versus sourcehead `e1300`
  improved clearly for the checked epochs:
  - `e0040`: `selected_score=0.2532`, `action_f1=0.3017`,
    `source_target_f1=0.3102`, `density_ratio=0.984`, score delta `+0.0714`.
  - `e0080`: `selected_score=0.2514`, `action_f1=0.3120`,
    `source_target_f1=0.3242`, `density_ratio=0.877`, score delta `+0.0696`.
  - `e0120`: `selected_score=0.2544`, `action_f1=0.3097`,
    `source_target_f1=0.3216`, `density_ratio=0.909`, score delta `+0.0727`.
  The remaining slow same-state sweep was stopped after these three passed; no
  need to spend more CPU before online confirmation.
- 64-game online safe-mask sweep versus regular:
  - `e0040`: best `launch_bias=-0.25`, `20W/44L/0D`, nonloss `0.3125`.
  - `e0080`: best `launch_bias=0.15`, `13W/51L/0D`, nonloss `0.2031`.
  - `e0120`: best `launch_bias=-0.05`, `16W/48L/0D`, nonloss `0.25`.
- 256-game confirmation of the two best variants:
  - `e0040`, `launch_bias=-0.25`: `46W/210L/0D`, nonloss `0.1797`.
  - `e0120`, `launch_bias=-0.05`: `62W/194L/0D`, nonloss `0.2422`.
- Decision: this is the first model-side branch in this phase that escapes the
  repeated `1/64` online-collapse regime, so the safe-mask data construction is
  a real improvement. It is still far below regular and should not be promoted
  as-is. The next single delta should keep this safe-mask data/decoder setup
  fixed and improve action coherence, especially source-target/ship/count
  coupling; do not spend more time merely extending epochs, since e40/e120
  online results beat the later best-imitation checkpoint behavior.

## 2026-05-28 corrected safe DAgger iter2

- Found and fixed a DAgger rollout/relabel semantic bug:
  `regular` agents are stateful, but the previous collection/diagnostic actors
  reused one regular instance across games and players. The DAgger rollout path
  also did not pass the runtime model target-mask mode, so safe-decoder online
  policies were generating DAgger states with the older `candidate` mask.
- Commit: `6be94ea Fix DAgger regular relabel rollout semantics`.
  The corrected collection now creates fresh regular agents per episode/player,
  uses a separate fresh regular teacher for the model-seat observation, and
  records `model_target_mask_mode` / `model_target_pair_weight` in metrics.
- Collected corrected iter2 DAgger from the previous best safe online branch:
  `tinyPPO/data/regular_bc_dagger_iter2_e0120_safedecode_m005_2p_0512g_rows8_safe_20260528_shard*.pkl`.
  It used checkpoint `regular_bc_ray_e0120.pt`, model p0 vs fresh regular p1,
  `--dagger-model-seat-only`, `--dagger-model-target-mask-mode safe`,
  `--launch-bias -0.05`, `--row-target-mask-mode safe`, 64 actors, and manual
  GPU IDs across 0-7. Total metrics:
  - `4096` games.
  - `32168` samples.
  - `69261` regular-labelled actions.
  - `70157` model-seat regular label actions before target-mask skips.
  - All shards reported `row_target_mask_mode=safe`,
    `model_target_mask_mode=safe`, `model_target_pair_weight=1.0`.
- Trained:
  `tinyPPO/runs/regular_bc_dagger_iter2_e0120_safedecode_m005_w1_allheads_e0120_e200_noeval_20260528`.
  It resumed the previous safe online `e0120` checkpoint, mixed the safe pure
  regular cache with only the corrected iter2 DAgger shards, used
  `dagger_loss_weight=1.0`, all heads trainable, hidden `128`, layers `2`,
  source-target summary, target-pair head, safe dataset target mask, 64 trainer
  actors, and no in-loop online eval.
- Offline result: training loss continued to decrease, but pure-regular
  validation did not improve. Best imitation was only `0.53246`, below the
  previous broad safe DAgger best `0.54904`. Later epochs showed the expected
  pattern of train target/pair metrics rising while pure-regular validation
  target metrics softened, so extending epochs alone is not promising.
- Online safe 64-game sweep:
  - `e0140`: best `14W/50L/0D`, nonloss `0.219`.
  - `e0160`: best `15W/49L/0D`, nonloss `0.234`.
  - `e0180`: best `launch_bias=-0.15`, `17W/47L/0D`, nonloss `0.266`.
  - `e0200`: best `launch_bias=-0.25`, `17W/47L/0D`, nonloss `0.266`.
- 256-game confirmation did not hold the apparent 64-game gain:
  - `e0180`, `launch_bias=-0.15`: `54W/202L/0D`, nonloss `0.211`.
  - `e0200`, `launch_bias=-0.25`: `48W/208L/0D`, nonloss `0.188`.
  Both are below the previous `e0120`, `launch_bias=-0.05` result
  `62W/194L/0D`, nonloss `0.242`.
- Decision: the corrected DAgger semantics are required for future data, but
  this iter2-only update did not become a better policy. The bottleneck now
  looks less like seed/epoch count and more like action coherence under shifted
  states: source-target-owner/ship/count coupling is still too weak. The next
  experiment should keep the corrected safe DAgger data and training recipe
  fixed, change exactly one model-structure delta, and validate by online
  rollout rather than pure loss.

## 2026-05-28 corrected iter2 capacity probe: layers 3

- Tested the simplest model-complexity hypothesis as a single delta: resume the
  same previous safe-online `e0120` checkpoint with `--resume-compatible`, keep
  hidden `128`, heads `4`, all data/loss/decode settings fixed, and change only
  Transformer context depth from layers `2` to layers `3`.
- Run:
  `tinyPPO/runs/regular_bc_dagger_iter2_e0120_safedecode_m005_w1_layers3_e0120_e220_noeval_20260528`.
  It used the safe pure regular cache plus the corrected iter2 DAgger shards,
  `dagger_loss_weight=1.0`, all heads trainable, source-target summary,
  target-pair head, safe dataset target mask, 64 trainer actors, and no in-loop
  online eval.
- Offline result: best imitation was only `0.53012`, worse than the layers-2
  corrected iter2 run (`0.53246`) and clearly below the previous broad safe
  DAgger run (`0.54904`). During training, train target/pair metrics continued
  improving (`train_target_acc` reached about `0.673`, `train_target_pair_acc`
  about `0.618`), while pure-regular validation target/pair degraded
  (`val_target_acc` about `0.609`, `val_target_pair_acc` about `0.553` at
  `e0220`), so the added layer mostly increased fit to the mixed training data
  rather than generalization.
- Online safe 64-game probe:
  - `e0020`: best `launch_bias=0.0`, `14W/50L/0D`, nonloss `0.219`.
  - `e0040`: best observed `launch_bias=-0.15`, `16W/48L/0D`,
    nonloss `0.25`.
  The longer coarse sweep was stopped after `e0040` because later offline
  checkpoints were already showing stronger overfit.
- 256-game confirmation of the apparent best did not hold:
  - `e0040`, `launch_bias=-0.15`: `55W/201L/0D`, nonloss `0.215`.
  This is below the previous best safe online confirmation (`e0120`,
  `launch_bias=-0.05`: `62W/194L/0D`, nonloss `0.242`).
- Decision: simple extra depth is not the missing piece. Model capacity might
  still matter, but the current evidence points away from "just make it bigger"
  and toward better action factorization / hard-state data: the model is not
  consistently binding source, target, ship amount, and action count under
  model-induced states.

## 2026-05-28 corrected iter2 DAgger weight probe

- Tested the "maybe the corrected DAgger rows are underweighted" hypothesis as
  a single data-weight delta. The corrected iter2 cache has only `32168`
  DAgger rows versus `192122` safe pure-regular rows, so `dagger_loss_weight=1`
  gave only about `14%` row mass. This run kept the layers-2 architecture,
  resume checkpoint, losses, masks, decoder, and no-eval training cadence fixed,
  and changed only `--dagger-loss-weight` from `1.0` to `2.0`.
- Run:
  `tinyPPO/runs/regular_bc_dagger_iter2_e0120_safedecode_m005_w2_layers2_e0120_e260_noeval_20260528`.
  Effective action mass was approximately `352925` pure-regular labelled
  actions versus `138522` weighted DAgger labelled actions, i.e. about `28%`
  DAgger exposure. This matches the intended `70-80%` pure regular /
  `20-30%` DAgger mix better than the previous `w1` run.
- Offline result: best imitation was only `0.52425`, worse than both the `w1`
  corrected iter2 layers-2 run (`0.53246`) and layers-3 probe (`0.53012`).
  This confirms that simply increasing corrected DAgger pressure hurts
  same-state generalization rather than repairing it.
- Online safe probe:
  - 64-game sweep on `e0140` showed an apparent high point at
    `launch_bias=-0.15`: `18W/46L/0D`, nonloss `0.281`.
  - 256-game confirmation of that exact setting did not hold:
    `49W/207L/0D`, nonloss `0.191`.
  This is below the previous best safe online confirmation (`e0120`,
  `launch_bias=-0.05`: `62W/194L/0D`, nonloss `0.242`) and below the layers-3
  confirmation (`55W/201L/0D`, nonloss `0.215`).
- Decision: do not continue later `w2` checkpoints unless explicitly requested.
  The failure mode is not just "too little DAgger weight". The next useful
  single delta should be data selection / hard-state quality, using the
  corrected safe DAgger semantics but filtering for states where the regular
  teacher has a clear, meaningful correction. Candidate filters already exist
  in `imitation_regular_ray.py` (`--dagger-min-abs-action-gap`,
  `--dagger-min-actions-per-row`, outcome/turn filters), but choose the filter
  only after running the cheap DAgger diagnostics on the corrected iter2 cache.

## 2026-05-28 corrected iter2 hard-gap data-selection probe

- Tested one data-quality delta on top of the corrected safe DAgger semantics:
  keep the previous safe-online `e0120` checkpoint, layers `2`, hidden `128`,
  all loss/mask/decode settings fixed, but train only on corrected iter2 DAgger
  rows where the regular teacher's labelled action count differed from the
  model action count by at least `2` (`--dagger-min-abs-action-gap 2`).
  `dagger_loss_weight=2.5` was used only to keep the retained hard-state action
  mass near the intended 20-30% range.
- Data diagnostics: the corrected iter2 cache had `32168` total DAgger rows and
  `69261` labelled actions before filtering. The hard-gap filter retained
  `10320` rows (`0.321` keep fraction) and `42103` labelled actions, for about
  `105257.5` weighted DAgger labelled actions versus `352925` pure-regular
  labelled actions.
- Run:
  `tinyPPO/runs/regular_bc_dagger_iter2_e0120_safedecode_m005_gap2_w25_layers2_e0120_e360_noeval_20260528`.
  Training was stopped after the `e0220` checkpoint / `e0229` current state
  because same-state validation had already degraded: train target/pair metrics
  kept improving while pure-regular validation target/pair softened. Best
  imitation was only about `0.505`, far below the previous safe baseline.
- Online safe probe:
  - 64-game sweep on `e0160` showed an apparent high point at `launch_bias=0.0`:
    `18W/46L/0D`, nonloss `0.281`.
  - 256-game confirmation of that exact setting did not hold:
    `57W/199L/0D`, nonloss `0.223`.
  This is below the previous best safe online confirmation (`e0120`,
  `launch_bias=-0.05`: `62W/194L/0D`, nonloss `0.242`).
- Decision: hard-gap filtering alone over-concentrates on shifted/hard states
  and does not produce a better rollout policy. Together with the layers-3
  capacity probe, this makes "just increase model complexity" a low-priority
  next step. Capacity may still matter if paired with a better objective, but
  the current bottleneck looks more like action coherence and distribution
  handling than raw parameter count.
