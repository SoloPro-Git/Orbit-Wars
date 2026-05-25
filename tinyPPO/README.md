# tinyPPO

Tiny 2P PPO baseline for Orbit Wars.

Design choices:

- Uses `training2.make_fast_orbit_wars(..., use_numba=True)` for rollout and smoke checks.
- Starts with 2P only and sparse terminal reward `+1/-1`.
- The model never outputs an angle. It builds every source-target pair from map coordinates, scores target choices on those edges, and predicts ship-fraction buckets conditioned on the selected edge. The agent computes the actual firing angle from source and target coordinates.
- Each owned source planet has a small fixed number of launch slots
  (`--action-slots`, default 3). A source can launch multiple fleets in the same
  turn, but at most three. The decoder tracks remaining ships per source planet
  and clips/skips launches so the total ships launched from one planet never
  exceeds its current garrison.
- Source planets are represented by current board coordinates, not just planet IDs. `from_planet_id` is only used after the geometric action is decoded for the simulator.
- The architecture is not a copy of the older Transformer policy. It is a small geometry-first actor critic: planet MLP, pairwise source-target edge MLP, per-source-slot launch head, per-source-slot target head, per-source-slot ship bucket head, and a pooled value head.
- `nearest_planet_agent` is eval-only. It is not sampled during training.

Phase split:

- Phase 1 is regular behavior cloning only. The checkpoint should imitate the
  regular rulebase on the same observed states: launch/source decisions, target
  choices, and ship buckets. Do not use winrate against regular as the BC stop
  condition, and do not use PPO reward or opponent promotion in this phase.
- Phase 1 diagnostics should prioritize held-out regular-state imitation
  (`source_f1`, `source_target_f1`, `action_f1`, action-count error, and target
  legality). Online games against regular are only a sanity check because both
  agents immediately change the state distribution after the first mismatch.
- For all-planets BC checkpoints, evaluate and later roll out with
  `--target-mask-mode all_planets`. The older candidate mask can exclude many
  regular targets before the learned target head gets to choose; the decoder
  still checks ship counts, sun paths, and board bounds before emitting actions.
- Phase 2 starts from the best Phase 1 checkpoint and runs PPO/self-play against
  a frozen older checkpoint. That is where winning the old checkpoint, promotion
  gates, and eventually beating regular belong.

Quick smoke:

```bash
uv run python -m tinyPPO.train --device cpu --updates 1 --episodes-per-update 1 --episode-steps 80 --eval-games 2 --no-numba
```

Train against random first:

```bash
uv run python -m tinyPPO.train \
  --out-dir tinyPPO/runs/random_2p_v1 \
  --opponent-mode random \
  --updates 200 \
  --episodes-per-update 16 \
  --eval-interval 10 \
  --eval-games 40 \
  --stop-winrate 0.55
```

Curriculum training, random first, then latest-checkpoint opponent:

```bash
uv run --active python -m tinyPPO.train \
  --out-dir tinyPPO/runs/curriculum_2p_v1 \
  --curriculum \
  --random-winrate-threshold 0.90 \
  --selfplay-entropy-coef 0.04 \
  --eval-interval 10 \
  --eval-games 40 \
  --stop-winrate 0.55
```

During curriculum training, `nearest_planet_agent` is eval-only. It is never used
as a rollout opponent and never contributes training samples.

Ray rollout training, local machine using GPUs 1-7 with 3 workers/GPU:

```bash
uv run --active python -m tinyPPO.train_ray \
  --gpu-ids 1,2,3,4,5,6,7 \
  --workers-per-gpu 3 \
  --episodes-per-worker 2 \
  --curriculum \
  --random-winrate-threshold 0.90 \
  --swanlab-experiment tinyPPO-autoreg-fastenv-ray-2p \
  --out-dir tinyPPO/runs/ray_local_2p_v1
```

Ray evaluation is asynchronous by default in `train_ray`: a few eval actors
reserve fractional GPUs, eval jobs are submitted every `--eval-interval`, and
training continues while eval runs. Stopping requires two passing evals by
default:

```bash
  --eval-workers 3 \
  --gpus-per-eval-worker 0.3333333333333333 \
  --stop-eval-confirmations 2
```

Use `--sync-eval` only when you explicitly want training to wait during eval.

To follow the existing `training2` multi-node Ray style, start/connect to the
cluster with `RAY_ADDRESS`:

```bash
export RAY_ADDRESS=10.0.104.198:6380
uv run --active python -m tinyPPO.train_ray --ray-address "$RAY_ADDRESS" ...
```

Worker nodes must already be joined with `ray start --address='10.0.104.198:6380'`.
Use `training2/sync_ray_assets.py` when code files need to be copied to remote
workers before launching.

To keep eval on local physical GPU 0 while rollout uses GPUs 1-7:

```bash
  --gpu-ids 1,2,3,4,5,6,7 \
  --eval-gpu-ids 0 \
  --gpus-per-eval-worker 0
```

To evaluate and promote against a frozen previous checkpoint, keep the
checkpoint as `opponent.pt` and enable the promotion gate. The stochastic
comparison is enabled automatically when the promotion metric needs it:

```bash
  --opponent-checkpoint tinyPPO/runs/<run>/opponent.pt \
  --freeze-latest-opponent \
  --promote-opponent-on-eval \
  --promote-opponent-threshold 0.80 \
  --promote-opponent-metric eval_stochastic_vs_opponent
```

By default Ray training runs for up to `100000` updates and stops only after
`eval_stochastic_vs_regular` reaches `0.70` for the configured confirmation
count.

To instead refresh the frozen opponent on a fixed eval cadence, use:

```bash
  --refresh-opponent-on-eval \
  --opponent-refresh-interval 1
```

Each refresh updates `opponent.pt` and also saves the exact switch point under
`opponent_history/opponent_uXXXXXX.pt`.

To add one async eval worker on physical GPU 1 while keeping it outside Ray's
fractional GPU accounting:

```bash
  --eval-workers 1 \
  --eval-gpu-ids 1 \
  --gpus-per-eval-worker 0
```

Ray eval also reports the strongest local regular rulebase as an eval-only
opponent. It appears in logs/SwanLab as `eval_vs_regular`, and when stochastic
comparison is enabled as `eval_stochastic_vs_regular`.

Ray rollout training on a remote node exposing GPUs 0-8:

```bash
uv run --active python -m tinyPPO.train_ray \
  --ray-address auto \
  --gpu-ids 0,1,2,3,4,5,6,7,8 \
  --workers-per-gpu 3 \
  --episodes-per-update 42 \
  --episodes-per-worker 2 \
  --out-dir tinyPPO/runs/ray_remote_2p_v1
```

`--episodes-per-update` keeps the fresh on-policy batch at a fixed waterline as
more GPUs/workers are added. `--episodes-per-worker` is only the per-worker cap.

Use the repo `.venv/bin/python` directly for multi-node Ray jobs if remote
workers hang while launching through `uv run --active`; plain `uv run` can make
Ray package the project and build a fresh runtime environment instead of reusing
the current `.venv`.

SwanLab is enabled by default for new tinyPPO runs. It reads
`SWANLAB_API_KEY` or `training/config/swanlab_key.txt`. Use `--no-swanlab` to
disable it, or `--allow-no-swanlab` for local debugging when credentials are
unavailable.

Short replay is enabled conservatively by default. For PPO, keep it recent and
age-decayed rather than treating old trajectories as equally valid:

```bash
  --replay-updates 2 \
  --replay-ratio 0.25 \
  --replay-age-decay 0.5
```

This lowers the loss weight of older samples; it does not alter or punish their
rewards.

Ship counts are learned as a continuous source-target-slot fraction via a Beta
distribution, then clamped by the decoder against the source planet's remaining
ships. Each source can still launch at most `--action-slots 3` actions.

Evaluate a checkpoint against nearest-planet:

```bash
uv run python -m tinyPPO.eval --checkpoint tinyPPO/runs/random_2p_v1/best.pt --opponent nearest --games 40
```

Stop criteria for this baseline:

- Primary: balanced-seat winrate versus nearest-planet reaches `--stop-winrate`.
- Early warning: `clip_frac` rises monotonically above about `0.30`, entropy collapses, or rollout `mean_launches` goes to all-zero/all-source behavior without eval improvement.
- If eval versus nearest does not move after a cheap random-opponent run, inspect action histograms and rollout replays before changing architecture.
