# tinyPPO

Tiny 2P PPO baseline for Orbit Wars.

Design choices:

- Uses `training2.make_fast_orbit_wars(..., use_numba=True)` for rollout and smoke checks.
- Starts with 2P only and sparse terminal reward `+1/-1`.
- The model never outputs an angle. It builds every source-target pair from map coordinates, scores target choices on those edges, and predicts ship-fraction buckets conditioned on the selected edge. The agent computes the actual firing angle from source and target coordinates.
- Each owned source planet uses an autoregressive launch loop: continue/stop, target, ship bucket, repeat until stop or no ships remain. A source can launch multiple fleets in the same turn. The decoder tracks remaining ships per source planet and clips/skips launches so the total ships launched from one planet never exceeds its current garrison. `--max-actions-per-source-safety` is only a dead-loop guard for training/inference, not a fixed slot head in the model.
- Source planets are represented by current board coordinates, not just planet IDs. `from_planet_id` is only used after the geometric action is decoded for the simulator.
- The architecture is not a copy of the older Transformer policy. It is a small geometry-first actor critic: planet MLP, pairwise source-target edge MLP, per-source continue/stop head, per-edge target head, per-edge ship bucket head, and a pooled value head.
- `nearest_planet_agent` is eval-only. It is not sampled during training.

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

Ray rollout training on a remote node exposing GPUs 0-8:

```bash
uv run --active python -m tinyPPO.train_ray \
  --ray-address auto \
  --gpu-ids 0,1,2,3,4,5,6,7,8 \
  --workers-per-gpu 3 \
  --episodes-per-worker 2 \
  --out-dir tinyPPO/runs/ray_remote_2p_v1
```

Use `uv run --active` for Ray jobs in this repo; plain `uv run` can make Ray
package the project and build a fresh runtime environment instead of reusing the
current `.venv`.

SwanLab is enabled by default for new tinyPPO runs. It reads
`SWANLAB_API_KEY` or `training/config/swanlab_key.txt`. Use `--no-swanlab` to
disable it, or `--allow-no-swanlab` for local debugging when credentials are
unavailable.

Short replay is available but disabled by default. For PPO, keep it recent and
age-decayed rather than treating old trajectories as equally valid:

```bash
  --replay-updates 2 \
  --replay-ratio 0.5 \
  --replay-age-decay 0.5
```

This lowers the loss weight of older samples; it does not alter or punish their
rewards.

Evaluate a checkpoint against nearest-planet:

```bash
uv run python -m tinyPPO.eval --checkpoint tinyPPO/runs/random_2p_v1/best.pt --opponent nearest --games 40
```

Stop criteria for this baseline:

- Primary: balanced-seat winrate versus nearest-planet reaches `--stop-winrate`.
- Early warning: `clip_frac` rises monotonically above about `0.30`, entropy collapses, or rollout `mean_launches` goes to all-zero/all-source behavior without eval improvement.
- If eval versus nearest does not move after a cheap random-opponent run, inspect action histograms and rollout replays before changing architecture.
