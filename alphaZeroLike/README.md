# alphaZeroLike

Independent AlphaZero-like prototype for Orbit Wars.

The important difference from `training2` is the action representation and
training target:

- Candidate actions are still generated from rulebase/heuristic plans, because
  the raw Orbit Wars action space is combinatorial and partly continuous.
- Each candidate is encoded as a set of launch tokens instead of one compressed
  32-d vector.
- The policy target is a search-improved visit distribution from shallow PUCT,
  not a one-hot oracle index.
- Value still trains from final game result.

Smoke test:

```bash
uv run python -m alphaZeroLike.smoke
```

Generate a tiny self-play JSONL:

```bash
uv run python -m alphaZeroLike.self_play \
  --games 1 \
  --simulations 8 \
  --rollout-depth 2 \
  --out data/alphaZeroLike/selfplay_smoke.jsonl
```

Train on generated rows:

```bash
uv run python -m alphaZeroLike.train \
  --data data/alphaZeroLike/selfplay_smoke.jsonl \
  --out alphaZeroLike/checkpoints/latest.pt
```

Continue from an `alphaZeroLike` checkpoint:

```bash
uv run python -m alphaZeroLike.train \
  --data data/alphaZeroLike/selfplay_next.jsonl \
  --resume alphaZeroLike/checkpoints/latest.pt \
  --out alphaZeroLike/checkpoints/latest.pt
```

Bootstrap the new model from a `training2` stage1 checkpoint:

```bash
uv run python -m alphaZeroLike.train \
  --data data/alphaZeroLike/selfplay_smoke.jsonl \
  --init-from-training2 training2/checkpoints/stage1_tactical_entities_20260519/latest.pt \
  --out alphaZeroLike/checkpoints/latest.pt
```

Only the compatible backbone/value tensors are loaded.  The old compressed
action projection, policy head, and proposal head are intentionally skipped
because `alphaZeroLike` uses a different move-set action encoder and a
search-improved policy target.

To phase out rulebase candidates, construct agents/search with:

```python
from alphaZeroLike.candidates import CandidateConfig

cfg = CandidateConfig(use_rulebase=False, include_heuristics=False)
```

In that mode, candidates come from the model proposal head plus a no-op
fallback; the model still computes launch angles by deterministic intercept
rules after choosing source, target, and ship ratio.

CLI equivalent:

```bash
uv run python -m alphaZeroLike.self_play \
  --checkpoint alphaZeroLike/checkpoints/latest.pt \
  --no-rulebase-candidates \
  --no-heuristics \
  --proposal-num-full-actions 16 \
  --proposal-num-candidates 32 \
  --out data/alphaZeroLike/model_only_selfplay.jsonl
```

The proposal head is not a separate model by default.  It shares the
planet/global backbone with the ranker/value network and has separate heads for:

- `send_logits`: whether each owned planet should launch
- `target_logits`: target planet distribution per source
- `ship_logits`: ship ratio per source

Gradients from `policy_loss`, `value_loss`, and `proposal_loss` all update the
shared backbone unless you explicitly freeze parameters in a custom training
script.  `ProposalConfig.num_full_actions` controls how many complete multi-move
plans are built from the proposal head before adding single-move/ratio variants.

This is intentionally a first search scaffold, not a finished full-tree
AlphaZero.  `mcts.py` isolates the shallow search and `env_clone.py` isolates
fast-simulator cloning so the next step can replace the root-only rollout with a
deeper transposition-aware tree.

## Ray + SwanLab

The unified config lives at:

```bash
alphaZeroLike/config/default.yaml
```

Run mixed training:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m alphaZeroLike.train_ray \
  --config alphaZeroLike/config/default.yaml \
  --stage mixed
```

Run model-only training after the proposal head is warm:

```bash
CUDA_VISIBLE_DEVICES=4,5,6,7 uv run python -m alphaZeroLike.train_ray \
  --config alphaZeroLike/config/default.yaml \
  --stage model_only
```

The stage override patches the same base YAML, so model size, Ray resources,
MCTS depth, proposal counts, and SwanLab project settings stay in one place.
