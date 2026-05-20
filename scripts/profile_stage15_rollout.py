"""Profile the local stage1.5 rollout hot path.

This intentionally runs outside Ray so it can answer a simple question:
where does one rollout worker spend time when using the same fast simulator,
rulebase opponents, candidate builder, encoder, and policy forward path as
training2.train_stage15_ray?
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import torch

from training2.candidates import _canonical, build_candidates
from training2.envs import make_orbit_wars_env
from training2.features import encode_position, result_value
from training2.model import load_compatible_state_dict
from training2.proposal import ProposalConfig, proposals_from_model
from training2.rulebase_bridge import make_rulebase_agent
from training2.train_stage15_ray import _load_yaml, _make_model, _raw_obs, _resolve_project_path, _score


def _timer(times: dict[str, float], key: str):
    class Timer:
        def __enter__(self):
            self.start = time.perf_counter()
            return self

        def __exit__(self, exc_type, exc, tb):
            times[key] += time.perf_counter() - self.start

    return Timer()


def _load_checkpoint(model, checkpoint: str, device: str) -> dict[str, list[str]]:
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    state = ckpt.get("model_state_dict", ckpt)
    return load_compatible_state_dict(model, state, strict=False)


def _rulebase_margin(
    *,
    seed: int,
    players: int,
    model_pid: int,
    oracle: str,
    env_backend: str,
    env_use_numba: bool,
    times: dict[str, float],
    counts: dict[str, int],
) -> tuple[float, float]:
    with _timer(times, "baseline_env_init_reset"):
        env = make_orbit_wars_env(
            {"episodeSteps": 500, "seed": seed},
            backend=env_backend,
            debug=True,
            use_numba=env_use_numba,
        )
        env.reset(players)
    agents = {pid: make_rulebase_agent(oracle) for pid in range(players)}
    for _ in range(500):
        actions = []
        for pid in range(players):
            obs = _raw_obs(env, pid)
            with _timer(times, "baseline_rulebase_action"):
                actions.append(agents[pid](obs) or [])
            counts["baseline_rulebase_calls"] += 1
        with _timer(times, "baseline_env_step"):
            env.step(actions)
        counts["baseline_env_steps"] += 1
        if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
            break
    final_obs = _raw_obs(env, model_pid)
    my_score = _score(final_obs, model_pid)
    opp_best = max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
    return my_score - opp_best, result_value(final_obs, model_pid)


def profile_rollout(args: argparse.Namespace) -> dict[str, Any]:
    if args.torch_threads is not None:
        torch.set_num_threads(max(1, int(args.torch_threads)))
        torch.set_num_interop_threads(max(1, int(args.torch_threads)))

    cfg = _load_yaml(args.config)
    stage = cfg.get(args.stage_key, {})
    if not stage:
        raise KeyError(f"Missing stage config section: {args.stage_key}")

    model_cfg = cfg.get("model", {})
    max_candidates = int(model_cfg.get("max_candidates", 32))
    proposal_cfg = dict(cfg.get("proposal", {}))
    proposal_cfg.update(dict(stage.get("proposal_overrides", {}) or {}))
    if args.force_proposals:
        proposal_cfg["enabled"] = True
    if args.disable_proposals:
        proposal_cfg["enabled"] = False

    env_cfg = cfg.get("env", {})
    env_backend = str(stage.get("env_backend", env_cfg.get("backend", "fast")))
    env_use_numba = bool(stage.get("env_use_numba", env_cfg.get("use_numba", True)))
    oracle = str(stage.get("oracle", "rl_informed_regular"))
    temperature = float(args.temperature if args.temperature is not None else stage.get("temperature", 0.25))
    epsilon = float(args.epsilon if args.epsilon is not None else stage.get("epsilon", 0.0))
    four_player_prob = float(args.four_player_prob if args.four_player_prob is not None else stage.get("four_player_prob", 0.3))
    reward_mode = str(stage.get("reward_mode", "paired_margin_advantage"))

    checkpoint = args.checkpoint
    if not checkpoint:
        output_dir = Path(stage.get("output_dir", "training2/checkpoints/stage15_proposal"))
        checkpoint = str(output_dir / "latest.pt")
        if not (PROJECT_ROOT / checkpoint).exists() and not Path(checkpoint).is_absolute():
            checkpoint = str(output_dir / "best.pt")
    checkpoint = _resolve_project_path(checkpoint)
    if checkpoint is None or not Path(checkpoint).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")

    times: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    margins: list[float] = []
    baseline_margins: list[float] = []
    advantages: list[float] = []
    game_lengths: list[int] = []
    wins = losses = draws = 0

    with _timer(times, "load_model"):
        model = _make_model(model_cfg, device)
        load_report = _load_checkpoint(model, checkpoint, device)
        model.eval()
    proposal = ProposalConfig(**proposal_cfg)

    if env_use_numba and args.warmup:
        with _timer(times, "numba_warmup"):
            warm = make_orbit_wars_env({"episodeSteps": 8, "seed": args.seed - 1}, backend=env_backend, debug=True, use_numba=True)
            warm.reset(2)
            warm_agents = [make_rulebase_agent(oracle), make_rulebase_agent(oracle)]
            for _ in range(8):
                warm.step([warm_agents[pid](_raw_obs(warm, pid)) or [] for pid in range(2)])
                if all(state.get("status") != "ACTIVE" for state in warm.steps[-1]):
                    break

    total_start = time.perf_counter()
    for game in range(args.games):
        seed = args.seed + game
        rng = random.Random(seed)
        players = 4 if rng.random() < four_player_prob else 2
        model_pid = rng.randrange(players)
        pending_rows = 0

        with _timer(times, "model_game_total"):
            with _timer(times, "model_env_init_reset"):
                env = make_orbit_wars_env(
                    {"episodeSteps": 500, "seed": seed},
                    backend=env_backend,
                    debug=True,
                    use_numba=env_use_numba,
                )
                env.reset(players)
            opponents = {pid: make_rulebase_agent("rl_informed_regular") for pid in range(players)}
            candidate_rulebase = make_rulebase_agent(oracle)

            steps = 0
            for steps in range(500):
                actions = []
                for pid in range(players):
                    obs = _raw_obs(env, pid)
                    if pid != model_pid:
                        with _timer(times, "opponent_rulebase_action"):
                            actions.append(opponents[pid](obs) or [])
                        counts["opponent_rulebase_calls"] += 1
                        continue

                    with _timer(times, "proposal_generation"):
                        extra = proposals_from_model(obs, pid, model, device, proposal)
                    counts["proposal_calls"] += 1
                    counts["proposal_actions"] += len(extra)

                    with _timer(times, "candidate_build"):
                        extra_keys = {_canonical(action) for action in extra}
                        candidates, oracle_idx = build_candidates(
                            obs,
                            candidate_rulebase,
                            max_candidates=max_candidates,
                            extra_candidates=extra,
                        )
                    counts["candidate_build_calls"] += 1
                    counts["candidate_actions"] += len(candidates)
                    if not candidates:
                        actions.append([])
                        continue

                    with _timer(times, "encode_position"):
                        enc = encode_position(obs, pid, candidates, max_candidates=max_candidates)
                    counts["model_decisions"] += 1

                    with _timer(times, "tensor_create"):
                        planet_t = torch.tensor(enc.planet_features, dtype=torch.float32, device=device).unsqueeze(0)
                        global_t = torch.tensor(enc.global_features, dtype=torch.float32, device=device).unsqueeze(0)
                        candidate_t = torch.tensor(enc.candidate_features, dtype=torch.float32, device=device).unsqueeze(0)
                        mask_t = torch.tensor(enc.candidate_mask, dtype=torch.float32, device=device).unsqueeze(0)

                    with _timer(times, "model_forward"):
                        with torch.no_grad():
                            logits, value_pred = model(planet_t, global_t, candidate_t, mask_t)
                            if device.startswith("cuda"):
                                torch.cuda.synchronize()
                            probs = torch.softmax(logits[0] / max(temperature, 1e-6), dim=-1)
                            valid = int(enc.candidate_mask.sum())
                            if epsilon > 0.0 and valid > 0:
                                uniform = torch.zeros_like(probs)
                                uniform[:valid] = 1.0 / float(valid)
                                probs = (1.0 - epsilon) * probs + epsilon * uniform
                            selected_idx = int(torch.multinomial(probs, 1).item())
                    selected_idx = min(selected_idx, len(candidates) - 1)
                    counts["selected_proposal"] += int(_canonical(candidates[selected_idx]) in extra_keys)
                    counts["selected_oracle"] += int(selected_idx == oracle_idx)
                    pending_rows += 1
                    _ = value_pred
                    actions.append(candidates[selected_idx])

                with _timer(times, "model_env_step"):
                    env.step(actions)
                counts["model_env_steps"] += 1
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break

        final_obs = _raw_obs(env, model_pid)
        my_score = _score(final_obs, model_pid)
        opp_best = max(_score(final_obs, pid) for pid in range(players) if pid != model_pid)
        margin = my_score - opp_best
        if args.include_baseline and reward_mode == "paired_margin_advantage":
            with _timer(times, "baseline_total"):
                baseline_margin, _ = _rulebase_margin(
                    seed=seed,
                    players=players,
                    model_pid=model_pid,
                    oracle=oracle,
                    env_backend=env_backend,
                    env_use_numba=env_use_numba,
                    times=times,
                    counts=counts,
                )
        else:
            baseline_margin = 0.0

        margins.append(margin)
        baseline_margins.append(baseline_margin)
        advantages.append(margin - baseline_margin)
        wins += int(margin > 0)
        losses += int(margin < 0)
        draws += int(margin == 0)
        game_lengths.append(steps + 1)
        counts["games"] += 1
        counts["games_4p"] += int(players == 4)
        counts["games_2p"] += int(players == 2)
        counts["pending_rows"] += pending_rows

    total_sec = time.perf_counter() - total_start
    times["profile_total_excluding_load"] = total_sec

    return {
        "config": {
            "stage_key": args.stage_key,
            "checkpoint": checkpoint,
            "device": device,
            "env_backend": env_backend,
            "env_use_numba": env_use_numba,
            "proposal": proposal_cfg,
            "include_baseline": args.include_baseline,
            "games": args.games,
            "seed": args.seed,
            "torch_threads": torch.get_num_threads(),
        },
        "load_report": load_report,
        "times": dict(times),
        "counts": dict(counts),
        "metrics": {
            "win_rate": wins / max(1, wins + losses + draws),
            "wins": wins,
            "losses": losses,
            "draws": draws,
            "avg_margin": float(np.mean(margins)) if margins else 0.0,
            "avg_baseline_margin": float(np.mean(baseline_margins)) if baseline_margins else 0.0,
            "avg_advantage_margin": float(np.mean(advantages)) if advantages else 0.0,
            "avg_game_length": float(np.mean(game_lengths)) if game_lengths else 0.0,
        },
    }


def _print_table(result: dict[str, Any]) -> None:
    times = result["times"]
    counts = result["counts"]
    total = times.get("profile_total_excluding_load", 0.0)
    model_total = times.get("model_game_total", 0.0)
    baseline_total = times.get("baseline_total", 0.0)
    groups = [
        ("model_game_total", total),
        ("  model_env_step", model_total),
        ("  opponent_rulebase_action", model_total),
        ("  candidate_build", model_total),
        ("  proposal_generation", model_total),
        ("  encode_position", model_total),
        ("  tensor_create", model_total),
        ("  model_forward", model_total),
        ("  model_env_init_reset", model_total),
        ("baseline_total", total),
        ("  baseline_rulebase_action", baseline_total),
        ("  baseline_env_step", baseline_total),
        ("  baseline_env_init_reset", baseline_total),
        ("numba_warmup", total),
        ("load_model", total + times.get("load_model", 0.0)),
    ]

    print("\nRollout timing")
    print(f"{'section':32s} {'sec':>10s} {'% parent':>10s} {'ms/decision':>12s} {'ms/envstep':>11s}")
    print("-" * 82)
    decisions = max(1, counts.get("model_decisions", 0))
    env_steps = max(1, counts.get("model_env_steps", 0))
    seen = set()
    for name, parent in groups:
        key = name.strip()
        if key in seen:
            continue
        seen.add(key)
        sec = times.get(key, 0.0)
        pct = 100.0 * sec / parent if parent > 0 else 0.0
        ms_decision = 1000.0 * sec / decisions if key in {
            "candidate_build",
            "proposal_generation",
            "encode_position",
            "tensor_create",
            "model_forward",
        } else 0.0
        ms_step = 1000.0 * sec / env_steps if key in {
            "model_env_step",
            "opponent_rulebase_action",
        } else 0.0
        print(f"{name:32s} {sec:10.3f} {pct:9.1f}% {ms_decision:12.3f} {ms_step:11.3f}")

    metrics = result["metrics"]
    print("\nCounts / metrics")
    for key in [
        "games",
        "games_2p",
        "games_4p",
        "model_decisions",
        "model_env_steps",
        "opponent_rulebase_calls",
        "baseline_env_steps",
        "baseline_rulebase_calls",
        "proposal_actions",
        "candidate_actions",
        "selected_proposal",
        "selected_oracle",
        "pending_rows",
    ]:
        print(f"{key:28s} {counts.get(key, 0)}")
    for key, value in metrics.items():
        print(f"{key:28s} {value:.4f}" if isinstance(value, float) else f"{key:28s} {value}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="training2/config/stage1_tactical_entities_20260519.yaml")
    parser.add_argument("--stage-key", default="stage15")
    parser.add_argument("--checkpoint")
    parser.add_argument("--games", type=int, default=4)
    parser.add_argument("--seed", type=int, default=260519)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, help="Set torch intra/inter-op threads before profiling.")
    parser.add_argument("--four-player-prob", type=float)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--epsilon", type=float)
    parser.add_argument("--include-baseline", dest="include_baseline", action="store_true", default=True)
    parser.add_argument("--no-baseline", dest="include_baseline", action="store_false")
    parser.add_argument("--force-proposals", action="store_true")
    parser.add_argument("--disable-proposals", action="store_true")
    parser.add_argument("--warmup", action="store_true", default=True)
    parser.add_argument("--no-warmup", dest="warmup", action="store_false")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON after the table.")
    args = parser.parse_args()

    result = profile_rollout(args)
    _print_table(result)
    if args.json:
        print("\nJSON")
        print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
