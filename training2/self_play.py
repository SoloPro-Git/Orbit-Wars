"""AlphaZero-like self-play improvement over rulebase candidates.

This is policy iteration rather than raw PPO: each state is expanded into
rulebase-derived candidates, the model samples/reranks candidates, and the
final game result trains the policy/value model.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F
from kaggle_environments import make

from training2.batching import pad_planets
from training2.candidates import build_candidates
from training2.features import encode_position, result_value
from training2.model import CandidatePolicyValueNet, load_compatible_state_dict
from training2.rulebase_bridge import make_rulebase_agent


def _obs(env, pid: int) -> dict:
    return env.steps[-1][pid]["observation"]


def _train_batch(model, opt, rows, device: str) -> dict:
    planets = pad_planets([r["planets"] for r in rows]).to(device)
    glob = torch.tensor([r["global"] for r in rows], dtype=torch.float32, device=device)
    candidates = torch.tensor([r["candidates"] for r in rows], dtype=torch.float32, device=device)
    mask = torch.tensor([r["candidate_mask"] for r in rows], dtype=torch.float32, device=device)
    target = torch.tensor([r["target"] for r in rows], dtype=torch.long, device=device)
    value_target = torch.tensor([r["value"] for r in rows], dtype=torch.float32, device=device)
    logits, value = model(planets, glob, candidates, mask)
    loss = F.cross_entropy(logits, target) + 0.5 * F.mse_loss(value, value_target)
    opt.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    return {"loss": float(loss.item()), "acc": float((logits.argmax(-1) == target).float().mean().item())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint")
    parser.add_argument("--out", default="training2/checkpoints/selfplay.pt")
    parser.add_argument("--games", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model = CandidatePolicyValueNet().to(args.device)
    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device, weights_only=False)
        load_compatible_state_dict(model, ckpt["model_state_dict"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    replay: list[dict] = []

    for it in range(args.iterations):
        new_rows: list[dict] = []
        for game in range(args.games):
            env = make("orbit_wars", configuration={"episodeSteps": 500, "seed": it * 100000 + game}, debug=True)
            env.reset(4)
            rulebases = {pid: make_rulebase_agent() for pid in range(4)}
            pending: list[dict] = []
            for _ in range(500):
                actions = []
                for pid in range(4):
                    obs = _obs(env, pid)
                    candidates, _ = build_candidates(obs, rulebases[pid])
                    enc = encode_position(obs, pid, candidates)
                    with torch.no_grad():
                        logits, _ = model(
                            torch.tensor(enc.planet_features, dtype=torch.float32, device=args.device).unsqueeze(0),
                            torch.tensor(enc.global_features, dtype=torch.float32, device=args.device).unsqueeze(0),
                            torch.tensor(enc.candidate_features, dtype=torch.float32, device=args.device).unsqueeze(0),
                            torch.tensor(enc.candidate_mask, dtype=torch.float32, device=args.device).unsqueeze(0),
                        )
                        probs = torch.softmax(logits[0] / max(args.temperature, 1e-6), dim=-1)
                        idx = int(torch.multinomial(probs, 1).item())
                    idx = min(idx, len(candidates) - 1)
                    pending.append(
                        {
                            "player": pid,
                            "planets": enc.planet_features.tolist(),
                            "global": enc.global_features.tolist(),
                            "candidates": enc.candidate_features.tolist(),
                            "candidate_mask": enc.candidate_mask.tolist(),
                            "target": idx,
                        }
                    )
                    actions.append(candidates[idx])
                env.step(actions)
                if all(state.get("status") != "ACTIVE" for state in env.steps[-1]):
                    break
            finals = [_obs(env, pid) for pid in range(4)]
            for row in pending:
                row["value"] = result_value(finals[row["player"]], row["player"])
            new_rows.extend(pending)

        replay.extend(new_rows)
        replay = replay[-200000:]
        model.train()
        metrics = {"loss": 0.0, "acc": 0.0}
        updates = max(1, min(200, len(replay) // args.batch_size))
        for u in range(updates):
            start = (u * args.batch_size) % max(1, len(replay) - args.batch_size + 1)
            batch = replay[start : start + args.batch_size]
            m = _train_batch(model, opt, batch, args.device)
            metrics["loss"] += m["loss"]
            metrics["acc"] += m["acc"]
        metrics = {k: v / updates for k, v in metrics.items()}
        print(json.dumps({"iteration": it, "new_rows": len(new_rows), **metrics}), flush=True)

        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": model.state_dict(), "iteration": it}, out)


if __name__ == "__main__":
    main()
