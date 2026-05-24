from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PPOConfig:
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.02
    learning_rate: float = 3e-4
    max_grad_norm: float = 0.5
    min_epochs: int = 1
    epochs: int = 4
    batch_size: int = 256
    target_kl: float = 0.0


class RolloutBuffer:
    def __init__(self):
        self.rows: list[dict] = []

    def add(self, **row) -> None:
        self.rows.append(row)

    def clear(self) -> None:
        self.rows.clear()

    def __len__(self) -> int:
        return len(self.rows)


def compute_gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray, gamma: float, lam: float) -> tuple[np.ndarray, np.ndarray]:
    adv = np.zeros_like(rewards, dtype=np.float32)
    last = 0.0
    for t in range(len(rewards) - 1, -1, -1):
        next_value = 0.0 if t == len(rewards) - 1 else values[t + 1]
        non_terminal = 1.0 - float(dones[t])
        delta = rewards[t] + gamma * next_value * non_terminal - values[t]
        last = delta + gamma * lam * non_terminal * last
        adv[t] = last
    returns = adv + values
    return adv, returns.astype(np.float32)


def action_log_prob_entropy(
    out: dict[str, torch.Tensor],
    launch_actions: torch.Tensor,
    target_actions: torch.Tensor,
    ship_actions: torch.Tensor,
    own_mask: torch.Tensor,
    launch_mask: torch.Tensor,
    target_safety_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    action_count = launch_actions.size(-1)
    source_logits = torch.nan_to_num(out["source_logits"], nan=0.0, posinf=1e9, neginf=-1e9)
    target_logits = torch.nan_to_num(out["target_logits"], nan=-1e9, posinf=1e9, neginf=-1e9)
    base_target_logits = target_logits
    if target_safety_mask is not None:
        safe_sources = target_safety_mask.any(dim=-1)
        target_logits = target_logits.clone().masked_fill(~target_safety_mask[:, :, None, :], -1e9)
        if (~safe_sources).any():
            target_logits[~safe_sources] = base_target_logits[~safe_sources]
            source_logits = source_logits.clone()
            source_logits[~safe_sources, :, 1] = -1e9
    source_dist = torch.distributions.Categorical(logits=source_logits)
    target_dist = torch.distributions.Categorical(logits=target_logits)
    ship_params = out["ship_params"]
    gather_idx = target_actions[:, :, :, None, None].expand(-1, -1, -1, 1, ship_params.size(-1))
    chosen_ship_params = ship_params.gather(dim=3, index=gather_idx).squeeze(3)
    ship_dist = torch.distributions.Beta(chosen_ship_params[..., 0], chosen_ship_params[..., 1])

    own_slot_mask = own_mask[:, :, None]
    decision_mask = own_slot_mask.expand(-1, -1, action_count)
    source_lp = source_dist.log_prob(launch_actions).masked_fill(~decision_mask, 0.0).sum(dim=(1, 2))
    target_lp = target_dist.log_prob(target_actions).masked_fill(~launch_mask, 0.0).sum(dim=1)
    target_lp = target_lp.sum(dim=1)
    ship_lp = ship_dist.log_prob(ship_actions.clamp(1e-4, 1.0 - 1e-4)).masked_fill(~launch_mask, 0.0).sum(dim=(1, 2))

    source_ent = source_dist.entropy().masked_fill(~decision_mask, 0.0).sum(dim=(1, 2))
    target_ent = target_dist.entropy().masked_fill(~launch_mask, 0.0).sum(dim=(1, 2))
    ship_ent = ship_dist.entropy().masked_fill(~launch_mask, 0.0).sum(dim=(1, 2))
    denom = decision_mask.sum(dim=(1, 2)).clamp_min(1)
    entropy = (source_ent + target_ent + ship_ent) / denom
    return source_lp + target_lp + ship_lp, entropy


class PPOUpdater:
    def __init__(self, model: nn.Module, cfg: PPOConfig, device: str = "cpu"):
        self.model = model
        self.cfg = cfg
        self.device = torch.device(device)
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, eps=1e-5)

    def update(self, buffer: RolloutBuffer) -> dict[str, float]:
        if len(buffer) == 0:
            return {}
        rewards = np.asarray([r["reward"] for r in buffer.rows], dtype=np.float32)
        values = np.asarray([r["value"] for r in buffer.rows], dtype=np.float32)
        dones = np.asarray([r["done"] for r in buffer.rows], dtype=np.float32)
        adv, returns = compute_gae(rewards, values, dones, self.cfg.gamma, self.cfg.gae_lambda)
        if adv.std() > 1e-6:
            adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        tensors = {
            "planets": torch.nan_to_num(torch.tensor(np.stack([r["planets"] for r in buffer.rows]), dtype=torch.float32, device=self.device)),
            "pair_features": torch.nan_to_num(
                torch.tensor(np.stack([r["pair_features"] for r in buffer.rows]), dtype=torch.float32, device=self.device)
            ),
            "global_features": torch.nan_to_num(
                torch.tensor(np.stack([r["global_features"] for r in buffer.rows]), dtype=torch.float32, device=self.device)
            ),
            "planet_mask": torch.tensor(np.stack([r["planet_mask"] for r in buffer.rows]), dtype=torch.bool, device=self.device),
            "own_mask": torch.tensor(np.stack([r["own_mask"] for r in buffer.rows]), dtype=torch.bool, device=self.device),
            "launch_actions": torch.tensor(np.stack([r["launch_actions"] for r in buffer.rows]), dtype=torch.long, device=self.device),
            "target_actions": torch.tensor(np.stack([r["target_actions"] for r in buffer.rows]), dtype=torch.long, device=self.device),
            "ship_actions": torch.tensor(np.stack([r["ship_actions"] for r in buffer.rows]), dtype=torch.float32, device=self.device),
            "launch_mask": torch.tensor(np.stack([r["launch_mask"] for r in buffer.rows]), dtype=torch.bool, device=self.device),
            "target_safety_mask": torch.tensor(
                np.stack(
                    [
                        r.get("target_safety_mask", np.ones((r["planet_mask"].shape[0], r["planet_mask"].shape[0]), dtype=np.bool_))
                        for r in buffer.rows
                    ]
                ),
                dtype=torch.bool,
                device=self.device,
            ),
            "old_logprob": torch.tensor([r["logprob"] for r in buffer.rows], dtype=torch.float32, device=self.device),
            "advantages": torch.tensor(adv, dtype=torch.float32, device=self.device),
            "returns": torch.tensor(returns, dtype=torch.float32, device=self.device),
            "weights": torch.tensor([r.get("replay_weight", 1.0) for r in buffer.rows], dtype=torch.float32, device=self.device),
        }

        n = len(buffer)
        batch_size = min(self.cfg.batch_size, n)
        metrics: dict[str, list[float]] = {"loss": [], "policy_loss": [], "value_loss": [], "entropy": [], "clip_frac": [], "approx_kl": []}
        epochs_used = 0
        update_steps = 0
        skipped_steps = 0
        early_stop = 0.0
        min_epochs = max(1, min(int(self.cfg.min_epochs), int(self.cfg.epochs)))
        max_epochs = max(min_epochs, int(self.cfg.epochs))
        for _epoch in range(max_epochs):
            order = torch.randperm(n, device=self.device)
            epoch_kl: list[float] = []
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                out = self.model(
                    tensors["planets"][idx],
                    tensors["pair_features"][idx],
                    tensors["global_features"][idx],
                    tensors["planet_mask"][idx],
                    tensors["own_mask"][idx],
                )
                new_logprob, entropy = action_log_prob_entropy(
                    out,
                    tensors["launch_actions"][idx],
                    tensors["target_actions"][idx],
                    tensors["ship_actions"][idx],
                    tensors["own_mask"][idx],
                    tensors["launch_mask"][idx],
                    tensors["target_safety_mask"][idx],
                )
                raw_logratio = torch.nan_to_num(new_logprob - tensors["old_logprob"][idx], nan=0.0, posinf=20.0, neginf=-20.0)
                logratio = raw_logratio.clamp(-20.0, 20.0)
                ratio = logratio.exp()
                adv_b = tensors["advantages"][idx]
                weight_b = tensors["weights"][idx].clamp_min(0.0)
                weight_norm = weight_b.sum().clamp_min(1e-6)
                pg1 = -adv_b * ratio
                pg2 = -adv_b * torch.clamp(ratio, 1.0 - self.cfg.clip_coef, 1.0 + self.cfg.clip_coef)
                policy_loss = (torch.max(pg1, pg2) * weight_b).sum() / weight_norm
                value_loss = (((out["value"] - tensors["returns"][idx]) ** 2) * weight_b).sum() / weight_norm
                entropy_loss = (entropy * weight_b).sum() / weight_norm
                loss = policy_loss + self.cfg.value_coef * value_loss - self.cfg.entropy_coef * entropy_loss
                if not torch.isfinite(loss):
                    skipped_steps += 1
                    continue

                self.optimizer.zero_grad(set_to_none=True)
                loss.backward()
                grad_norm = nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.max_grad_norm)
                if not torch.isfinite(grad_norm):
                    self.optimizer.zero_grad(set_to_none=True)
                    skipped_steps += 1
                    continue
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac = ((ratio - 1.0).abs() > self.cfg.clip_coef).float().mean()
                    approx_kl = ((ratio - 1.0) - logratio).mean()
                metrics["loss"].append(float(loss.item()))
                metrics["policy_loss"].append(float(policy_loss.item()))
                metrics["value_loss"].append(float(value_loss.item()))
                metrics["entropy"].append(float(entropy_loss.item()))
                metrics["clip_frac"].append(float(clip_frac.item()))
                kl_value = float(approx_kl.item())
                metrics["approx_kl"].append(kl_value)
                epoch_kl.append(kl_value)
                update_steps += 1
            epochs_used += 1
            if epochs_used >= min_epochs and self.cfg.target_kl > 0.0 and epoch_kl and float(np.mean(epoch_kl)) > 1.5 * self.cfg.target_kl:
                early_stop = 1.0
                break
        summary = {key: float(np.mean(vals)) for key, vals in metrics.items() if vals}
        summary["epochs_used"] = float(epochs_used)
        summary["update_steps"] = float(update_steps)
        summary["kl_early_stop"] = early_stop
        summary["skipped_steps"] = float(skipped_steps)
        summary["min_epochs"] = float(min_epochs)
        summary["max_epochs"] = float(max_epochs)
        summary["target_kl"] = float(self.cfg.target_kl)
        return summary
