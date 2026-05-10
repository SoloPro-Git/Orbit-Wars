"""Action decoding module for Orbit Wars RL training.

Converts neural network outputs to Kaggle game action format:
[[from_planet_id, direction_angle, num_ships], ...]
"""
from __future__ import annotations

import math

import numpy as np
import torch


def decode_actions(
    target_logits: np.ndarray,  # [N_owned, N_all_planets]
    num_ships_raw: np.ndarray,  # [N_owned, 1] sigmoid output (0~1)
    owned_planets: list[dict],  # [{id, x, y, ships, ...}, ...]
    all_planets: list[dict],  # all planet info
    threshold: float = 0.5,  # launch threshold
) -> list[list]:
    """Convert network outputs to Kaggle action format [[from_id, angle, ships], ...].

    For each owned planet:
    - argmax target_logits to get target planet index
    - compute angle via atan2
    - scale num_ships_raw by source planet's ship count
    - skip if below threshold or no ships available
    """
    actions: list[list] = []

    for i, source in enumerate(owned_planets):
        # No ships to send
        source_ships = source.get("ships", 0)
        if source_ships <= 0:
            continue

        # Determine target planet
        target_idx = int(np.argmax(target_logits[i]))

        # 边界检查
        if target_idx >= len(all_planets) or target_idx < 0:
            continue

        target = all_planets[target_idx]

        # Compute angle from source to target
        dx = target["x"] - source["x"]
        dy = target["y"] - source["y"]
        angle = math.atan2(dy, dx)

        # Scale ship count
        num_ships = float(num_ships_raw[i, 0]) * source_ships

        # Skip if below threshold
        if num_ships < threshold:
            continue

        # Ensure at least 1 ship
        num_ships_int = max(1, int(num_ships))
        # Cannot send more than we have
        num_ships_int = min(num_ships_int, int(source_ships))

        if num_ships_int <= 0:
            continue

        actions.append([source["id"], angle, num_ships_int])

    return actions


def decode_actions_batch(
    target_logits: np.ndarray,  # [batch, N_owned, N_all_planets]
    num_ships_raw: np.ndarray,  # [batch, N_owned, 1]
    owned_planets_batch: list[list[dict]],
    all_planets_batch: list[list[dict]],
    threshold: float = 0.5,
) -> list[list[list]]:
    """Batch version of decode_actions, returns batch-sized list of action lists."""
    batch_size = target_logits.shape[0]
    results: list[list[list]] = []

    for b in range(batch_size):
        actions = decode_actions(
            target_logits=target_logits[b],
            num_ships_raw=num_ships_raw[b],
            owned_planets=owned_planets_batch[b],
            all_planets=all_planets_batch[b],
            threshold=threshold,
        )
        results.append(actions)

    return results


def actions_to_kaggle(actions: list[list]) -> list[list]:
    """Validate and fix action format to ensure legality.

    Checks:
    - num_ships >= 1 (integer)
    - angle in [-pi, pi] range
    - filters out invalid actions
    """
    valid_actions: list[list] = []

    for action in actions:
        if len(action) != 3:
            continue

        from_id, angle, num_ships = action

        # Validate planet ID is integer
        from_id = int(from_id)

        # Validate angle range
        angle = float(angle)
        # Normalize to [-pi, pi]
        angle = math.atan2(math.sin(angle), math.cos(angle))

        # Validate ship count
        num_ships = int(num_ships)
        if num_ships < 1:
            continue

        # Check angle is finite
        if not math.isfinite(angle):
            continue

        valid_actions.append([from_id, angle, num_ships])

    return valid_actions


def sample_actions(
    target_logits: np.ndarray,  # [N_owned, N_all_planets]
    num_ships_raw: np.ndarray,  # [N_owned, 1] sigmoid output
    temperature: float = 1.0,
    return_log_probs: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample actions from policy distribution (for training).

    Uses Gumbel-Softmax trick for target sampling and adds Gaussian noise
    for ship count exploration.

    Returns:
        (target_indices, num_ships) - target_indices: [N_owned], num_ships: [N_owned, 1]
    """
    n_owned, n_planets = target_logits.shape

    # Gumbel-Max sampling for targets
    if temperature <= 0:
        target_indices = np.argmax(target_logits, axis=-1)
        logits_max = target_logits.max(axis=-1, keepdims=True)
        stable_logits = target_logits - logits_max
        log_softmax = stable_logits - np.log(np.exp(stable_logits).sum(axis=-1, keepdims=True) + 1e-10)
        target_log_prob = log_softmax[np.arange(n_owned), target_indices]
    else:
        # Generate Gumbel noise
        gumbel_noise = -np.log(-np.log(np.random.uniform(0, 1, size=target_logits.shape) + 1e-20) + 1e-20)
        # Apply temperature scaling + Gumbel noise, then argmax
        perturbed_logits = target_logits / temperature + gumbel_noise
        target_indices = np.argmax(perturbed_logits, axis=-1)
        scaled = target_logits / max(temperature, 1e-6)
        scaled_max = scaled.max(axis=-1, keepdims=True)
        log_softmax = scaled - scaled_max - np.log(np.exp(scaled - scaled_max).sum(axis=-1, keepdims=True) + 1e-10)
        target_log_prob = log_softmax[np.arange(n_owned), target_indices]

    # Add Gaussian noise for ship count exploration
    ship_noise = np.random.normal(0, 0.1, size=num_ships_raw.shape)
    num_ships = np.clip(num_ships_raw + ship_noise, 0.0, 1.0)

    if not return_log_probs:
        return target_indices, num_ships

    sigma = 0.15
    ship_log_prob = -0.5 * (((num_ships - num_ships_raw) / sigma) ** 2).squeeze(-1)
    total_log_prob = target_log_prob + ship_log_prob
    return target_indices, num_ships, total_log_prob


def compute_action_log_probs(
    target_logits: torch.Tensor,  # [batch, N_owned, N_planets]
    num_ships_pred: torch.Tensor,  # [batch, N_owned, 1]
    target_indices: torch.Tensor,  # [batch, N_owned] actual chosen targets
    num_ships_actual: torch.Tensor,  # [batch, N_owned, 1] actual ship counts
) -> torch.Tensor:
    """Compute log probability of actions for PPO loss.

    Target log_prob: CrossEntropyLoss(reduction='none') -> -log_prob of correct class
    Ship log_prob: Gaussian distribution, log_prob = -0.5 * ((actual - pred) / sigma)^2

    Returns:
        Total log_prob [batch, N_owned]
    """
    batch_size, n_owned, n_planets = target_logits.shape

    # --- Target log probability ---
    # CrossEntropyLoss expects [N, C] logits and [N] targets
    logits_flat = target_logits.reshape(-1, n_planets)  # [batch*N_owned, N_planets]
    targets_flat = target_indices.reshape(-1).long()  # [batch*N_owned]

    # CrossEntropyLoss computes log_softmax internally,
    # nll_loss gives -log_prob of the correct class
    ce_loss = torch.nn.CrossEntropyLoss(reduction="none")
    target_neg_log_prob = ce_loss(logits_flat, targets_flat)  # [batch*N_owned]
    target_log_prob = -target_neg_log_prob.reshape(batch_size, n_owned)  # [batch, N_owned]

    # --- Ship count log probability ---
    # Assume Gaussian distribution with fixed sigma=0.3 for exploration
    sigma = 0.3
    ship_diff = num_ships_actual - num_ships_pred  # [batch, N_owned, 1]
    ship_log_prob = -0.5 * (ship_diff / sigma) ** 2  # [batch, N_owned, 1]
    ship_log_prob = ship_log_prob.squeeze(-1)  # [batch, N_owned]

    # --- Combined log probability ---
    total_log_prob = target_log_prob + ship_log_prob  # [batch, N_owned]

    return total_log_prob
