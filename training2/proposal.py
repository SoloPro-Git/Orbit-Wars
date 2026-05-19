"""Model-generated action proposals for hybrid candidate ranking."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from training.expert.action_labeling import infer_target_planet_id
from training2.features import encode_global, encode_planets


@dataclass(frozen=True)
class ProposalConfig:
    enabled: bool = True
    num_candidates: int = 8
    send_threshold: float = 0.45
    max_angle_offset: float = 0.35
    min_ship_ratio: float = 0.08
    ship_ratio_choices: tuple[float, ...] = (0.35, 0.6, 0.9)
    top_targets_per_source: int = 2


def _angle_diff(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def proposal_labels(
    obs: dict,
    player: int,
    action: list[list],
    max_angle_offset: float = 0.35,
) -> dict:
    planets = obs.get("planets", [])
    n = len(planets)
    id_to_idx = {int(p[0]): i for i, p in enumerate(planets)}
    send = [0.0] * n
    target = [0] * n
    ship_ratio = [0.0] * n
    angle_offset = [0.0] * n
    valid = [1.0 if int(p[1]) == player else 0.0 for p in planets]

    best_by_source: dict[int, list] = {}
    for move in action:
        if len(move) < 3:
            continue
        src = int(move[0])
        prev = best_by_source.get(src)
        if prev is None or int(move[2]) > int(prev[2]):
            best_by_source[src] = move

    for src, move in best_by_source.items():
        src_idx = id_to_idx.get(src)
        if src_idx is None:
            continue
        source = planets[src_idx]
        source_ships = max(float(source[5]), 1.0)
        inferred_target_id = infer_target_planet_id(obs, src, float(move[1]), float(move[2]))
        tgt_idx = id_to_idx.get(int(inferred_target_id)) if inferred_target_id is not None else None
        if tgt_idx is None or tgt_idx == src_idx:
            continue
        target_planet = planets[tgt_idx]
        center_angle = math.atan2(float(target_planet[3]) - float(source[3]), float(target_planet[2]) - float(source[2]))
        send[src_idx] = 1.0
        target[src_idx] = int(tgt_idx)
        ship_ratio[src_idx] = min(max(float(move[2]) / source_ships, 0.0), 1.0)
        raw_offset = _angle_diff(float(move[1]) - center_angle)
        angle_offset[src_idx] = min(max(raw_offset / max(max_angle_offset, 1e-6), -1.0), 1.0)

    return {
        "proposal_send": send,
        "proposal_target": target,
        "proposal_ship_ratio": ship_ratio,
        "proposal_angle_offset": angle_offset,
        "proposal_valid": valid,
    }


def proposals_from_model(
    obs: dict,
    player: int,
    model,
    device: torch.device | str,
    cfg: ProposalConfig,
) -> list[list[list]]:
    if not cfg.enabled:
        return []
    planets = obs.get("planets", [])
    if not planets:
        return []

    planet_features = torch.tensor(encode_planets(obs, player), dtype=torch.float32, device=device).unsqueeze(0)
    global_features = torch.tensor(encode_global(obs, player), dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        pred = model.proposal(planet_features, global_features)
    send_prob = torch.sigmoid(pred["send_logits"])[0].detach().cpu()
    target_logits = pred["target_logits"][0].detach().cpu()
    ship_ratio = torch.sigmoid(pred["ship_logits"])[0].detach().cpu()
    angle_offset = pred["angle_offsets"][0].detach().cpu()

    source_indices = [
        i
        for i, p in enumerate(planets)
        if int(p[1]) == player and float(p[5]) >= 1.0 and float(send_prob[i]) >= cfg.send_threshold
    ]
    source_indices.sort(key=lambda i: float(send_prob[i]), reverse=True)
    if not source_indices:
        source_indices = [
            i
            for i, p in sorted(
                enumerate(planets),
                key=lambda row: float(send_prob[row[0]]),
                reverse=True,
            )
            if int(p[1]) == player and float(p[5]) >= 1.0
        ][:2]

    action: list[list] = []
    variants: list[list[list]] = []
    for src_idx in source_indices:
        src = planets[src_idx]
        logits = target_logits[src_idx].clone()
        logits[src_idx] = -1e9
        top_targets = torch.topk(logits, k=min(cfg.top_targets_per_source, len(planets))).indices.tolist()
        target_idx = int(top_targets[0])
        target = planets[target_idx]
        base_angle = math.atan2(float(target[3]) - float(src[3]), float(target[2]) - float(src[2]))
        angle = base_angle + float(angle_offset[src_idx]) * cfg.max_angle_offset
        ratio = max(float(ship_ratio[src_idx]), cfg.min_ship_ratio)
        ships = max(1, min(int(float(src[5])), int(float(src[5]) * ratio)))
        if ships > 0:
            action.append([int(src[0]), float(angle), int(ships)])

        for choice in cfg.ship_ratio_choices:
            ships_variant = max(1, min(int(float(src[5])), int(float(src[5]) * choice)))
            variants.append([[int(src[0]), float(angle), int(ships_variant)]])
        for alt_idx in top_targets[1:]:
            alt = planets[int(alt_idx)]
            alt_angle = math.atan2(float(alt[3]) - float(src[3]), float(alt[2]) - float(src[2]))
            variants.append([[int(src[0]), float(alt_angle), ships]])

    out: list[list[list]] = []
    if action:
        out.append(action)
    out.extend(variants)
    return out[: cfg.num_candidates]
