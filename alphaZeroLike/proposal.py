"""Model-owned action proposal head.

The model predicts source selection, target planet, and ship ratio.  Angles are
computed by deterministic intercept rules so the policy does not need to learn
raw radians.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from alphaZeroLike.features import encode_global, encode_planets
from training.expert.action_labeling import infer_target_planet_id


@dataclass(frozen=True)
class ProposalConfig:
    enabled: bool = True
    num_candidates: int = 64
    num_full_actions: int = 16
    num_sampled_actions: int = 48
    raw_sampled_actions: int = 128
    send_threshold: float = 0.30
    min_ship_ratio: float = 0.08
    ship_ratio_choices: tuple[float, ...] = (0.25, 0.4, 0.6, 0.85)
    top_targets_per_source: int = 2
    max_sources: int = 4
    source_temperature: float = 1.0
    target_temperature: float = 1.0
    ship_noise_std: float = 0.12
    sample_source_prob: float = 0.65


def _fleet_speed(ships: float, max_speed: float = 6.0) -> float:
    if ships <= 1:
        return 1.0
    ratio = math.log(max(ships, 1.0)) / math.log(1000.0)
    ratio = min(max(ratio, 0.0), 1.0)
    return 1.0 + (max_speed - 1.0) * (ratio**1.5)


def _is_orbital_target(obs: dict, planet: list) -> bool:
    if int(planet[0]) in set(obs.get("comet_planet_ids", [])):
        return False
    return math.hypot(float(planet[2]) - 50.0, float(planet[3]) - 50.0) + float(planet[4]) < 50.0


def _comet_future_position(obs: dict, planet_id: int, eta: int) -> tuple[float, float] | None:
    for group in obs.get("comets", []) or []:
        planet_ids = group.get("planet_ids", []) if isinstance(group, dict) else []
        if planet_id not in planet_ids:
            continue
        try:
            idx = list(planet_ids).index(planet_id)
            path = group.get("paths", [])[idx]
            path_index = int(group.get("path_index", 0))
            future_index = min(max(0, path_index + max(0, eta)), len(path) - 1)
            x, y = path[future_index]
            return float(x), float(y)
        except (IndexError, TypeError, ValueError):
            return None
    return None


def _future_target_position(obs: dict, target: list, eta: int) -> tuple[float, float]:
    comet_pos = _comet_future_position(obs, int(target[0]), eta)
    if comet_pos is not None:
        return comet_pos
    if _is_orbital_target(obs, target):
        angle = math.atan2(float(target[3]) - 50.0, float(target[2]) - 50.0)
        radius = math.hypot(float(target[2]) - 50.0, float(target[3]) - 50.0)
        angle += float(obs.get("angular_velocity", 0.0)) * max(0, eta)
        return 50.0 + math.cos(angle) * radius, 50.0 + math.sin(angle) * radius
    return float(target[2]), float(target[3])


def aim_angle(obs: dict, source: list, target: list, ships: int) -> float:
    speed = _fleet_speed(max(1, ships))
    tx = float(target[2])
    ty = float(target[3])
    for _ in range(3):
        dist = math.hypot(tx - float(source[2]), ty - float(source[3]))
        eta = int(math.ceil(dist / max(speed, 1e-6)))
        tx, ty = _future_target_position(obs, target, eta)
    return math.atan2(ty - float(source[3]), tx - float(source[2]))


def proposal_labels(obs: dict, player: int, action: list[list]) -> dict:
    planets = obs.get("planets", [])
    n = len(planets)
    id_to_idx = {int(p[0]): i for i, p in enumerate(planets)}
    send = [0.0] * n
    target = [0] * n
    ship_ratio = [0.0] * n
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
        target_id = infer_target_planet_id(obs, src, float(move[1]), float(move[2]))
        tgt_idx = id_to_idx.get(int(target_id)) if target_id is not None else None
        if tgt_idx is None or tgt_idx == src_idx:
            continue
        send[src_idx] = 1.0
        target[src_idx] = int(tgt_idx)
        ship_ratio[src_idx] = min(max(float(move[2]) / max(float(source[5]), 1.0), 0.0), 1.0)

    return {
        "proposal_send": send,
        "proposal_target": target,
        "proposal_ship_ratio": ship_ratio,
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
    source_mask = torch.zeros((1, planet_features.size(1)), dtype=torch.bool, device=device)
    target_mask = torch.zeros((1, planet_features.size(1)), dtype=torch.bool, device=device)
    for i, planet in enumerate(planets[: planet_features.size(1)]):
        target_mask[0, i] = True
        source_mask[0, i] = int(planet[1]) == player and float(planet[5]) >= 1.0
    with torch.no_grad():
        pred = model.proposal(planet_features, global_features, source_mask=source_mask, target_mask=target_mask)
    send_prob = torch.sigmoid(pred["send_logits"])[0].detach().cpu()
    target_logits = pred["target_logits"][0].detach().cpu()
    ship_ratio = torch.sigmoid(pred["ship_logits"])[0].detach().cpu()
    valid_source_indices = [
        i for i, p in enumerate(planets) if int(p[1]) == player and float(p[5]) >= 1.0 and i < len(send_prob)
    ]

    sources = [
        i
        for i in valid_source_indices
        if float(send_prob[i]) >= cfg.send_threshold
    ]
    sources.sort(key=lambda i: float(send_prob[i]), reverse=True)
    if not sources:
        sources = [
            i
            for i, p in sorted(enumerate(planets), key=lambda row: float(send_prob[row[0]]), reverse=True)
            if int(p[1]) == player and float(p[5]) >= 1.0
        ][:2]
    sources = sources[: cfg.max_sources]

    per_source_moves: list[list[list]] = []
    base_action: list[list] = []
    variants: list[list[list]] = []
    for src_idx in sources:
        src = planets[src_idx]
        logits = target_logits[src_idx].clone()
        if len(logits) > len(planets):
            logits[len(planets) :] = -1e9
        logits[src_idx] = -1e9
        top_targets = torch.topk(logits, k=min(cfg.top_targets_per_source, len(planets))).indices.tolist()
        ratio = max(float(ship_ratio[src_idx]), cfg.min_ship_ratio)
        moves_for_source: list[list] = []
        for tpos, target_idx in enumerate(top_targets):
            target = planets[int(target_idx)]
            ships = max(1, min(int(float(src[5])), int(float(src[5]) * ratio)))
            angle = aim_angle(obs, src, target, ships)
            move = [int(src[0]), float(angle), int(ships)]
            if tpos == 0:
                base_action.append(move)
            moves_for_source.append(move)
            variants.append([move])
            for choice in cfg.ship_ratio_choices:
                ships_variant = max(1, min(int(float(src[5])), int(float(src[5]) * choice)))
                variant_angle = aim_angle(obs, src, target, ships_variant)
                variant = [int(src[0]), float(variant_angle), int(ships_variant)]
                moves_for_source.append(variant)
                variants.append([variant])
        per_source_moves.append(moves_for_source)

    out: list[list[list]] = []
    seen: set[tuple] = set()

    def canonical(action: list[list]) -> tuple:
        return tuple((int(a[0]), round(float(a[1]), 6), int(a[2])) for a in action if len(a) >= 3)

    def add(action: list[list]) -> None:
        key = canonical(action)
        if key in seen:
            return
        seen.add(key)
        out.append(action)

    if base_action:
        add(base_action)

    # Full multi-source alternatives: keep the same selected sources, but vary
    # target rank and ship-ratio choice coherently across sources.
    for alt_idx in range(max(0, cfg.num_full_actions - len(out))):
        action: list[list] = []
        for moves_for_source in per_source_moves:
            if not moves_for_source:
                continue
            action.append(moves_for_source[alt_idx % len(moves_for_source)])
        if action:
            add(action)

    # Source-subset actions help search choose partial launches instead of only
    # all-or-nothing plans.
    for prefix_len in range(1, len(base_action)):
        add(base_action[:prefix_len])
    for drop_idx in range(len(base_action)):
        add([move for i, move in enumerate(base_action) if i != drop_idx])

    def sample_from_weights(indices: list[int], weights: list[float], k: int) -> list[int]:
        selected: list[int] = []
        pool = list(indices)
        w = [max(float(x), 1e-6) for x in weights]
        for _ in range(min(k, len(pool))):
            total = sum(w)
            if total <= 0:
                break
            r = torch.rand(1).item() * total
            acc = 0.0
            chosen = 0
            for j, weight in enumerate(w):
                acc += weight
                if acc >= r:
                    chosen = j
                    break
            selected.append(pool.pop(chosen))
            w.pop(chosen)
        return selected

    source_temp = max(float(cfg.source_temperature), 1e-3)
    target_temp = max(float(cfg.target_temperature), 1e-3)
    sampled: list[list[list]] = []
    for _ in range(max(int(cfg.num_sampled_actions), int(cfg.raw_sampled_actions))):
        if not valid_source_indices:
            break
        source_weights = [min(max(float(send_prob[i]), 1e-4), 1.0) ** (1.0 / source_temp) for i in valid_source_indices]
        sampled_sources = [
            idx
            for idx in sample_from_weights(valid_source_indices, source_weights, cfg.max_sources)
            if torch.rand(1).item() < max(min(float(cfg.sample_source_prob), 1.0), 0.0) or float(send_prob[idx]) >= cfg.send_threshold
        ]
        if not sampled_sources:
            sampled_sources = sample_from_weights(valid_source_indices, source_weights, 1)
        action: list[list] = []
        for src_idx in sampled_sources:
            src = planets[src_idx]
            logits = target_logits[src_idx].clone()
            if len(logits) > len(planets):
                logits[len(planets) :] = -1e9
            logits[src_idx] = -1e9
            probs = torch.softmax(logits[: len(planets)] / target_temp, dim=-1)
            if not torch.isfinite(probs).all() or float(probs.sum()) <= 0.0:
                continue
            target_idx = int(torch.multinomial(probs, num_samples=1).item())
            if target_idx == src_idx or target_idx >= len(planets):
                continue
            ratio = float(ship_ratio[src_idx])
            if cfg.ship_noise_std > 0:
                ratio += float(torch.randn(1).item()) * float(cfg.ship_noise_std)
            ratio = min(max(ratio, cfg.min_ship_ratio), 1.0)
            ships = max(1, min(int(float(src[5])), int(float(src[5]) * ratio)))
            target = planets[target_idx]
            action.append([int(src[0]), float(aim_angle(obs, src, target, ships)), int(ships)])
        if action:
            sampled.append(action)

    # Keep stochastic proposals before single-move variants so the retained
    # candidate set has real exploratory multi-source actions when truncated.
    for action in sampled:
        add(action)

    out.extend(variants)

    deduped: list[list[list]] = []
    seen.clear()
    for action in out:
        key = canonical(action)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(action)
    return deduped[: cfg.num_candidates]
