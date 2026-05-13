"""Kaggle submission agent using combined rulebase strategies from baseline ablation notebook."""

import math

CENTER_X = 50.0
CENTER_Y = 50.0
SUN_RADIUS = 10.0
MAX_SPEED = 6.0


def fleet_speed(num_ships, max_speed=MAX_SPEED):
    if num_ships <= 1:
        return 1.0
    ratio = math.log(num_ships) / math.log(1000)
    ratio = max(0.0, min(1.0, ratio))
    return 1.0 + (max_speed - 1.0) * (ratio ** 1.5)


def split_planets(obs):
    player = obs["player"]
    planets = obs["planets"]
    my_planets = [p for p in planets if p[1] == player]
    targets = [p for p in planets if p[1] != player]
    return my_planets, targets


def get_nearest_target(mine, targets):
    return min(targets, key=lambda t: math.hypot(mine[2] - t[2], mine[3] - t[3]))


def get_angle(source, target):
    return math.atan2(target[3] - source[3], target[2] - source[2])


def get_best_value_target(mine, targets, score_type="prod_dist", eps=1e-6):
    def score(t):
        dist = math.hypot(mine[2] - t[2], mine[3] - t[3])
        ships = t[5]
        prod = t[6]
        if score_type == "prod_dist":
            return prod / (dist + eps)
        if score_type == "prod_ship_dist":
            return prod / ((ships + 1) * (dist + eps))
        if score_type == "prod_over_ships":
            return prod / (ships + 1)
        if score_type == "nearest":
            return -dist
        return prod / (dist + eps)

    return max(targets, key=score)


def get_reserved_targets(obs, angle_threshold=0.1, use_capture_filter=False):
    planets = obs["planets"]
    fleets = obs["fleets"]
    player = obs["player"]
    reserved_targets = set()
    my_fleets = [f for f in fleets if f[1] == player]

    for f in my_fleets:
        fx, fy = f[2], f[3]
        angle = f[4]
        best_planet = None
        best_angle_diff = float("inf")

        for p in planets:
            if p[0] == f[5]:
                continue
            dx = p[2] - fx
            dy = p[3] - fy
            target_angle = math.atan2(dy, dx)
            diff = abs(math.atan2(math.sin(target_angle - angle), math.cos(target_angle - angle)))
            if diff < best_angle_diff:
                best_angle_diff = diff
                best_planet = p

        if best_planet is None or best_angle_diff >= angle_threshold:
            continue
        if use_capture_filter and f[6] <= best_planet[5]:
            continue

        reserved_targets.add(best_planet[0])

    return reserved_targets


def estimate_target_defense(source, target, ships_to_send):
    distance = math.hypot(source[2] - target[2], source[3] - target[3])
    speed = fleet_speed(max(1, ships_to_send))
    arrival_turns = distance / speed
    estimated_defense = target[5] + target[6] * arrival_turns
    return math.ceil(estimated_defense)


def estimate_enemy_incoming_to_target(obs, target, angle_threshold=0.1):
    fleets = obs["fleets"]
    player = obs["player"]
    incoming = 0

    for f in fleets:
        if f[1] == player:
            continue
        fx, fy = f[2], f[3]
        angle = f[4]
        dx = target[2] - fx
        dy = target[3] - fy
        target_angle = math.atan2(dy, dx)
        diff = abs(math.atan2(math.sin(target_angle - angle), math.cos(target_angle - angle)))
        if diff < angle_threshold:
            incoming += f[6]

    return incoming


def filter_targets_by_enemy_radar(obs, targets, angle_threshold=0.1):
    filtered_targets = []
    for t in targets:
        if t[1] != -1:
            filtered_targets.append(t)
            continue
        enemy_incoming = estimate_enemy_incoming_to_target(obs, t, angle_threshold=angle_threshold)
        if enemy_incoming >= t[5] + 1:
            continue
        filtered_targets.append(t)
    return filtered_targets


def get_comet_ids(obs):
    comet_ids = set()
    for c in obs.get("comets", []):
        for pid in c.get("planet_ids", []):
            comet_ids.add(pid)
    return comet_ids


def get_comet_remaining_steps(obs, planet_id):
    for c in obs.get("comets", []):
        pids = c.get("planet_ids", [])
        if planet_id not in pids:
            continue
        idx = pids.index(planet_id)
        paths = c.get("paths", [])
        path_index = c.get("path_index", 0)
        if idx < len(paths):
            return max(0, len(paths[idx]) - path_index)
    return 0


def get_nearest_owned_non_comet(source, my_planets, comet_ids):
    candidates = [p for p in my_planets if p[0] != source[0] and p[0] not in comet_ids]
    if not candidates:
        return None
    return min(candidates, key=lambda p: math.hypot(source[2] - p[2], source[3] - p[3]))


def get_best_comet_target_v2(mine, available_targets, obs, comet_ids):
    comet_targets = [t for t in available_targets if t[0] in comet_ids]
    if not comet_targets:
        return None

    valid = []
    for t in comet_targets:
        remaining = get_comet_remaining_steps(obs, t[0])
        distance = math.hypot(mine[2] - t[2], mine[3] - t[3])
        if remaining < 15:
            continue
        if distance > 35:
            continue
        if t[5] > 30:
            continue
        valid.append(t)

    if not valid:
        return None

    return max(valid, key=lambda t: (t[6] / (t[5] + 1), -math.hypot(mine[2] - t[2], mine[3] - t[3])))


def segment_hits_sun(x1, y1, x2, y2, buffer=0.5):
    r = SUN_RADIUS + buffer
    dx, dy = x2 - x1, y2 - y1
    fx, fy = x1 - CENTER_X, y1 - CENTER_Y
    a = dx * dx + dy * dy
    if a < 1e-9:
        return (x1 - CENTER_X) ** 2 + (y1 - CENTER_Y) ** 2 <= r * r
    b = 2 * (fx * dx + fy * dy)
    c = fx * fx + fy * fy - r * r
    disc = b * b - 4 * a * c
    if disc < 0:
        return False
    sqrt_disc = math.sqrt(disc)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)
    return (0 <= t1 <= 1) or (0 <= t2 <= 1)


def path_hits_sun(source, angle, max_len=200.0, buffer=0.5):
    x1, y1 = source[2], source[3]
    x2 = x1 + math.cos(angle) * max_len
    y2 = y1 + math.sin(angle) * max_len
    return segment_hits_sun(x1, y1, x2, y2, buffer=buffer)


def make_all_strategy_agent(
    early_off=True,
    early_off_until=50,
    angle_threshold=0.1,
    use_capture_filter=False,
    enemy_margin=5,
    use_estimate_defense=True,
    estimate_scale=0.8,
    use_enemy_radar=True,
    radar_angle_threshold=0.15,
    radar_start_step=0,
    wait_margin=0,
    use_best_target=True,
    target_score_type="prod_dist",
    target_phase_step=0,
    use_comets_v2=True,
    comet_evac_remaining=8,
    avoid_sun=True,
    sun_buffer=0.5,
):
    def _agent(obs):
        moves = []
        step = obs["step"]
        player = obs["player"]

        my_planets, targets = split_planets(obs)
        if not my_planets or not targets:
            return moves

        comet_ids = get_comet_ids(obs) if use_comets_v2 else set()
        evacuated_sources = set()

        if use_comets_v2 and comet_ids:
            for comet in my_planets:
                if comet[0] not in comet_ids:
                    continue
                remaining = get_comet_remaining_steps(obs, comet[0])
                if remaining > comet_evac_remaining:
                    continue
                destination = get_nearest_owned_non_comet(comet, my_planets, comet_ids)
                if destination is None:
                    continue
                ships_to_send = comet[5]
                if ships_to_send <= 0:
                    continue
                angle = get_angle(comet, destination)
                if avoid_sun and path_hits_sun(comet, angle, buffer=sun_buffer):
                    continue
                moves.append([comet[0], angle, ships_to_send])
                evacuated_sources.add(comet[0])

        if early_off and step < early_off_until:
            reserved_targets = set()
        else:
            reserved_targets = get_reserved_targets(
                obs,
                angle_threshold=angle_threshold,
                use_capture_filter=use_capture_filter,
            )

        planet_by_id = {p[0]: p for p in obs["planets"]}

        for mine in my_planets:
            if mine[0] in evacuated_sources:
                continue

            available_targets = [t for t in targets if t[0] not in reserved_targets]
            if not available_targets:
                continue

            radar_enabled = use_enemy_radar and step >= radar_start_step
            if radar_enabled:
                available_targets = filter_targets_by_enemy_radar(
                    obs, available_targets, angle_threshold=radar_angle_threshold
                )
                if not available_targets:
                    continue

            if use_comets_v2 and comet_ids:
                normal_targets = [t for t in available_targets if t[0] not in comet_ids]
                if normal_targets:
                    candidate_targets = normal_targets
                else:
                    comet_fallback = get_best_comet_target_v2(mine, available_targets, obs, comet_ids)
                    if comet_fallback is None:
                        continue
                    candidate_targets = [comet_fallback]
            else:
                candidate_targets = available_targets

            if use_best_target and step >= target_phase_step:
                target = get_best_value_target(
                    mine=mine,
                    targets=candidate_targets,
                    score_type=target_score_type,
                )
            else:
                target = get_nearest_target(mine, candidate_targets)

            ships_needed = target[5] + 1

            if target[1] not in (-1, player):
                ships_needed += enemy_margin
                if use_estimate_defense:
                    estimated_defense = estimate_target_defense(
                        source=mine,
                        target=target,
                        ships_to_send=ships_needed,
                    )
                    scaled_estimate = math.ceil(estimated_defense * estimate_scale)
                    ships_needed = max(ships_needed, scaled_estimate + 1)

            if wait_margin > 0 and ships_needed > mine[5] - wait_margin:
                continue

            if mine[5] >= ships_needed:
                angle = get_angle(mine, target)
                src = planet_by_id.get(mine[0], mine)
                if avoid_sun and path_hits_sun(src, angle, buffer=sun_buffer):
                    continue
                moves.append([mine[0], angle, ships_needed])
                reserved_targets.add(target[0])

        return moves

    return _agent


# Default submission configuration: enable all major strategies from ablation notebook.
_COMBINED_AGENT = make_all_strategy_agent(
    early_off=True,
    early_off_until=50,
    angle_threshold=0.1,
    use_capture_filter=False,
    enemy_margin=5,
    use_estimate_defense=True,
    estimate_scale=0.8,
    use_enemy_radar=True,
    radar_angle_threshold=0.15,
    radar_start_step=0,
    wait_margin=0,
    use_best_target=True,
    target_score_type="prod_dist",
    target_phase_step=0,
    use_comets_v2=True,
    comet_evac_remaining=8,
    avoid_sun=True,
    sun_buffer=0.5,
)


def agent(obs):
    try:
        return _COMBINED_AGENT(obs)
    except Exception:
        return []
