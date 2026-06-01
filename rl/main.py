"""
Orbit Wars Agent v5 — BC Model + Rule-based hybrid
Uses behavior cloning model for decisions, falls back to rules for safety.
"""
import math
import os
import sys

# Try to load BC model
try:
    import torch
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))
    from bc_model_v2 import BCModelV2
    
    PLANET_NORM = [1.0, 1.0, 100.0, 100.0, 3.0, 100.0, 5.0]
    FLEET_NORM = [1.0, 1.0, 100.0, 100.0, math.pi, 1.0, 100.0]
    
    _model = None
    _model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints', 'bc_1v1_v2_best.pt')
    
    def _load_model():
        global _model
        if _model is None:
            ckpt = torch.load(_model_path, map_location='cpu', weights_only=False)
            args = ckpt.get('args', {})
            _model = BCModelV2(
                embed_dim=args.get('embed_dim', 64),
                num_players=args.get('num_players', 2)
            )
            _model.load_state_dict(ckpt['model_state_dict'])
            _model.eval()
        return _model
    
    HAS_MODEL = True
except Exception:
    HAS_MODEL = False


def agent(obs):
    try:
        return _agent_impl(obs)
    except Exception as e:
        import traceback
        print(f"[Agent ERROR] {e}")
        traceback.print_exc()
        return []


def _agent_impl(obs):
    """Main agent implementation"""
    if isinstance(obs, dict):
        player = obs.get("player", 0)
        raw_planets = obs.get("planets", [])
        raw_fleets = obs.get("fleets", [])
        step = obs.get("step", 0)
    else:
        player = obs.player
        raw_planets = obs.planets
        raw_fleets = obs.fleets if hasattr(obs, 'fleets') else []
        step = obs.step if hasattr(obs, 'step') else 0
    
    if not raw_planets:
        return []
    
    my_planets = [p for p in raw_planets if p[1] == player]
    if not my_planets:
        return []
    
    # Try BC model
    if HAS_MODEL:
        try:
            return _bc_agent(raw_planets, raw_fleets, player, step)
        except Exception:
            pass
    
    # Fallback: simple rule-based
    return _rule_agent(raw_planets, raw_fleets, player, step)


def _bc_agent(raw_planets, raw_fleets, player, step):
    """BC model-based agent."""
    model = _load_model()
    
    MAX_P = 40
    MAX_F = 100
    
    # Keep raw planets for ship count lookup
    raw_planet_ships = {p[0]: p[5] for p in raw_planets if p[1] == player}
    
    # Normalize and pad planets
    planets = torch.zeros(1, MAX_P, 7)
    planet_mask = torch.zeros(1, MAX_P, dtype=torch.bool)
    for i, p in enumerate(raw_planets[:MAX_P]):
        for j in range(7):
            planets[0, i, j] = p[j] / PLANET_NORM[j]
        planet_mask[0, i] = True
    
    # Normalize and pad fleets
    fleets = torch.zeros(1, MAX_F, 7)
    fleet_mask = torch.zeros(1, MAX_F, dtype=torch.bool)
    for i, f in enumerate(raw_fleets[:MAX_F]):
        for j in range(7):
            fleets[0, i, j] = f[j] / FLEET_NORM[j]
        fleet_mask[0, i] = True
    
    pid = torch.tensor([player], dtype=torch.long)
    step_t = torch.tensor([step], dtype=torch.float32)
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        outputs = model(planets, fleets, pid, step_t, planet_mask, fleet_mask)
    
    send_probs = torch.sigmoid(outputs['should_send']).squeeze(0).squeeze(-1)  # [N]
    angles = outputs['angle'].squeeze(0).squeeze(-1)  # [N]
    ratios = outputs['ship_ratio'].squeeze(0).squeeze(-1)  # [N]
    
    # Build action using RAW ship counts
    actions = []
    owners = [p[1] for p in raw_planets[:MAX_P]]
    for i in range(min(len(raw_planets), MAX_P)):
        if owners[i] != player:
            continue
        
        prob = send_probs[i].item() if send_probs.dim() > 0 else send_probs.item()
        
        if prob > 0.15:
            planet_id = raw_planets[i][0]
            available = raw_planet_ships.get(planet_id, 0)
            if available <= 1:
                if step < 10:
                    print(f"[BC]     SKIP: available={available} <= 1")
                continue
            
            send_angle = angles[i].item() if angles.dim() > 0 else angles.item()
            ratio = ratios[i].item() if ratios.dim() > 0 else ratios.item()
            send_ships = max(int(available * ratio), 1)
            send_ships = min(send_ships, available - 1)  # keep at least 1
            
            if send_ships >= 1:
                actions.append([planet_id, send_angle, send_ships])
                if step < 10:
                    print(f"[BC]     ACTION: planet={planet_id}, angle={send_angle:.2f}, ships={send_ships}")
    
    if step < 10:
        print(f"[BC]   total actions: {len(actions)}")
    
    return actions


def _rule_agent(raw_planets, raw_fleets, player, step):
    """Simple rule-based fallback (improved v4)."""
    from kaggle_environments.envs.orbit_wars.orbit_wars import CENTER, ROTATION_RADIUS_LIMIT
    
    my = [p for p in raw_planets if p[1] == player]
    targets = [p for p in raw_planets if p[1] != player]
    if not targets:
        return []
    
    moves = []
    for mp in my:
        if mp[5] <= 5:
            continue
        
        best = None
        best_score = -999
        
        for t in targets:
            d = math.hypot(mp[2] - t[2], mp[3] - t[3])
            if d < 1:
                continue
            
            send = mp[5] // 2
            if send < 16:
                continue
            
            speed = 1.0 + 5.0 * (math.log(max(send, 1)) / math.log(1000)) ** 1.5
            tt = math.ceil(d / min(speed, 6.0))
            pred = t[5] + tt * t[6]
            surplus = send - pred
            
            if surplus > 0:
                score = t[6] / math.sqrt(send + 1) + 2.0 / (tt + 5)
            else:
                score = surplus / 100.0
            
            # Static bonus
            if math.hypot(t[2] - CENTER, t[3] - CENTER) + t[4] >= ROTATION_RADIUS_LIMIT:
                score *= 1.2
            
            if score > best_score:
                best_score = score
                best = t
        
        if best and best_score > -10:
            angle = math.atan2(best[3] - mp[3], best[2] - mp[2])
            ships = mp[5] // 2
            if ships >= 16:
                moves.append([mp[0], angle, ships])
    
    return moves
