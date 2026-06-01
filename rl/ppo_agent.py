"""
Orbit Wars PPO Agent — Policy + Value network with pointer network targeting.

Key difference from BC model: instead of predicting raw angle, the policy
outputs a target_planet_idx (pointer network style) — it selects which planet
to attack from the set of all non-self planets. The angle is then computed
from the target planet's position.

Architecture:
- Encoder: reuses BC model's planet/fleet/global encoders (warm start)
- Policy head per owned planet: (send_prob, target_logits, ship_ratio)
- Value head: V(s) scalar
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical, Bernoulli


class PPOAgent(nn.Module):
    """
    PPO agent for Orbit Wars.
    
    For each owned planet, outputs:
    - send_prob: probability of sending a fleet
    - target_logits: distribution over all other planets (pointer network)
    - ship_ratio: fraction of ships to send (0-1)
    
    Also outputs value V(s) for PPO baseline.
    """
    def __init__(self, embed_dim=64, max_planets=40, max_fleets=100, num_players=2):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_planets = max_planets
        self.max_fleets = max_fleets
        self.num_players = num_players

        # --- Encoders (will be loaded from BC checkpoint) ---
        self.planet_enc = nn.Sequential(
            nn.Linear(7, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.fleet_enc = nn.Sequential(
            nn.Linear(7, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        # global_input: planet_summary(8) + fleet_summary(4) + step(1) + player_oh(num_players)
        global_input_dim = 8 + 4 + 1 + num_players
        self.global_enc = nn.Sequential(
            nn.Linear(global_input_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )

        # --- Policy heads (per owned planet) ---
        # Local + global -> action features
        self.action_head = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
        )
        self.send_head = nn.Linear(embed_dim, 1)  # logit
        self.ship_head = nn.Linear(embed_dim, 1)  # sigmoid

        # Pointer network: query = action_feat, keys = planet embeddings
        self.ptr_query = nn.Linear(embed_dim, embed_dim)
        self.ptr_key = nn.Linear(embed_dim, embed_dim)
        self.ptr_scale = math.sqrt(embed_dim)

        # --- Value head ---
        self.value_head = nn.Sequential(
            nn.Linear(embed_dim, embed_dim), nn.ReLU(),
            nn.Linear(embed_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def load_bc_encoder(self, bc_checkpoint_path):
        """Load encoder weights from BC model checkpoint (warm start)."""
        ckpt = torch.load(bc_checkpoint_path, map_location='cpu')
        state = ckpt.get('model_state_dict', ckpt)
        # Map BC model keys to our keys
        mapping = {}
        for k, v in state.items():
            # BC model uses: planet_enc, fleet_enc, global_enc
            if any(k.startswith(p) for p in ['planet_enc', 'fleet_enc', 'global_enc']):
                mapping[k] = v
        missing, unexpected = self.load_state_dict(mapping, strict=False)
        print(f"BC warm start: loaded {len(mapping)} params, "
              f"missing={len(missing)}, unexpected={len(unexpected)}")
        return len(mapping)

    def encode_state(self, planets, fleets, player_id, step,
                     planet_mask=None, fleet_mask=None):
        """Encode full state into embeddings."""
        B, N, _ = planets.shape
        # Planet embeddings
        planet_emb = self.planet_enc(planets)  # [B, N, D]
        if planet_mask is not None:
            planet_emb = planet_emb * planet_mask.unsqueeze(-1).float()
            count = planet_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            planet_pool = planet_emb.sum(dim=1) / count  # [B, D]
        else:
            planet_pool = planet_emb.mean(dim=1)

        # Fleet embeddings
        fleet_emb = self.fleet_enc(fleets)  # [B, M, D]
        if fleet_mask is not None:
            fleet_emb = fleet_emb * fleet_mask.unsqueeze(-1).float()
            count = fleet_mask.float().sum(dim=1, keepdim=True).clamp(min=1)
            fleet_pool = fleet_emb.sum(dim=1) / count
        else:
            fleet_pool = fleet_emb.mean(dim=1) if fleets.size(1) > 0 else torch.zeros(B, self.embed_dim, device=fleets.device)

        # Game state summary
        owners = planets[:, :, 1]
        ships = planets[:, :, 5]
        prod = planets[:, :, 6]
        pid = player_id.unsqueeze(1)
        my_m = (owners == pid)
        en_m = (owners >= 0) & (owners != pid)
        neu_m = (owners < 0)
        if planet_mask is not None:
            my_m = my_m & planet_mask
            en_m = en_m & planet_mask
            neu_m = neu_m & planet_mask

        ps = torch.cat([
            my_m.sum(1, keepdim=True).float(), en_m.sum(1, keepdim=True).float(),
            neu_m.sum(1, keepdim=True).float(),
            (ships * my_m.float()).sum(1, keepdim=True),
            (ships * en_m.float()).sum(1, keepdim=True),
            (ships * neu_m.float()).sum(1, keepdim=True),
            (prod * my_m.float()).sum(1, keepdim=True),
            (prod * en_m.float()).sum(1, keepdim=True),
        ], dim=1)  # [B, 8]

        if fleets.size(1) > 0:
            fo = fleets[:, :, 1]; fs = fleets[:, :, 6]
            fm = (fo == pid); fe = (fo >= 0) & (fo != pid)
            if fleet_mask is not None:
                fm = fm & fleet_mask; fe = fe & fleet_mask
            fsumm = torch.cat([
                fm.sum(1, keepdim=True).float(), fe.sum(1, keepdim=True).float(),
                (fs * fm.float()).sum(1, keepdim=True),
                (fs * fe.float()).sum(1, keepdim=True),
            ], dim=1)
        else:
            fsumm = torch.zeros(B, 4, device=planets.device)

        pid_oh = F.one_hot(player_id.clamp(0, self.num_players - 1), self.num_players).float()
        step_norm = step.unsqueeze(1) / 500.0
        global_input = torch.cat([ps, fsumm, step_norm, pid_oh], dim=1)
        global_feat = self.global_enc(global_input)  # [B, D]

        # Value
        value = self.value_head(global_feat).squeeze(-1)  # [B]

        return planet_emb, global_feat, value

    def forward(self, planets, fleets, player_id, step,
                planet_mask=None, fleet_mask=None, owned_mask=None):
        """
        Forward pass. Returns dict with:
        - send_logits: [B, N, 1]
        - target_logits: [B, N, N] (over all planets, self masked)
        - ship_ratio: [B, N, 1] in [0,1]
        - value: [B]
        """
        B, N, _ = planets.shape
        planet_emb, global_feat, value = self.encode_state(
            planets, fleets, player_id, step, planet_mask, fleet_mask)

        # Per-planet action features
        g_exp = global_feat.unsqueeze(1).expand(-1, N, -1)
        action_input = torch.cat([planet_emb, g_exp], dim=2)  # [B, N, 2D]
        action_feat = self.action_head(action_input)  # [B, N, D]

        # Send decision
        send_logits = self.send_head(action_feat)  # [B, N, 1]
        ship_ratio = torch.sigmoid(self.ship_head(action_feat))  # [B, N, 1]

        # Pointer network: target selection
        query = self.ptr_query(action_feat)  # [B, N, D]
        keys = self.ptr_key(planet_emb)  # [B, N, D]
        # Attention: [B, N_src, N_tgt]
        target_logits = torch.bmm(query, keys.transpose(1, 2)) / self.ptr_scale

        # Mask self (diagonal) and invalid planets
        eye_mask = torch.eye(N, device=planets.device).bool().unsqueeze(0)
        target_logits = target_logits.masked_fill(eye_mask, -1e9)
        if planet_mask is not None:
            invalid = ~planet_mask  # [B, N]
            target_logits = target_logits.masked_fill(invalid.unsqueeze(1), -1e9)

        # Also mask own planets as targets (don't attack yourself)
        if owned_mask is not None:
            own_expand = owned_mask.unsqueeze(1).expand(-1, N, -1)  # [B, N, N]
            target_logits = target_logits.masked_fill(own_expand, -1e9)

        return {
            'send_logits': send_logits,
            'target_logits': target_logits,
            'ship_ratio': ship_ratio,
            'value': value,
        }

    def get_action_and_value(self, planets, fleets, player_id, step,
                             planet_mask=None, fleet_mask=None, owned_mask=None):
        """
        Sample actions and compute log probs for PPO.
        Returns: actions, log_probs, entropy, value
        """
        out = self.forward(planets, fleets, player_id, step,
                          planet_mask, fleet_mask, owned_mask)
        B, N, _ = planets.shape
        device = planets.device

        send_logits = out['send_logits'].squeeze(-1)  # [B, N]
        target_logits = out['target_logits']  # [B, N, N]
        ship_ratio = out['ship_ratio'].squeeze(-1)  # [B, N]

        # For owned planets, sample send/no-send, target, ship_ratio
        send_probs = torch.sigmoid(send_logits)
        send_dist = Bernoulli(probs=send_probs)

        # Only act on owned planets
        if owned_mask is not None:
            # Zero out non-owned send logits
            send_logits_masked = send_logits.masked_fill(~owned_mask, -1e9)
            send_probs_m = torch.sigmoid(send_logits_masked)
            send_dist = Bernoulli(probs=send_probs_m)
        
        send_action = send_dist.sample()  # [B, N]
        send_log_prob = send_dist.log_prob(send_action)  # [B, N]

        # Sample target planet
        target_dist = Categorical(logits=target_logits)
        target_action = target_dist.sample()  # [B, N]
        target_log_prob = target_dist.log_prob(target_action)  # [B, N]

        # Entropy
        send_entropy = send_dist.entropy()  # [B, N]
        target_entropy = target_dist.entropy()  # [B, N]

        # Combine: only count owned planets
        if owned_mask is not None:
            own_f = owned_mask.float()
            total_log_prob = (send_log_prob * own_f).sum(1) + \
                            (target_log_prob * send_action * own_f).sum(1)
            total_entropy = (send_entropy * own_f).sum(1) + \
                           (target_entropy * send_action * own_f).sum(1)
        else:
            total_log_prob = send_log_prob.sum(1) + (target_log_prob * send_action).sum(1)
            total_entropy = send_entropy.sum(1) + (target_entropy * send_action).sum(1)

        # Build action list per batch
        actions = []
        owners = planets[:, :, 1]
        for b in range(B):
            batch_actions = []
            for i in range(N):
                if owned_mask is not None and not owned_mask[b, i]:
                    continue
                if send_action[b, i] < 0.5:
                    continue
                tid = int(target_action[b, i])
                # Compute angle from source to target planet
                src_x, src_y = planets[b, i, 2].item(), planets[b, i, 3].item()
                tgt_x, tgt_y = planets[b, tid, 2].item(), planets[b, tid, 3].item()
                angle = math.atan2(tgt_y - src_y, tgt_x - src_x)
                ships_total = int(planets[b, i, 5].item())
                ratio = float(ship_ratio[b, i])
                send_ships = max(int(ships_total * ratio), 1)
                if ships_total >= send_ships and send_ships >= 1:
                    planet_id = int(planets[b, i, 0].item())
                    batch_actions.append([planet_id, angle, send_ships])
            actions.append(batch_actions)

        return actions, total_log_prob, total_entropy, out['value'], {
            'send_action': send_action,
            'target_action': target_action,
            'ship_ratio': ship_ratio,
        }

    def evaluate_actions(self, planets, fleets, player_id, step,
                         send_actions, target_actions,
                         planet_mask=None, fleet_mask=None, owned_mask=None):
        """
        Evaluate log probs and entropy for given actions (for PPO update).
        send_actions: [B, N] binary
        target_actions: [B, N] int
        """
        out = self.forward(planets, fleets, player_id, step,
                          planet_mask, fleet_mask, owned_mask)
        B, N, _ = planets.shape

        send_logits = out['send_logits'].squeeze(-1)  # [B, N]
        target_logits = out['target_logits']  # [B, N, N]

        # Send log prob
        if owned_mask is not None:
            send_logits_m = send_logits.masked_fill(~owned_mask, -1e9)
        else:
            send_logits_m = send_logits
        send_dist = Bernoulli(logits=send_logits_m)
        send_lp = send_dist.log_prob(send_actions)
        send_ent = send_dist.entropy()

        # Target log prob
        target_dist = Categorical(logits=target_logits)
        target_lp = target_dist.log_prob(target_actions.long())
        target_ent = target_dist.entropy()

        own_f = owned_mask.float() if owned_mask is not None else torch.ones(B, N, device=planets.device)
        total_lp = (send_lp * own_f).sum(1) + (target_lp * send_actions * own_f).sum(1)
        total_ent = (send_ent * own_f).sum(1) + (target_ent * send_actions * own_f).sum(1)

        return total_lp, total_ent, out['value']
