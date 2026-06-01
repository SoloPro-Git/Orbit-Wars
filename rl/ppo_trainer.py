"""
PPO Trainer for Orbit Wars.

Handles rollout collection, GAE computation, and PPO updates.
Supports parallel environments and opponent pools.
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import List, Dict, Tuple
from env import OrbitWarsEnv, BOT_REGISTRY


class RolloutBuffer:
    """Stores rollout data for PPO training."""
    
    def __init__(self):
        self.planets = []
        self.fleets = []
        self.player_ids = []
        self.steps = []
        self.planet_masks = []
        self.fleet_masks = []
        self.owned_masks = []
        
        self.send_actions = []
        self.target_actions = []
        
        self.log_probs = []
        self.rewards = []
        self.values = []
        self.dones = []
        
    def add(self, planets, fleets, player_id, step, planet_mask, fleet_mask,
            owned_mask, send_action, target_action, log_prob, reward, value, done):
        self.planets.append(planets)
        self.fleets.append(fleets)
        self.player_ids.append(player_id)
        self.steps.append(step)
        self.planet_masks.append(planet_mask)
        self.fleet_masks.append(fleet_mask)
        self.owned_masks.append(owned_mask)
        self.send_actions.append(send_action)
        self.target_actions.append(target_action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.values.append(value)
        self.dones.append(done)
        
    def get(self):
        return {
            'planets': torch.stack(self.planets),
            'fleets': torch.stack(self.fleets),
            'player_ids': torch.stack(self.player_ids),
            'steps': torch.stack(self.steps),
            'planet_masks': torch.stack(self.planet_masks),
            'fleet_masks': torch.stack(self.fleet_masks),
            'owned_masks': torch.stack(self.owned_masks),
            'send_actions': torch.stack(self.send_actions),
            'target_actions': torch.stack(self.target_actions),
            'log_probs': torch.stack(self.log_probs),
            'rewards': torch.tensor(self.rewards),
            'values': torch.stack(self.values),
            'dones': torch.tensor(self.dones),
        }
        
    def clear(self):
        self.__init__()


class PPOTrainer:
    """
    PPO trainer for Orbit Wars.
    
    Features:
    - GAE (Generalized Advantage Estimation)
    - Clipped surrogate objective
    - Value function clipping
    - Entropy bonus
    - Opponent pool for diverse training
    """
    
    def __init__(self, env, agent, opponent_pool=None, device='cpu', replay_opponents=None):
        self.env = env
        self.agent = agent
        self.device = device
        self.agent.to(device)
        
        # PPO hyperparameters
        self.gamma = 0.99
        self.gae_lambda = 0.95
        self.clip_ratio = 0.2
        self.vf_clip_ratio = 10.0
        self.ppo_epochs = 4
        self.batch_size = 64
        self.lr = 3e-4
        self.ent_coef = 0.01
        self.vf_coef = 0.5
        self.max_grad_norm = 0.5
        
        self.optimizer = optim.Adam(agent.parameters(), lr=self.lr, eps=1e-5)
        
        # Opponent pool
        self.opponent_pool = opponent_pool or ['starter']
        self.replay_opponents = replay_opponents or {}  # Dict[str, HybridOpponent]
        self.current_opponent_idx = 0
        
        # Normalization constants
        self.planet_norm = torch.tensor([1.0, 1.0, 100.0, 100.0, 3.0, 100.0, 5.0], device=device)
        self.fleet_norm = torch.tensor([1.0, 1.0, 100.0, 100.0, np.pi, 1.0, 100.0], device=device)
        
        self.buffer = RolloutBuffer()
        
    def collect_rollouts(self, num_steps=2048):
        """
        Collect rollout data by playing games.
        
        Args:
            num_steps: target number of steps to collect
            
        Returns:
            actual_steps: number of steps collected
        """
        self.buffer.clear()
        actual_steps = 0
        
        while actual_steps < num_steps:
            # Select opponent
            opponent_name = self.opponent_pool[self.current_opponent_idx % len(self.opponent_pool)]
            self.current_opponent_idx += 1
            
            # Use replay opponent if available, otherwise use BOT_REGISTRY
            if opponent_name == 'replay' and self.replay_opponents:
                # 随机选择一个 replay opponent
                import random
                replay_name = random.choice(list(self.replay_opponents.keys()))
                opponent = self.replay_opponents[replay_name]
            else:
                opponent = BOT_REGISTRY[opponent_name]()
            
            # Reset environment
            obs_list = self.env.reset()
            done = False
            
            while not done and actual_steps < num_steps:
                obs = obs_list[0]  # Player 0 is the agent
                
                # Convert observation to tensors
                planets_t, fleets_t, pid_t, step_t, p_mask, f_mask, o_mask = \
                    self._obs_to_tensors(obs)
                
                # Get action from agent
                with torch.no_grad():
                    actions, log_prob, entropy, value, info = self.agent.get_action_and_value(
                        planets_t, fleets_t, pid_t, step_t, p_mask, f_mask, o_mask)
                
                # Get opponent action
                opp_obs = obs_list[1]
                opp_actions = opponent.act(opp_obs)
                
                # Step environment
                obs_list, rewards, dones, infos = self.env.step([actions[0], opp_actions])
                
                # Store transition
                self.buffer.add(
                    planets_t.squeeze(0),
                    fleets_t.squeeze(0),
                    pid_t.squeeze(0),
                    step_t.squeeze(0),
                    p_mask.squeeze(0),
                    f_mask.squeeze(0),
                    o_mask.squeeze(0),
                    info['send_action'].squeeze(0),
                    info['target_action'].squeeze(0),
                    log_prob.squeeze(0),
                    rewards[0],
                    value.squeeze(0),
                    dones[0],
                )
                
                done = dones[0]
                actual_steps += 1
                
        return actual_steps
    
    def train_step(self):
        """
        Perform one PPO update using collected rollouts.
        
        Returns:
            stats: dict with training statistics
        """
        data = self.buffer.get()
        
        # Compute advantages using GAE
        advantages = self._compute_gae(data['rewards'], data['values'], data['dones'])
        # Ensure advantages is on the same device as values
        returns = advantages.to(data['values'].device) + data['values']
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # PPO epochs
        total_loss = 0
        total_policy_loss = 0
        total_value_loss = 0
        total_entropy = 0
        
        num_samples = len(data['rewards'])
        indices = np.arange(num_samples)
        
        for _ in range(self.ppo_epochs):
            np.random.shuffle(indices)
            
            for start in range(0, num_samples, self.batch_size):
                end = start + self.batch_size
                batch_idx = indices[start:end]
                
                # Get batch
                batch = {k: v[batch_idx].to(self.device) for k, v in data.items()}
                batch_adv = advantages[batch_idx].to(self.device)
                batch_ret = returns[batch_idx].to(self.device)
                
                # Forward pass
                old_lp = batch['log_probs']
                old_val = batch['values']
                
                new_lp, entropy, new_val = self.agent.evaluate_actions(
                    batch['planets'],
                    batch['fleets'],
                    batch['player_ids'],
                    batch['steps'],
                    batch['send_actions'],
                    batch['target_actions'],
                    batch['planet_masks'],
                    batch['fleet_masks'],
                    batch['owned_masks'],
                )
                
                # Policy loss (clipped surrogate)
                ratio = torch.exp(new_lp - old_lp)
                surr1 = ratio * batch_adv
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                v_clipped = old_val + torch.clamp(
                    new_val - old_val, -self.vf_clip_ratio, self.vf_clip_ratio)
                vf_loss1 = (new_val - batch_ret) ** 2
                vf_loss2 = (v_clipped - batch_ret) ** 2
                value_loss = 0.5 * torch.max(vf_loss1, vf_loss2).mean()
                
                # Entropy bonus
                entropy_loss = -entropy.mean()
                
                # Total loss
                loss = policy_loss + self.vf_coef * value_loss + self.ent_coef * entropy_loss
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.agent.parameters(), self.max_grad_norm)
                self.optimizer.step()
                
                total_loss += loss.item()
                total_policy_loss += policy_loss.item()
                total_value_loss += value_loss.item()
                total_entropy += entropy.mean().item()
        
        # Compute stats
        num_updates = self.ppo_epochs * ((num_samples + self.batch_size - 1) // self.batch_size)
        stats = {
            'loss': total_loss / num_updates,
            'policy_loss': total_policy_loss / num_updates,
            'value_loss': total_value_loss / num_updates,
            'entropy': total_entropy / num_updates,
            'mean_reward': data['rewards'].mean().item(),
            'mean_value': data['values'].mean().item(),
        }
        
        self.buffer.clear()
        return stats
    
    def evaluate(self, opponent='starter', num_games=50):
        """
        Evaluate agent against an opponent.
        
        Args:
            opponent: opponent name
            num_games: number of games to play
            
        Returns:
            stats: dict with win_rate, avg_reward, etc.
        """
        self.agent.eval()
        opp_bot = BOT_REGISTRY[opponent]()
        
        wins = 0
        total_reward = 0
        
        for _ in range(num_games):
            obs_list = self.env.reset()
            done = False
            game_reward = 0
            
            while not done:
                obs = obs_list[0]
                planets_t, fleets_t, pid_t, step_t, p_mask, f_mask, o_mask = \
                    self._obs_to_tensors(obs)
                
                with torch.no_grad():
                    actions, _, _, _, _ = self.agent.get_action_and_value(
                        planets_t, fleets_t, pid_t, step_t, p_mask, f_mask, o_mask)
                
                opp_obs = obs_list[1]
                opp_actions = opp_bot.act(opp_obs)
                
                obs_list, rewards, dones, infos = self.env.step([actions[0], opp_actions])
                game_reward += rewards[0]
                done = dones[0]
            
            if infos[0]['winner'] == 0:
                wins += 1
            total_reward += game_reward
        
        self.agent.train()
        
        return {
            'win_rate': wins / num_games,
            'avg_reward': total_reward / num_games,
            'wins': wins,
            'games': num_games,
        }
    
    def _compute_gae(self, rewards, values, dones):
        """Compute Generalized Advantage Estimation."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_non_terminal = 1 - dones[t].float()
            else:
                next_value = values[t + 1]
                next_non_terminal = 1 - dones[t].float()
            
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae
            advantages[t] = last_gae
            
        return advantages
    
    def _obs_to_tensors(self, obs):
        """Convert observation dict to tensors."""
        raw_planets = obs['planets']
        raw_fleets = obs['fleets']
        player = obs['player']
        step = obs['step']
        
        # Pad and normalize planets
        planets = torch.zeros(1, self.agent.max_planets, 7, device=self.device)
        planet_mask = torch.zeros(1, self.agent.max_planets, dtype=torch.bool, device=self.device)
        owned_mask = torch.zeros(1, self.agent.max_planets, dtype=torch.bool, device=self.device)
        
        for i, p in enumerate(raw_planets[:self.agent.max_planets]):
            for j in range(7):
                planets[0, i, j] = p[j] / self.planet_norm[j].item()
            planet_mask[0, i] = True
            if p[1] == player:
                owned_mask[0, i] = True
        
        # Pad and normalize fleets
        fleets = torch.zeros(1, self.agent.max_fleets, 7, device=self.device)
        fleet_mask = torch.zeros(1, self.agent.max_fleets, dtype=torch.bool, device=self.device)
        
        for i, f in enumerate(raw_fleets[:self.agent.max_fleets]):
            for j in range(7):
                fleets[0, i, j] = f[j] / self.fleet_norm[j].item()
            fleet_mask[0, i] = True
        
        pid = torch.tensor([player], dtype=torch.long, device=self.device)
        step_t = torch.tensor([step], dtype=torch.float32, device=self.device)
        
        return planets, fleets, pid, step_t, planet_mask, fleet_mask, owned_mask
