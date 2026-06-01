"""
Replay Opponent: 从顶级选手的爬分 replay 中学习策略，创建可对战的 bot。

核心思想：
- 加载选手的 replay 数据（扁平格式：每条是一个 (obs, action) 对）
- 对于给定的游戏状态，找到最相似的历史状态
- 使用该状态下选手的实际动作作为 bot 的动作
- 这样 PPO 就能与"真正的冠军"对战
"""
import json
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path


class ReplayOpponent:
    """
    基于历史 replay 的对手 bot。
    
    对于当前游戏状态，找到最相似的历史状态，返回选手的实际动作。
    """
    
    def __init__(self, replay_file: str, similarity_threshold: float = 0.8, max_pairs: int = 5000):
        """
        Args:
            replay_file: replay JSON 文件路径
            similarity_threshold: 相似度阈值，低于此值则使用 fallback 策略
            max_pairs: 最多加载的 (state, action) 对数量
        """
        self.similarity_threshold = similarity_threshold
        self.state_action_pairs = []
        self.fallback_bot = None  # 当找不到相似状态时使用的 fallback
        
        self._load_replay(replay_file, max_pairs)
    
    def _load_replay(self, replay_file: str, max_pairs: int):
        """加载 replay 数据，提取 (state, action) 对"""
        with open(replay_file, 'r') as f:
            data = json.load(f)
        
        # 数据格式：扁平列表，每条是一个 (obs, action) 对
        # keys: player, step, planets, fleets, action, player_id, reward, phase
        
        # 采样以限制内存
        if len(data) > max_pairs:
            import random
            data = random.sample(data, max_pairs)
        
        for item in data:
            action = item.get('action')
            if not action or len(action) == 0:
                continue
            
            # 构造 observation 格式（兼容 env.py 的 get_obs 返回格式）
            obs = {
                'player': item['player_id'],
                'step': item['step'],
                'planets': item['planets'],
                'fleets': item['fleets'],
            }
            
            state = self._extract_state_features(obs, item['player_id'])
            self.state_action_pairs.append({
                'state': state,
                'action': action,
                'player_id': item['player_id']
            })
        
        print(f"Loaded {len(self.state_action_pairs)} state-action pairs from {replay_file}")
    
    def _extract_state_features(self, obs: Dict, player_id: int) -> np.ndarray:
        """
        从 observation 提取状态特征向量，用于相似度计算。
        """
        planets = obs['planets']
        fleets = obs['fleets']
        step = obs['step']
        
        # 计算统计特征
        my_planets = [p for p in planets if p[1] == player_id]
        enemy_planets = [p for p in planets if p[1] != player_id and p[1] != -1]
        neutral_planets = [p for p in planets if p[1] == -1]
        
        my_ships = sum(p[5] for p in my_planets)
        enemy_ships = sum(p[5] for p in enemy_planets)
        neutral_ships = sum(p[5] for p in neutral_planets)
        
        my_prod = sum(p[6] for p in my_planets)
        enemy_prod = sum(p[6] for p in enemy_planets)
        
        # 特征向量
        features = np.array([
            len(my_planets),
            my_ships,
            my_prod,
            len(enemy_planets),
            enemy_ships,
            enemy_prod,
            len(neutral_planets),
            neutral_ships,
            step / 500.0,
            my_ships / (enemy_ships + 1e-6),
            my_prod / (enemy_prod + 1e-6),
        ])
        
        return features
    
    def _compute_similarity(self, state1: np.ndarray, state2: np.ndarray) -> float:
        """计算两个状态的相似度（0-1）"""
        norm1 = np.linalg.norm(state1)
        norm2 = np.linalg.norm(state2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_sim = np.dot(state1, state2) / (norm1 * norm2)
        return (cos_sim + 1) / 2
    
    def act(self, obs: Dict) -> List[List]:
        """
        根据当前 observation 返回动作。
        
        策略：
        1. 提取当前状态特征
        2. 找到最相似的历史状态
        3. 如果相似度 > threshold，返回历史动作
        4. 否则使用 fallback 策略
        """
        player_id = obs['player']
        current_state = self._extract_state_features(obs, player_id)
        
        # 如果没有数据，使用 fallback
        if not self.state_action_pairs:
            if self.fallback_bot:
                return self.fallback_bot.act(obs)
            return []
        
        # 找到最相似的状态
        best_match = None
        best_similarity = 0.0
        
        for pair in self.state_action_pairs:
            similarity = self._compute_similarity(current_state, pair['state'])
            
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pair
        
        # 如果找到足够相似的状态，返回历史动作
        if best_match and best_similarity > self.similarity_threshold:
            return best_match['action']
        
        # 否则使用 fallback
        if self.fallback_bot:
            return self.fallback_bot.act(obs)
        
        return []
    
    def set_fallback(self, bot):
        """设置 fallback bot"""
        self.fallback_bot = bot


class HybridOpponent:
    """
    混合对手：结合 Replay Opponent 和规则 Bot。
    
    策略：
    - 有概率使用 replay 对手（学习冠军策略）
    - 有概率使用规则 bot（增加多样性）
    """
    
    def __init__(self, replay_opponent, rule_bot, replay_prob: float = 0.7):
        self.replay_opponent = replay_opponent
        self.rule_bot = rule_bot
        self.replay_prob = replay_prob
    
    def act(self, obs: Dict) -> List[List]:
        """根据概率选择对手策略"""
        if np.random.random() < self.replay_prob:
            return self.replay_opponent.act(obs)
        else:
            return self.rule_bot.act(obs)


def load_replay_opponents(replay_dir: str, mode: str = '1v1', replay_prob: float = 0.7) -> Dict[str, HybridOpponent]:
    """
    从 replay 目录加载多个选手的对手 bot。
    
    使用 early 阶段数据（最容易匹配到的状态）。
    """
    from env import StarterBot
    
    replay_path = Path(replay_dir)
    opponents = {}
    
    # 加载 early 阶段的 replay（状态更简单，更容易匹配）
    replay_file = replay_path / f'climb_{mode}_early.json'
    
    if not replay_file.exists():
        print(f"Warning: {replay_file} not found, skipping replay opponents")
        return opponents
    
    # 加载数据
    with open(replay_file, 'r') as f:
        data = json.load(f)
    
    # 按选手分组
    player_data = {}
    for item in data[:20000]:  # 限制总量
        player_name = item.get('player', 'unknown')
        if player_name not in player_data:
            player_data[player_name] = []
        player_data[player_name].append(item)
    
    # 为每个选手创建对手
    fallback_bot = StarterBot()
    
    for player_name, items in player_data.items():
        if len(items) < 50:  # 太少的数据跳过
            continue
        
        # 创建 ReplayOpponent
        replay_opp = ReplayOpponent.__new__(ReplayOpponent)
        replay_opp.similarity_threshold = 0.8
        replay_opp.state_action_pairs = []
        replay_opp.fallback_bot = fallback_bot
        
        # 直接加载数据（跳过文件读取）
        import random
        sampled = random.sample(items, min(len(items), 2000))
        
        for item in sampled:
            action = item.get('action')
            if not action or len(action) == 0:
                continue
            
            obs = {
                'player': item['player_id'],
                'step': item['step'],
                'planets': item['planets'],
                'fleets': item['fleets'],
            }
            state = replay_opp._extract_state_features(obs, item['player_id'])
            replay_opp.state_action_pairs.append({
                'state': state,
                'action': action,
                'player_id': item['player_id']
            })
        
        if len(replay_opp.state_action_pairs) > 0:
            hybrid_opp = HybridOpponent(replay_opp, fallback_bot, replay_prob)
            opponents[player_name] = hybrid_opp
            print(f"  Loaded {len(replay_opp.state_action_pairs)} pairs for {player_name}")
    
    return opponents
