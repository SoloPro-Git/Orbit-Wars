"""调试评估脚本 - 加载检查点与最近星球策略对战，计算胜率并生成 HTML 回放"""
import sys
import os
import math
from pathlib import Path

# 添加项目根目录到 sys.path
_PROJECT_ROOT = str(Path(__file__).resolve().parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import torch
import numpy as np
from kaggle_environments import make
from kaggle_environments.envs.orbit_wars.orbit_wars import Planet

from training.core.config import ModelConfig
from training.core.feature_engineering import FeatureEngineer
from training.core.model import OrbitWarsModel
from training.core.action import decode_actions, actions_to_kaggle


# ---------------------------------------------------------------------------
# 最近星球策略（初始 commit 548c6aa 的 main.py）
# ---------------------------------------------------------------------------
def nearest_planet_agent(obs):
    """初始 commit 的最近星球狙击策略"""
    moves = []
    player = obs.get("player", 0) if isinstance(obs, dict) else obs.player
    raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
    planets = [Planet(*p) for p in raw_planets]

    my_planets = [p for p in planets if p.owner == player]
    targets = [p for p in planets if p.owner != player]

    if not targets:
        return moves

    for mine in my_planets:
        nearest = min(targets, key=lambda t: math.hypot(mine.x - t.x, mine.y - t.y))
        ships_needed = nearest.ships + 1
        if mine.ships >= ships_needed:
            angle = math.atan2(nearest.y - mine.y, nearest.x - mine.x)
            moves.append([mine.id, angle, ships_needed])

    return moves


# ---------------------------------------------------------------------------
# 模型 agent（加载 pkl 检查点）
# ---------------------------------------------------------------------------
class ModelAgent:
    """加载训练好的模型作为 Kaggle agent"""

    def __init__(self, checkpoint_path: str, device: str = "cuda:0"):
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.config = ModelConfig()
        self.feature_engineer = FeatureEngineer()

        # 加载检查点
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
            print(f"加载检查点: iteration={checkpoint.get('iteration', '?')}")
        else:
            state_dict = checkpoint

        # 推断 n_planets（从 policy_head 的输出维度）
        n_planets = 32
        for key, val in state_dict.items():
            if "policy_head.target_mlp" in key and val.shape[0] <= 40:
                n_planets = int(val.shape[0])
                break

        print(f"模型配置: d_model={self.config.d_model}, n_planets={n_planets}")

        self.model = OrbitWarsModel(self.config, n_planets=n_planets)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()
        self.n_planets = n_planets

        # 修补位置编码，支持超过 max_len 的序列
        self._patch_positional_encoding()

    def _patch_positional_encoding(self):
        """将位置编码改为动态扩展，避免超过 max_len 报错"""
        original_pe = self.model.pos_enc

        # 扩展到 2048
        max_len = 2048
        d_model = original_pe.pe.shape[-1]
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).to(self.device)
        original_pe.pe = pe

    def __call__(self, obs, configuration=None):
        """Kaggle agent 接口"""
        try:
            return self._predict(obs)
        except Exception as e:
            print(f"[ModelAgent] 错误: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _predict(self, obs):
        player_id = obs.get("player", 0) if isinstance(obs, dict) else obs.player
        raw_planets = obs.get("planets", []) if isinstance(obs, dict) else obs.planets
        my_planets = [p for p in raw_planets if int(p[1]) == player_id]

        # 特征提取
        planet_features, fleet_features, global_features, metadata = (
            self.feature_engineer.compute(obs, player_id)
        )

        # 转 tensor
        planet_feat_t = torch.from_numpy(planet_features).unsqueeze(0).to(self.device)
        global_feat_t = torch.from_numpy(global_features).unsqueeze(0).to(self.device)

        if fleet_features.shape[0] > 0:
            fleet_feat_t = torch.from_numpy(fleet_features).unsqueeze(0).to(self.device)
        else:
            fleet_feat_t = torch.zeros(1, 0, fleet_features.shape[1], device=self.device)

        # 构建 mask
        raw_planets = obs.get("planets", [])
        n_planets_actual = len(raw_planets)

        owned_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)
        enemy_mask = torch.zeros(1, n_planets_actual, dtype=torch.bool, device=self.device)

        for i, p in enumerate(raw_planets):
            owner = int(p[1])
            if owner == player_id:
                owned_mask[0, i] = True
            elif owner != -1:
                enemy_mask[0, i] = True

        # num_players
        owners = set()
        for p in raw_planets:
            o = int(p[1])
            if 0 <= o <= 3:
                owners.add(o)
        num_players_val = max(len(owners), 2)
        num_players = torch.tensor([num_players_val], dtype=torch.long, device=self.device)

        # planet_ships
        planet_ships = torch.tensor(
            [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=self.device
        )

        # 模型推理
        with torch.no_grad():
            target_logits, num_ships_out, value, opp_target, opp_num_ships = self.model(
                planet_features=planet_feat_t,
                fleet_features=fleet_feat_t,
                global_features=global_feat_t,
                owned_mask=owned_mask,
                enemy_mask=enemy_mask,
                num_players=num_players,
                planet_ships=planet_ships,
            )

            # 截断或填充 target_logits 以匹配实际星球数量
            n_model_planets = target_logits.shape[2]
            if n_planets_actual < n_model_planets:
                target_logits = target_logits[:, :, :n_planets_actual]
            elif n_planets_actual > n_model_planets:
                pad_size = n_planets_actual - n_model_planets
                padding = torch.zeros(1, target_logits.shape[1], pad_size, device=self.device)
                target_logits = torch.cat([target_logits, padding], dim=2)

        # 转 numpy
        target_logits_np = target_logits.cpu().numpy()
        num_ships_np = num_ships_out.cpu().numpy()

        if target_logits_np.shape[1] == 0:
            return []

        target_logits_np = target_logits_np[0]
        num_ships_np = num_ships_np[0]

        # 构建星球列表
        all_planets = []
        owned_planets = []
        for p in raw_planets:
            pid, owner, x, y, radius, ships, production = p
            planet_dict = {
                "id": int(pid),
                "owner": int(owner),
                "x": float(x),
                "y": float(y),
                "radius": float(radius),
                "ships": float(ships),
                "production": float(production),
            }
            all_planets.append(planet_dict)
            if int(owner) == player_id:
                owned_planets.append(planet_dict)

        # 将绝对数量转回比例
        num_ships_raw_for_decode = np.zeros_like(num_ships_np)
        for i, src in enumerate(owned_planets):
            src_ships = src.get("ships", 0)
            if src_ships > 0:
                num_ships_raw_for_decode[i, 0] = num_ships_np[i, 0] / src_ships
            else:
                num_ships_raw_for_decode[i, 0] = 0.0

        if len(owned_planets) == 0 or len(all_planets) == 0:
            return []

        if target_logits_np.shape[1] != len(all_planets):
            return []

        actions = decode_actions(
            target_logits=target_logits_np,
            num_ships_raw=num_ships_raw_for_decode,
            owned_planets=owned_planets,
            all_planets=all_planets,
            threshold=0.01,
        )

        if len(actions) > 0:
            print(f"  [Player {player_id}] 产出 {len(actions)} 个动作: {actions[:3]}...")
        else:
            print(f"  [Player {player_id}] 无动作 (owned={len(owned_planets)}, all={len(all_planets)})")

        return actions


# ---------------------------------------------------------------------------
# 主评估流程
# ---------------------------------------------------------------------------
def run_evaluation(
    player0,
    player1,
    player0_name: str = "Player0",
    player1_name: str = "Player1",
    num_games: int = 10,
    replay_path: str | None = None,
):
    """运行多局对战，计算胜率。player0/player1 可以是 ModelAgent 或 callable"""
    wins = 0
    losses = 0
    draws = 0
    results = []
    replay_env = None

    for game_idx in range(num_games):
        seed = 42 + game_idx
        env = make("orbit_wars", configuration={"seed": seed}, debug=True)
        env.run([player0, player1])

        final = env.steps[-1]
        p0_reward = final[0].reward
        p1_reward = final[1].reward

        if p0_reward > p1_reward:
            wins += 1
            result_str = "胜"
        elif p0_reward < p1_reward:
            losses += 1
            result_str = "负"
        else:
            draws += 1
            result_str = "平"

        results.append({
            "game": game_idx + 1,
            "seed": seed,
            "p0_reward": p0_reward,
            "p1_reward": p1_reward,
            "result": result_str,
        })

        print(f"  第 {game_idx+1}/{num_games} 局 (seed={seed}): "
              f"{player0_name}={p0_reward} vs {player1_name}={p1_reward} -> {result_str}")

        if game_idx == 0:
            replay_env = env

    print("\n" + "=" * 60)
    print(f"评估结果汇总 ({num_games} 局): {player0_name} vs {player1_name}")
    print(f"  {player0_name} 胜: {wins} 局 ({wins/num_games*100:.1f}%)")
    print(f"  {player0_name} 负: {losses} 局 ({losses/num_games*100:.1f}%)")
    print(f"  平局:   {draws} 局 ({draws/num_games*100:.1f}%)")
    print("=" * 60)

    if replay_env is not None and replay_path:
        html = replay_env.render(mode="html")
        full_path = os.path.join(_PROJECT_ROOT, replay_path)
        with open(full_path, "w") as f:
            f.write(html)
        print(f"回放文件已保存: {full_path}")

    return results


if __name__ == "__main__":
    import os

    ckpt_dir = os.path.join(_PROJECT_ROOT, "training/checkpoints")

    # ---- 测试 1: iter_100 vs 最近星球策略 ----
    ckpt_100 = os.path.join(ckpt_dir, "pretrain_iter_100.pkl")
    if os.path.exists(ckpt_100):
        print("\n" + "#" * 60)
        print("# 测试 1: iter_100 vs 最近星球策略")
        print("#" * 60)
        model_100 = ModelAgent(ckpt_100)
        run_evaluation(
            model_100, nearest_planet_agent,
            player0_name="iter_100",
            player1_name="最近星球",
            num_games=10,
            replay_path="replay_iter100_vs_nearest.html",
        )
    else:
        print(f"检查点不存在: {ckpt_100}")

    # ---- 测试 2: iter_50 vs iter_100 ----
    ckpt_50 = os.path.join(ckpt_dir, "pretrain_iter_50.pkl")
    if os.path.exists(ckpt_50) and os.path.exists(ckpt_100):
        print("\n" + "#" * 60)
        print("# 测试 2: iter_50 vs iter_100")
        print("#" * 60)
        model_50 = ModelAgent(ckpt_50)
        model_100 = ModelAgent(ckpt_100)
        run_evaluation(
            model_50, model_100,
            player0_name="iter_50",
            player1_name="iter_100",
            num_games=10,
            replay_path="replay_iter50_vs_iter100.html",
        )

    # ---- 测试 3: 专家策略 vs 最近星球策略 ----
    from training.expert.kaggle_expert import KaggleExpertAgent

    _expert = KaggleExpertAgent()

    def expert_agent(obs, configuration=None):
        """Kaggle 环境适配器"""
        return _expert.get_actions(obs)

    print("\n" + "#" * 60)
    print("# 测试 3: 专家策略 vs 最近星球策略")
    print("#" * 60)
    run_evaluation(
        expert_agent, nearest_planet_agent,
        player0_name="KaggleExpert",
        player1_name="最近星球",
        num_games=10,
        replay_path="replay_expert_vs_nearest.html",
    )

    # ---- 测试 4: 专家策略 vs iter_100 ----
    print("\n" + "#" * 60)
    print("# 测试 4: 专家策略 vs iter_100")
    print("#" * 60)
    if os.path.exists(ckpt_100):
        model_100 = ModelAgent(ckpt_100)
        run_evaluation(
            expert_agent, model_100,
            player0_name="KaggleExpert",
            player1_name="iter_100",
            num_games=10,
            replay_path="replay_expert_vs_iter100.html",
        )
