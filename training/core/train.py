"""训练主循环 — PPO 自我博弈 + 对手池进化。"""
from __future__ import annotations

import random
import time
from pathlib import Path

import numpy as np
import torch

from core.config import AppConfig
from core.model import OrbitWarsModel
from core.feature_engineering import FeatureEngineer
from core.reward import RewardCalculator, RewardConfig
from core.ppo import PPOTrainer
from core.rollout import RolloutWorker, parallel_rollout
from core.opponent_pool import OpponentPool, OpponentPoolConfig


def _try_swanlab():
    """尝试导入 SwanLab，返回 None 如果不可用。"""
    try:
        import swanlab
        return swanlab
    except ImportError:
        return None


def evaluate(
    model: OrbitWarsModel,
    feature_engineer: FeatureEngineer,
    device: str = "cuda",
    num_games: int = 200,
    num_players: int = 4,
) -> float:
    """评估当前策略 vs 自身（self-play），返回 player 0 胜率。"""
    from core.env_wrapper import OrbitWarsEnv
    from core.action import decode_actions

    model.eval()
    wins = 0

    for _ in range(num_games):
        env = OrbitWarsEnv(num_players=num_players)
        observations = env.reset()

        while not env.done:
            all_actions: dict[int, list] = {}
            for pid in range(num_players):
                obs_dict = env.get_raw_observation(pid)
                planet_feat, fleet_feat, global_feat, metadata = (
                    feature_engineer.compute(obs_dict, pid)
                )

                planet_feat_t = torch.from_numpy(planet_feat).unsqueeze(0).to(device)
                fleet_feat_t = (
                    torch.from_numpy(fleet_feat).unsqueeze(0).to(device)
                    if fleet_feat.shape[0] > 0
                    else torch.zeros(1, 0, 11, device=device)
                )
                global_feat_t = torch.from_numpy(global_feat).unsqueeze(0).to(device)

                raw_planets = obs_dict.get("planets", [])
                n = len(raw_planets)
                owned_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
                enemy_mask = torch.zeros(1, n, dtype=torch.bool, device=device)
                for i, p in enumerate(raw_planets):
                    if int(p[1]) == pid:
                        owned_mask[0, i] = True
                    elif int(p[1]) != -1:
                        enemy_mask[0, i] = True

                planet_ships = torch.tensor(
                    [[float(p[5]) for p in raw_planets]], dtype=torch.float32, device=device
                )
                num_players_t = torch.tensor([num_players], dtype=torch.long, device=device)

                with torch.no_grad():
                    target_logits, num_ships_out, _, _, _ = model(
                        planet_features=planet_feat_t,
                        fleet_features=fleet_feat_t,
                        global_features=global_feat_t,
                        owned_mask=owned_mask,
                        enemy_mask=enemy_mask,
                        num_players=num_players_t,
                        planet_ships=planet_ships,
                    )

                tl = target_logits[0].cpu().numpy()
                ns = num_ships_out[0].cpu().numpy()

                all_planets_dicts = [
                    {"id": int(p[0]), "x": float(p[2]), "y": float(p[3]),
                     "ships": float(p[5]), "owner": int(p[1]), "production": float(p[6])}
                    for p in raw_planets
                ]
                owned_p = [p for p in all_planets_dicts if p["owner"] == pid]

                # 模型输出 sigmoid 比例，decode_actions 内部乘以飞船数
                actions = decode_actions(tl, ns, owned_p, all_planets_dicts, threshold=0.3)
                all_actions[pid] = actions

            env.step(all_actions)

        # 判断 player 0 是否获胜
        scores = {}
        for pid in range(num_players):
            obs_dict = env.get_raw_observation(pid)
            planets = obs_dict.get("planets", [])
            fleets = obs_dict.get("fleets", [])
            s = sum(float(p[5]) for p in planets if int(p[1]) == pid)
            s += sum(float(f[6]) for f in fleets if int(f[1]) == pid)
            scores[pid] = s

        if scores.get(0, 0) >= max(scores.values()):
            wins += 1

    model.train()
    return wins / max(num_games, 1)


def train(config_path: str = "training/config/default.yaml"):
    """主训练入口。"""
    # 加载配置
    config_path = Path(config_path)
    if config_path.exists():
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Train] device={device}")

    # SwanLab
    swanlab = _try_swanlab()
    if swanlab:
        swanlab.init(
            project=config.training.swanlab_project,
            experiment_name=config.training.swanlab_experiment,
            config={
                "model": vars(config.model),
                "training": vars(config.training),
            },
        )
        print(f"[Train] SwanLab initialized: {config.training.swanlab_project}")
    else:
        print("[Train] SwanLab not available, using print logging")

    def log_metrics(metrics: dict, step: int):
        if swanlab:
            swanlab.log(metrics, step=step)
        else:
            print(f"[Step {step}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # 初始化组件
    model = OrbitWarsModel(
        config.model,
        n_planets=40,
        max_players=4,
    ).to(device)

    pretrained_ckpt = Path("training/checkpoints/pretrained_model.pkl")
    if config.expert_data.enabled and pretrained_ckpt.exists():
        ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
        model.load_state_dict(state_dict, strict=False)
        print(f"[Train] Loaded pretrained weights: {pretrained_ckpt}")

    feature_engineer = FeatureEngineer(
        board_size=config.environment.board_size,
        sun_radius=config.environment.sun_radius,
        max_speed=config.environment.ship_speed,
        max_turns=config.environment.episode_steps,
    )

    reward_calculator = RewardCalculator(
        config=RewardConfig(
            terminal_weight=config.reward.terminal_weight,
            intermediate_weight=config.reward.intermediate_weight,
            capture_reward_weight=config.reward.capture_reward_weight,
            loss_penalty_weight=config.reward.loss_penalty_weight,
            defense_reward_weight=config.reward.defense_reward_weight,
            transit_cost_weight=config.reward.transit_cost_weight,
            production_advantage_weight=config.reward.production_advantage_weight,
            comet_roi_weight=config.reward.comet_roi_weight,
        ),
        max_turns=config.environment.episode_steps,
    )

    trainer = PPOTrainer(model, config.training, device=device)
    worker = RolloutWorker(
        model,
        feature_engineer,
        reward_calculator,
        device=device,
        num_feature_workers=config.training.num_feature_workers,
        enable_rollout_timing=config.training.enable_rollout_timing,
    )

    pool_config = OpponentPoolConfig(
        pool_size=config.self_play.pool_size,
        sample_expert_ratio=config.self_play.sample_expert_ratio,
        sample_heuristic_ratio=config.self_play.sample_heuristic_ratio,
        sample_checkpoint_ratio=config.self_play.sample_checkpoint_ratio,
        checkpoint_latest_ratio=config.self_play.checkpoint_latest_ratio,
        checkpoint_best_ratio=config.self_play.checkpoint_best_ratio,
        checkpoint_random_ratio=config.self_play.checkpoint_random_ratio,
        add_to_pool_win_rate=config.self_play.add_to_pool_win_rate,
        diversity_threshold=config.self_play.diversity_threshold,
    )
    pool = OpponentPool(config=pool_config, pool_dir="checkpoints/pool")
    pool.load_index()

    # 注册专家策略
    pool.register_default_experts()

    # 将已有 pretrained / 历史 RL checkpoint 自动加入对手池
    pool.add_checkpoint_dir("training/checkpoints", elo=580, generation=0)
    pool.add_checkpoint_dir("checkpoints", patterns=("model_iter_*.pt", "model_final.pt"), elo=600, generation=0)
    pool_stats = pool.get_stats()
    print(
        "[Train] Opponent pool: "
        f"{pool.size} checkpoints, "
        f"kaggle={pool_stats.get('kaggle_experts', [])}, "
        f"debug={pool_stats.get('debug_experts', [])}"
    )

    # Checkpoint 目录
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    print(f"[Train] Model params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"[Train] Starting training for {config.training.max_iterations} iterations")

    # 训练循环
    for iteration in range(config.training.max_iterations):
        t0 = time.time()

        # 1. 决定 2 人 / 4 人局
        num_players = 2 if random.random() < config.self_play.two_player_prob else 4

        # 2. 并行 rollout（使用对手池采样对手）
        buffer = parallel_rollout(
            worker,
            num_games=min(config.training.num_parallel_games, 32),
            num_players=num_players,
            temperature=max(1.0 - iteration * 0.001, 0.3),
            opponent_pool=pool,
        )

        rollout_time = time.time() - t0

        if len(buffer) == 0:
            print(f"[Iter {iteration}] Empty buffer, skipping")
            continue

        # 3. PPO 更新
        metrics = trainer.update(buffer)
        metrics["rollout_time"] = rollout_time
        metrics["buffer_size"] = len(buffer)
        metrics["num_players"] = num_players

        # 4. 定期评估 & 保存
        if iteration % config.training.save_interval == 0 and iteration > 0:
            ckpt_path = ckpt_dir / f"model_iter_{iteration}.pt"
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config.model,
                "n_planets": 40,
                "iteration": iteration,
            }, ckpt_path)
            metrics["checkpoint_saved"] = str(ckpt_path)

            # 评估
            eval_games = min(50, config.training.num_parallel_games)
            win_rate = evaluate(
                model, feature_engineer, device,
                num_games=eval_games, num_players=4,
            )
            metrics["eval_win_rate"] = win_rate

            # 加入对手池
            if pool.should_add(win_rate):
                pool.add(str(ckpt_path), elo=600 + win_rate * 200, generation=iteration)
                print(f"[Iter {iteration}] Added to pool: win_rate={win_rate:.3f}")

            pool_stats = pool.get_stats()
            metrics["pool_size"] = pool_stats["size"]
            metrics["pool_best_elo"] = pool_stats.get("best_elo", 0)

        log_metrics(metrics, iteration)

        if iteration % 100 == 0:
            elapsed = time.time() - t0
            print(f"[Iter {iteration}] elapsed={elapsed:.1f}s buffer={len(buffer)}")

    # 保存最终模型
    final_path = ckpt_dir / "model_final.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.model,
        "n_planets": 40,
    }, final_path)
    print(f"[Train] Final model saved to {final_path}")

    # 同时保存为 model.pt (Kaggle 提交用)
    submit_path = Path("model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": config.model,
        "n_planets": 40,
    }, submit_path)
    print(f"[Train] Submission model saved to {submit_path}")

    if swanlab:
        swanlab.finish()


if __name__ == "__main__":
    train()
