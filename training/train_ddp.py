"""分布式训练主循环 — 8卡 DDP + PPO 自我博弈 + 对手池进化。"""
from __future__ import annotations

import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

from core.config import AppConfig
from core.model import OrbitWarsModel
from core.feature_engineering import FeatureEngineer
from core.reward import RewardCalculator, RewardConfig
from core.ppo import PPOTrainer
from core.rollout import RolloutWorker, parallel_rollout
from core.rollout_mp import MultiProcessRolloutWorker, parallel_rollout_mp
from core.opponent_pool import OpponentPool, OpponentPoolConfig
from core.monitoring import TrainingMonitor
from core.checkpoint_manager import CheckpointManager


def setup_ddp():
    """初始化分布式训练环境。"""
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        print("[DDP] 未检测到分布式环境变量，使用单卡模式")
        return 0, 1, 0  # rank, world_size, local_rank

    # 初始化进程组
    dist.init_process_group(
        backend='nccl',
        init_method='env://',
        rank=rank,
        world_size=world_size
    )

    torch.cuda.set_device(local_rank)
    print(f"[DDP] Rank {rank}/{world_size} 初始化完成 (local_rank={local_rank})")

    return rank, world_size, local_rank


def cleanup_ddp():
    """清理分布式训练环境。"""
    if dist.is_initialized():
        dist.destroy_process_group()
        print("[DDP] 进程组已销毁")


def _load_swanlab_key():
    """加载SwanLab API key。"""
    key_file = Path(__file__).parent / "config" / "swanlab_key.txt"
    if key_file.exists():
        return key_file.read_text().strip()
    return None


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
                    # 如果是DDP模型，需要访问module
                    model_obj = model.module if isinstance(model, DDP) else model
                    target_logits, num_ships_out, _, _, _ = model_obj(
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

                # 需要将绝对飞船数转回比例给 decode_actions
                ns_raw = np.zeros_like(ns)
                for i, src in enumerate(owned_p):
                    if src["ships"] > 0:
                        ns_raw[i, 0] = ns[i, 0] / src["ships"]

                actions = decode_actions(tl, ns_raw, owned_p, all_planets_dicts, threshold=0.3)
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


def train(
    config_path: str = "training/config/default.yaml",
    rank: int = 0,
    world_size: int = 1,
    local_rank: int = 0,
):
    """主训练入口（支持分布式）。"""

    # 加载配置
    config_path = Path(config_path)
    if config_path.exists():
        config = AppConfig.from_yaml(config_path)
    else:
        config = AppConfig()

    device = f"cuda:{local_rank}"
    print(f"[Rank {rank}] device={device}")

    # SwanLab (只在 rank 0 初始化)
    swanlab = None
    if rank == 0:
        swanlab = _try_swanlab()
        if swanlab:
            api_key = _load_swanlab_key()
            if not api_key:
                print("[Rank 0] 警告: 未找到SwanLab API key，将尝试使用环境变量")
            else:
                print(f"[Rank 0] 已加载SwanLab API key")

            init_kwargs = {
                "project": config.training.swanlab_project,
                "experiment_name": f"{config.training.swanlab_experiment}_8gpu",
                "config": {
                    "model": vars(config.model),
                    "training": vars(config.training),
                    "distributed": {"world_size": world_size},
                },
            }

            # 确定mode：优先使用配置文件，其次根据api_key
            mode = getattr(config.training, 'swanlab_mode', None)
            if mode:
                init_kwargs["mode"] = mode
            elif api_key:
                init_kwargs["api_key"] = api_key
                init_kwargs["mode"] = "cloud"
            else:
                init_kwargs["mode"] = "local"

            swanlab.init(**init_kwargs)
            print(f"[Rank 0] SwanLab initialized: {config.training.swanlab_project}")
        else:
            print("[Rank 0] SwanLab not available, using print logging")

    def log_metrics(metrics: dict, step: int):
        if rank == 0 and swanlab:
            swanlab.log(metrics, step=step)
        elif rank == 0:
            print(f"[Step {step}] " + " | ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # 设置随机种子（每个进程使用不同的种子）
    seed = getattr(config.training, 'seed', 42) + rank
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 初始化模型
    model = OrbitWarsModel(
        config.model,
        n_planets=40,
        max_players=4,
    ).to(device)

    # 包装为 DDP 模型
    if world_size > 1:
        model = DDP(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True,  # 某些参数可能不参与梯度计算
        )
        print(f"[Rank {rank}] DDP 模型已包装")

    # 初始化特征工程和奖励计算器（每个进程独立）
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

    # 训练器（使用DDP模型）
    trainer = PPOTrainer(model, config.training, device=device)

    # Rollout worker - 使用多进程版本充分利用CPU核心
    if world_size == 1:
        # 单GPU模式：使用多进程worker（充分利用128个CPU核心）
        worker = MultiProcessRolloutWorker(
            model, feature_engineer, reward_calculator, device=device,
            num_workers=64  # 使用64个worker进程
        )
        use_mp_rollout = True
        if rank == 0:
            print(f"[Rank 0] 使用多进程Rollout Worker（64个进程）")
    else:
        # 多GPU模式：使用普通worker（避免进程过多）
        worker = RolloutWorker(model, feature_engineer, reward_calculator, device=device)
        use_mp_rollout = False

    # 对手池（只在 rank 0 管理）
    pool = None
    if rank == 0:
        pool_config = OpponentPoolConfig(
            pool_size=config.self_play.pool_size,
            sample_latest_ratio=config.self_play.sample_latest_ratio,
            sample_random_ratio=config.self_play.sample_random_ratio,
            sample_best_ratio=config.self_play.sample_best_ratio,
            sample_heuristic_ratio=config.self_play.sample_heuristic_ratio,
            add_to_pool_win_rate=config.self_play.add_to_pool_win_rate,
            diversity_threshold=config.self_play.diversity_threshold,
        )
        pool = OpponentPool(config=pool_config, pool_dir="checkpoints/pool")
        pool.load_index()
        pool.add_heuristic("random")

    # Checkpoint 目录
    ckpt_dir = Path("checkpoints")
    if rank == 0:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    # 初始化训练监控器（只在 rank 0）
    monitor = None
    if rank == 0:
        monitor = TrainingMonitor(device=device)

    # 初始化checkpoint管理器（只在 rank 0）
    ckpt_manager = None
    start_iteration = 0
    if rank == 0:
        max_ckpt = getattr(config.training, 'max_checkpoints', 5)
        keep_best = getattr(config.training, 'keep_best_n', 2)
        ckpt_manager = CheckpointManager(
            checkpoint_dir=ckpt_dir,
            max_checkpoints=max_ckpt,
            keep_best_n=keep_best,
        )
        stats = ckpt_manager.get_stats()
        if stats['total_checkpoints'] > 0:
            print(f"[Rank 0] 找到 {stats['total_checkpoints']} 个已有checkpoint")

        # 检查是否需要从checkpoint恢复
        if config.training.resume_from_checkpoint:
            latest_ckpt = ckpt_manager.get_latest_checkpoint()
            if latest_ckpt:
                print(f"[Rank 0] 从checkpoint恢复训练: {latest_ckpt}")
                try:
                    checkpoint = torch.load(latest_ckpt, map_location=device)
                    model_state = checkpoint.get("model_state_dict", {})
                    target_model = model.module if isinstance(model, DDP) else model

                    # 加载模型权重
                    if model_state:
                        try:
                            target_model.load_state_dict(model_state)
                            start_iteration = checkpoint.get("iteration", 0)
                            print(f"[Rank 0] ✓ 成功加载checkpoint，从 iteration {start_iteration} 继续")
                        except Exception as e:
                            print(f"[Rank 0] ✗ 加载checkpoint失败: {e}")
                            print(f"[Rank 0]  将从头开始训练")
                            start_iteration = 0
                except Exception as e:
                    print(f"[Rank 0] ✗ 读取checkpoint文件失败: {e}")
                    print(f"[Rank 0]  将从头开始训练")
                    start_iteration = 0
            else:
                print(f"[Rank 0] 未找到checkpoint，从头开始训练")
        else:
            print(f"[Rank 0] resume_from_checkpoint=False，从头开始训练")

    # 同步所有进程（确保初始化完成）
    if world_size > 1:
        dist.barrier()

    if rank == 0:
        model_params = model.module if isinstance(model, DDP) else model
        print(f"[Rank 0] Model params: {sum(p.numel() for p in model_params.parameters()):,}")
        if start_iteration > 0:
            print(f"[Rank 0] 从 iteration {start_iteration} 继续训练，目标 {config.training.max_iterations}")
        else:
            print(f"[Rank 0] Starting training for {config.training.max_iterations} iterations with {world_size} GPUs")

        # 打印训练配置（单卡模式）
        if world_size == 1:
            print(f"[Rank 0] 训练配置:")
            print(f"[Rank 0]   - 批次大小: {config.training.batch_size}")
            print(f"[Rank 0]   - 并行游戏: {config.training.num_parallel_games}")
            print(f"[Rank 0]   - PPO epochs: {config.training.ppo_epochs}")
            print(f"[Rank 0]   - 学习率: {config.training.learning_rate}")
            print(f"[Rank 0]   - 保存间隔: 每 {config.training.save_interval} 次迭代")
            print(f"[Rank 0] ========================================")

    # 训练循环
    for iteration in range(start_iteration, config.training.max_iterations):
        t0 = time.time()

        # 1. 决定 2 人 / 4 人局
        num_players = 2 if random.random() < config.self_play.two_player_prob else 4

        # 2. 并行 rollout（每个进程独立采集）
        # 将总游戏数分配到各个进程
        games_per_rank = max(1, config.training.num_parallel_games // world_size)

        # 使用多进程rollout（单GPU模式）
        if use_mp_rollout:
            buffer = parallel_rollout_mp(
                worker,
                num_games=games_per_rank,
                num_players=num_players,
                temperature=max(1.0 - iteration * 0.001, 0.3),
            )
        else:
            buffer = parallel_rollout(
                worker,
                num_games=games_per_rank,
                num_players=num_players,
                temperature=max(1.0 - iteration * 0.001, 0.3),
            )

        rollout_time = time.time() - t0

        # 3. 收集所有进程的经验（DDP中每个进程独立更新，不需要手动聚合）
        # 但如果需要全局同步，可以使用 all_gather

        if len(buffer) == 0:
            if rank == 0:
                print(f"[Iter {iteration}] Empty buffer, skipping")
            continue

        # 4. PPO 更新（每个进程独立更新自己的模型副本）
        metrics = trainer.update(buffer)

        # 计算reward统计
        reward_stats = trainer.compute_buffer_stats(buffer)
        metrics.update(reward_stats)

        metrics["rollout_time"] = rollout_time
        metrics["buffer_size"] = len(buffer)
        metrics["num_players"] = num_players

        # 5. 定期评估 & 保存（只在 rank 0）
        if iteration % config.training.save_interval == 0 and iteration > 0:
            if rank == 0:
                ckpt_path = ckpt_dir / f"model_iter_{iteration}.pt"
                model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
                torch.save({
                    "model_state_dict": model_state,
                    "config": config.model,
                    "n_planets": 40,
                    "iteration": iteration,
                }, ckpt_path)
                metrics["checkpoint_saved"] = str(ckpt_path)
                print(f"[Iter {iteration}] Checkpoint saved: {ckpt_path}")

                # 评估
                eval_games = min(50, config.training.num_parallel_games // world_size)
                win_rate = evaluate(
                    model, feature_engineer, device,
                    num_games=eval_games, num_players=4,
                )
                metrics["eval_win_rate"] = win_rate

                # 更新checkpoint管理器
                if ckpt_manager:
                    ckpt_manager.add_checkpoint(iteration, win_rate)
                    ckpt_manager.update_best_checkpoint(iteration, win_rate)
                    # 清理旧checkpoint
                    ckpt_manager.cleanup_old_checkpoints()

                # 加入对手池
                if pool.should_add(win_rate):
                    pool.add(str(ckpt_path), elo=600 + win_rate * 200, generation=iteration)
                    print(f"[Iter {iteration}] Added to pool: win_rate={win_rate:.3f}")

                pool_stats = pool.get_stats()
                metrics["pool_size"] = pool_stats["size"]
                metrics["pool_best_elo"] = pool_stats.get("best_elo", 0)

                # 添加checkpoint统计
                if ckpt_manager:
                    ckpt_stats = ckpt_manager.get_stats()
                    metrics["checkpoint_count"] = ckpt_stats["total_checkpoints"]
                    metrics["checkpoint_size_mb"] = ckpt_stats["total_size_mb"]

            # 同步所有进程
            if world_size > 1:
                dist.barrier()

        # 添加监控指标（只在 rank 0）
        if rank == 0 and monitor:
            metrics = monitor.update_metrics_with_monitoring(
                metrics,
                num_games=config.training.num_parallel_games // world_size,
                rollout_time=rollout_time,
            )

        log_metrics(metrics, iteration)

        # 定期打印训练信息
        if iteration % 10 == 0 and rank == 0:
            elapsed = time.time() - t0
            iter_time = rollout_time + (time.time() - t0 - rollout_time)
            samples_per_sec = len(buffer) / iter_time if iter_time > 0 else 0

            print(f"[Iter {iteration}/{config.training.max_iterations}] "
                  f"loss={metrics.get('total_loss', 0):.4f}, "
                  f"reward={metrics.get('mean_reward', 0):.4f}, "
                  f"buffer={len(buffer)}, "
                  f"time={iter_time:.2f}s, "
                  f"throughput={samples_per_sec:.0f} samples/s")

            # 单卡模式：打印更详细的GPU信息
            if world_size == 1 and iteration % 50 == 0:
                if torch.cuda.is_available():
                    gpu_mem = torch.cuda.memory_allocated(0) / 1024**3
                    gpu_cached = torch.cuda.memory_reserved(0) / 1024**3
                    print(f"[Iter {iteration}] GPU: {gpu_mem:.2f}GB allocated, {gpu_cached:.2f}GB reserved")

    # 保存最终模型（只在 rank 0）
    if rank == 0:
        final_path = ckpt_dir / "model_final.pt"
        model_state = model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
        torch.save({
            "model_state_dict": model_state,
            "config": config.model,
            "n_planets": 40,
        }, final_path)
        print(f"[Rank 0] Final model saved to {final_path}")

        # 同时保存为 model.pt (Kaggle 提交用)
        submit_path = Path("model.pt")
        torch.save({
            "model_state_dict": model_state,
            "config": config.model,
            "n_planets": 40,
        }, submit_path)
        print(f"[Rank 0] Submission model saved to {submit_path}")

    if rank == 0 and swanlab:
        swanlab.finish()


if __name__ == "__main__":
    # 初始化分布式环境
    rank, world_size, local_rank = setup_ddp()

    try:
        # 开始训练
        train(
            config_path="training/config/default.yaml",
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
        )
    finally:
        # 清理分布式环境
        cleanup_ddp()
