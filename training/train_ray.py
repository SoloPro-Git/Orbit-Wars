"""Ray 分布式训练 - Rollout 和 PPO 训练分离。

架构:
  - Trainer Actor: 独占 GPU，负责 PPO 更新和模型管理
  - Rollout Workers: 多个 Actor 并行执行 rollout，共享 GPU
  - 异步执行: Rollout 完成后立即返回数据，Trainer 持续训练

配置示例 (config/default.yaml):
  ray:
    num_rollout_workers: 8     # 8个 rollout workers
    num_gpus_per_worker: 0.25  # 每个 worker 使用 1/4 GPU
    trainer_num_gpus: 1        # trainer 使用 1 个完整 GPU

实际 GPU 分配 (4卡系统):
  - Workers 0,1 → GPU 0
  - Workers 2,3 → GPU 1
  - Workers 4,5 → GPU 2
  - Workers 6,7 → GPU 3
  - Trainer → GPU 0 (共享)

使用方式:
  # 单卡测试
  CUDA_VISIBLE_DEVICES=0 python train_ray.py

  # 4卡训练
  CUDA_VISIBLE_DEVICES=0,1,2,3 python train_ray.py

  # 自定义配置
  python train_ray.py --rollout-workers 16 --gpus-per-worker 0.125
"""
from __future__ import annotations

import sys
from pathlib import Path

# 禁用 kaggle_environments 的冗长日志
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['KAGGLE_ENGINES_LOG_LEVEL'] = '0'

# ⚠️ 禁用 SwanLab 交互式提示 (必须在导入 swanlab 之前设置)
os.environ['SWANLAB_NO_INTERACTIVE'] = '1'
os.environ['SWANLAB_DISABLE_INTERACTIVE'] = '1'


# 设置导入路径（支持从 training 目录运行）
script_dir = Path(__file__).parent.resolve()
project_root = script_dir.parent.resolve()
training_dir = script_dir

if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

if str(training_dir) not in sys.path:
    sys.path.insert(0, str(training_dir))

os.chdir(project_root)

# 先导入 core 模块以触发 __init__.py 的路径设置
import core  # noqa: F401

import random
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

try:
    import ray
    from ray import serve
    from ray.util.queue import Queue as RayQueue
except ImportError:
    print("Ray 未安装，请运行: pip install ray")
    exit(1)

from core.config import AppConfig
from core.model import OrbitWarsModel
from core.feature_engineering import FeatureEngineer
from core.reward import RewardCalculator, RewardConfig
from core.ppo import PPOTrainer, PPOBuffer
from core.rollout import RolloutWorker, parallel_rollout
from core.opponent_pool import OpponentPool, OpponentPoolConfig

import logging
logging.getLogger('kaggle_environments').setLevel(logging.WARNING)

# 进度条
from tqdm import tqdm


# ==================== Ray Actors ====================

@ray.remote
class TrainingActor:
    """
    训练 Actor - 负责 PPO 更新和模型管理。

    独占一个 GPU，持续从 rollout workers 收集数据并更新模型。
    """

    def __init__(
        self,
        config_path: str,
        device: str = "cuda:0",
    ):
        print("[TrainingActor] 初始化...")

        # 加载配置
        config_path = Path(config_path)
        if config_path.exists():
            self.config = AppConfig.from_yaml(config_path)
        else:
            self.config = AppConfig()

        self.device = device

        # 设置随机种子
        torch.manual_seed(42)
        np.random.seed(42)
        random.seed(42)

        # 初始化模型
        self.model = OrbitWarsModel(
            self.config.model,
            n_planets=40,
            max_players=4,
        ).to(self.device)

        pretrained_ckpt = Path("training/checkpoints/pretrained_model.pkl")
        if self.config.expert_data.enabled and pretrained_ckpt.exists():
            try:
                ckpt = torch.load(pretrained_ckpt, map_location="cpu", weights_only=False)
                state_dict = ckpt.get("model_state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
                self.model.load_state_dict(state_dict, strict=False)
                print(f"[TrainingActor] 已加载预训练权重: {pretrained_ckpt}")
            except Exception as e:
                print(f"[TrainingActor] 预训练权重加载失败: {e}")

        # 训练器
        self.trainer = PPOTrainer(self.model, self.config.training, device=self.device)

        # SwanLab
        self.swanlab = None
        # 检查环境变量中是否有 API key
        if os.environ.get('SWANLAB_API_KEY'):
            try:
                import swanlab
                self.swanlab = swanlab

                # 使用 launcher=False 禁用交互式 launcher
                self.swanlab.init(
                    project=self.config.training.swanlab_project,
                    experiment_name=f"{self.config.training.swanlab_experiment}_ray",
                    config={
                        "model": vars(self.config.model),
                        "training": vars(self.config.training),
                        "ray": vars(self.config.ray),
                    },
                    mode=getattr(self.config.training, 'swanlab_mode', 'cloud'),
                    logdir=None,
                    public=False,
                    launcher=False,  # 禁用交互式 launcher
                )
                print("[TrainingActor] SwanLab initialized")
            except Exception as e:
                print(f"[TrainingActor] SwanLab 初始化失败: {e}")
                # 失败时不影响训练
                self.swanlab = None

        self.iteration = 0
        print(f"[TrainingActor] 初始化完成，参数量: {sum(p.numel() for p in self.model.parameters()):,}")

    def get_model_state(self) -> Dict:
        """获取当前模型权重。"""
        import time
        print(f"[DEBUG][TrainingActor] get_model_state() 开始复制模型到 CPU...")
        t0 = time.time()
        state = {k: v.cpu() for k, v in self.model.state_dict().items()}
        elapsed = time.time() - t0
        print(f"[DEBUG][TrainingActor] get_model_state() 完成，耗时 {elapsed:.2f}s")
        return state

    def set_model_state(self, model_state: Dict) -> None:
        """设置模型权重。"""
        self.model.load_state_dict(model_state)

    def update(self, buffers: List[Dict], num_players: int) -> Dict:
        """
        执行 PPO 更新。

        Args:
            buffers: 来自多个 rollout workers 的 buffer 数据
            num_players: 游戏玩家数

        Returns:
            训练指标
        """
        # 汇总所有 buffer
        combined_buffer = PPOBuffer()
        total_size = 0

        for buffer_data in buffers:
            size = buffer_data.get('size', 0)
            if size == 0:
                continue

            for attr in [
                "planet_features", "fleet_features", "global_features",
                "owned_masks", "enemy_masks", "num_players_list",
                "planet_ships_list", "target_indices", "num_ships_actual",
                "log_probs", "values", "rewards", "dones",
                "opp_target_indices",
            ]:
                if attr in buffer_data:
                    src = buffer_data[attr]
                    dst = getattr(combined_buffer, attr)
                    dst.extend(src)

            total_size += size

        if len(combined_buffer) == 0:
            return {"buffer_size": 0}

        # PPO 更新
        metrics = self.trainer.update(combined_buffer, show_progress=False)

        # 统计
        reward_stats = self.trainer.compute_buffer_stats(combined_buffer)
        metrics.update(reward_stats)
        metrics["buffer_size"] = len(combined_buffer)
        metrics["num_players"] = num_players
        metrics["iteration"] = self.iteration

        # 记录到 SwanLab
        if self.swanlab:
            self.swanlab.log(metrics, step=self.iteration)

        self.iteration += 1
        return metrics

    def save_checkpoint(self, path: str) -> None:
        """保存 checkpoint。"""
        model_state = self.model.state_dict()
        torch.save({
            "model_state_dict": model_state,
            "config": self.config.model,
            "n_planets": 40,
            "iteration": self.iteration,
        }, path)

    def shutdown(self) -> None:
        """关闭 SwanLab。"""
        if self.swanlab:
            self.swanlab.finish()


@ray.remote
class RolloutActor:
    """
    Rollout Actor - 并行执行 rollout。

    每个 actor 使用部分 GPU，多个 actor 共享 GPU 资源。
    """

    def __init__(
        self,
        worker_id: int,
        config_path: str,
        device: str = "cuda:0",
    ):
        self.worker_id = worker_id
        self.device = device

        print(f"[RolloutActor {worker_id}] 初始化 on {device}...")

        # 加载配置
        config_path = Path(config_path)
        if config_path.exists():
            self.config = AppConfig.from_yaml(config_path)
        else:
            self.config = AppConfig()

        # 设置随机种子
        seed = 42 + worker_id
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # 初始化模型
        self.model = OrbitWarsModel(
            self.config.model,
            n_planets=40,
            max_players=4,
        ).to(self.device)

        # 初始化特征工程和奖励计算器
        self.feature_engineer = FeatureEngineer(
            board_size=self.config.environment.board_size,
            sun_radius=self.config.environment.sun_radius,
            max_speed=self.config.environment.ship_speed,
            max_turns=self.config.environment.episode_steps,
        )

        self.reward_calculator = RewardCalculator(
            config=RewardConfig(
                terminal_weight=self.config.reward.terminal_weight,
                intermediate_weight=self.config.reward.intermediate_weight,
                capture_reward_weight=self.config.reward.capture_reward_weight,
                loss_penalty_weight=self.config.reward.loss_penalty_weight,
                defense_reward_weight=self.config.reward.defense_reward_weight,
                transit_cost_weight=self.config.reward.transit_cost_weight,
                production_advantage_weight=self.config.reward.production_advantage_weight,
                comet_roi_weight=self.config.reward.comet_roi_weight,
            ),
            max_turns=self.config.environment.episode_steps,
        )

        # RolloutWorker
        use_gpu_features = getattr(self.config.training, 'use_gpu_features', False)
        self.worker = RolloutWorker(
            self.model,
            self.feature_engineer,
            self.reward_calculator,
            device=self.device,
            use_gpu_features=use_gpu_features,
        )

        pool_config = OpponentPoolConfig(
            pool_size=self.config.self_play.pool_size,
            sample_expert_ratio=self.config.self_play.sample_expert_ratio,
            sample_heuristic_ratio=self.config.self_play.sample_heuristic_ratio,
            sample_checkpoint_ratio=self.config.self_play.sample_checkpoint_ratio,
            checkpoint_latest_ratio=self.config.self_play.checkpoint_latest_ratio,
            checkpoint_best_ratio=self.config.self_play.checkpoint_best_ratio,
            checkpoint_random_ratio=self.config.self_play.checkpoint_random_ratio,
            add_to_pool_win_rate=self.config.self_play.add_to_pool_win_rate,
            diversity_threshold=self.config.self_play.diversity_threshold,
        )
        self.opponent_pool = OpponentPool(config=pool_config, pool_dir="checkpoints/pool")
        self.opponent_pool.load_index()
        self.opponent_pool.register_default_experts()
        self._refresh_opponent_checkpoints()

        print(f"[RolloutActor {worker_id}] 初始化完成")

    def _refresh_opponent_checkpoints(self) -> None:
        self.opponent_pool.add_checkpoint_dir("training/checkpoints", elo=580, generation=0)
        self.opponent_pool.add_checkpoint_dir(
            "checkpoints",
            patterns=("model_iter_*.pt", "model_final.pt"),
            elo=600,
            generation=0,
        )

    def rollout(
        self,
        model_state: Dict,
        num_games: int,
        num_players: int,
        temperature: float,
    ) -> Dict:
        """
        执行 rollout。

        Args:
            model_state: 当前模型权重
            num_games: 游戏数量
            num_players: 玩家数
            temperature: 采样温度

        Returns:
            buffer 数据字典
        """
        import time
        print(f"[DEBUG][RolloutActor {self.worker_id}] rollout() 开始，参数: num_games={num_games}, num_players={num_players}, temperature={temperature}")

        # 加载模型权重
        t_load = time.time()
        self.model.load_state_dict(model_state)
        self.model.eval()
        print(f"[DEBUG][RolloutActor {self.worker_id}] 模型加载完成，耗时 {time.time()-t_load:.2f}s")

        # 执行 rollout
        t_rollout = time.time()
        print(f"[DEBUG][RolloutActor {self.worker_id}] 开始 parallel_rollout()...")
        self._refresh_opponent_checkpoints()
        buffer = parallel_rollout(
            self.worker,
            num_games=num_games,
            num_players=num_players,
            temperature=temperature,
            show_progress=False,
            opponent_pool=self.opponent_pool,
        )
        print(f"[DEBUG][RolloutActor {self.worker_id}] parallel_rollout() 完成，耗时 {time.time()-t_rollout:.2f}s，buffer size={len(buffer)}")

        # 转换为可序列化的字典
        buffer_data = {
            'planet_features': buffer.planet_features,
            'fleet_features': buffer.fleet_features,
            'global_features': buffer.global_features,
            'owned_masks': buffer.owned_masks,
            'enemy_masks': buffer.enemy_masks,
            'num_players_list': buffer.num_players_list,
            'planet_ships_list': buffer.planet_ships_list,
            'target_indices': buffer.target_indices,
            'num_ships_actual': buffer.num_ships_actual,
            'log_probs': buffer.log_probs,
            'values': buffer.values,
            'rewards': buffer.rewards,
            'dones': buffer.dones,
            'opp_target_indices': buffer.opp_target_indices,
            'size': len(buffer),
        }

        return buffer_data


# ==================== 主训练逻辑 ====================

def main():
    """主训练函数。"""

    import argparse

    parser = argparse.ArgumentParser(description="Ray 分布式训练")
    parser.add_argument("--rollout-workers", type=int, default=None,
                        help="Rollout worker 数量 (覆盖配置文件)")
    parser.add_argument("--gpus-per-worker", type=float, default=None,
                        help="每个 worker 的 GPU 数量 (覆盖配置文件)")
    parser.add_argument("--trainer-gpus", type=float, default=None,
                        help="Trainer 的 GPU 数量 (覆盖配置文件)")
    parser.add_argument("--games-per-rollout", type=int, default=None,
                        help="每次 rollout 的游戏数 (覆盖配置文件)")
    parser.add_argument("--max-iterations", type=int, default=None,
                        help="最大迭代次数 (覆盖配置文件)")
    parser.add_argument("--config", default="config/default.yaml",
                        help="配置文件路径（相对于脚本目录或项目根目录）")
    parser.add_argument("--no-ray-redis", action="store_true",
                        help="不使用 Ray Redis (单机模式)")

    args = parser.parse_args()

    # 加载配置（支持多种路径格式）
    config_path = Path(args.config)

    # 如果配置文件不存在，尝试相对于脚本目录的路径
    if not config_path.exists():
        script_dir = Path(__file__).parent
        config_path = script_dir / args.config

    # 如果还是不存在，尝试相对于项目根目录的路径
    if not config_path.exists():
        project_root = Path(__file__).parent.parent
        config_path = project_root / args.config

    if config_path.exists():
        config = AppConfig.from_yaml(config_path)
        print(f"✓ 加载配置: {config_path}")
    else:
        print(f"⚠️  配置文件不存在: {args.config}")
        print(f"   尝试的路径: {config_path}")
        print(f"   使用默认配置")
        config = AppConfig()

    # 命令行参数覆盖配置
    num_workers = args.rollout_workers or config.ray.num_rollout_workers
    gpus_per_worker = args.gpus_per_worker or config.ray.num_gpus_per_worker
    trainer_gpus = args.trainer_gpus or config.ray.trainer_num_gpus
    games_per_rollout = args.games_per_rollout or config.ray.games_per_rollout
    max_iterations = args.max_iterations or config.training.max_iterations

    # 禁用 SwanLab 交互式提示 (必须在所有进程启动前设置)
    os.environ['SWANLAB_NO_INTERACTIVE'] = '1'
    os.environ['SWANLAB_DISABLE_INTERACTIVE'] = '1'

    # 获取 SwanLab API key 并设置环境变量
    swanlab_key = os.environ.get('SWANLAB_API_KEY')
    if not swanlab_key:
        key_file = Path(__file__).parent / "config" / "swanlab_key.txt"
        if key_file.exists():
            swanlab_key = key_file.read_text().strip()
            os.environ['SWANLAB_API_KEY'] = swanlab_key

    # 打印配置
    print("=" * 60)
    print("Ray 分布式训练")
    print("=" * 60)
    print(f"可见GPU: {os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")
    print(f"Rollout Workers: {num_workers}")
    print(f"每个 Worker GPU: {gpus_per_worker}")
    print(f"Trainer GPU: {trainer_gpus}")
    print(f"每次 Rollout 游戏数: {games_per_rollout}")
    print(f"总并行游戏: {num_workers * games_per_rollout}")
    print(f"最大迭代: {max_iterations}")
    print("=" * 60)
    print()

    # 检查 GPU 数量
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        print(f"检测到 {num_gpus} 个 GPU")
        total_gpus_needed = num_workers * gpus_per_worker + trainer_gpus
        print(f"所需 GPU 总量: {total_gpus_needed:.2f}")
        if total_gpus_needed > num_gpus:
            print(f"⚠️  警告: 所需 GPU ({total_gpus_needed:.2f}) 超过可用 GPU ({num_gpus})")
        print()
    else:
        print("⚠️  警告: 未检测到 GPU，将使用 CPU 模式")
        gpus_per_worker = 0
        trainer_gpus = 0

    # 初始化 Ray
    print("[Ray] 初始化...")

    # 配置 Ray 临时目录到 /data2 分区（避免根分区空间不足）
    import tempfile
    ray_temp_dir = Path("/data2/solo/Orbit-Wars/training/.ray_temp")
    ray_temp_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Ray] 临时目录: {ray_temp_dir}")

    ray_init_kwargs = {
        "num_gpus": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "ignore_reinit_error": True,
        "_temp_dir": str(ray_temp_dir),  # 设置临时目录
        "include_dashboard": False,  # 禁用 Dashboard（Python 3.12 兼容性问题）
    }

    if args.no_ray_redis:
        # 单机模式，不启动 Redis
        ray_init_kwargs["_memory"] = 10**9  # 1GB

    ray.init(**ray_init_kwargs)
    print(f"[Ray] 初始化完成: {ray.cluster_resources()}")

    # 创建 Trainer Actor
    print("\n[Ray] 创建 TrainingActor...")
    trainer_actor = TrainingActor.options(
        num_gpus=trainer_gpus,
    ).remote(
        config_path=args.config,
        device="cuda:0" if trainer_gpus > 0 else "cpu",
    )

    # 创建 Rollout Actors
    print(f"[Ray] 创建 {num_workers} 个 RolloutActors...")
    rollout_actors = []
    for i in range(num_workers):
        # 每个 actor 可以使用不同 GPU（Ray 自动调度）
        # 这里我们让 Ray 自动分配，但可以通过 resources 指定
        actor = RolloutActor.options(
            num_gpus=gpus_per_worker,
        ).remote(
            worker_id=i,
            config_path=args.config,
            device="cuda:0" if gpus_per_worker > 0 else "cpu",
        )
        rollout_actors.append(actor)

    print(f"[Ray] 所有 Actors 创建完成\n")

    # 训练循环
    print("=" * 60)
    print("开始训练")
    print("=" * 60)

    try:
        # 创建迭代进度条
        pbar = tqdm(range(max_iterations), desc="[Training]", unit="iter")

        for iteration in pbar:
            t0 = time.time()

            # 决定 2 人 / 4 人局
            num_players = 2 if random.random() < config.self_play.two_player_prob else 4
            temperature = max(1.0 - iteration * 0.001, 0.3)

            pbar.set_description(f"[{iteration+1}/{max_iterations}] Rollout ({num_players}人局)")
            pbar.set_postfix_str(f"等待 {len(rollout_actors)} workers...")

            # 获取当前模型
            print(f"\n[DEBUG][主循环] Iteration {iteration}: 开始获取模型状态...")
            t_get_model = time.time()
            model_state_ref = trainer_actor.get_model_state.remote()
            model_state = ray.get(model_state_ref)
            print(f"[DEBUG][主循环] 模型状态获取完成，耗时 {time.time()-t_get_model:.2f}s")

            # 并行执行所有 rollouts
            print(f"[DEBUG][主循环] 提交 {len(rollout_actors)} 个 rollout 任务...")
            pbar.set_postfix_str(f"Rollout 启动中...")
            t_submit = time.time()
            rollout_refs = [
                actor.rollout.remote(model_state, games_per_rollout, num_players, temperature)
                for actor in rollout_actors
            ]
            print(f"[DEBUG][主循环] 任务提交完成，耗时 {time.time()-t_submit:.2f}s")

            # 等待所有 rollout 完成
            print(f"[DEBUG][主循环] 等待 {len(rollout_refs)} 个 rollouts 完成...")
            rollout_start = time.time()
            buffers = ray.get(rollout_refs)
            rollout_time = time.time() - rollout_start
            print(f"[DEBUG][主循环] 所有 rollouts 完成，总耗时 {rollout_time:.2f}s")

            # 统计实际收集的数据
            print(f"[DEBUG][主循环] 统计各 actor 返回的数据:")
            for i, b in enumerate(buffers):
                size = b.get('size', 0)
                print(f"[DEBUG][主循环]   Actor {i}: {size} 条数据")
            total_samples = sum(b.get('size', 0) for b in buffers)
            print(f"[DEBUG][主循环] 总计: {total_samples} 条数据")
            pbar.set_postfix_str(f"Rollout: {total_samples}条, {rollout_time:.1f}s | PPO更新...")

            # PPO 更新
            update_start = time.time()
            metrics_ref = trainer_actor.update.remote(buffers, num_players)
            metrics = ray.get(metrics_ref)
            update_time = time.time() - update_start

            if metrics.get('buffer_size', 0) == 0:
                pbar.write(f"[Iter {iteration}] Empty buffer, skipping")
                continue

            total_time = time.time() - t0

            # 更新进度条后缀
            reward_val = metrics.get('mean_reward', 0)
            pbar.set_postfix_str(f"Reward:{reward_val:.3f} Total:{total_time:.1f}s R:{rollout_time:.1f}s U:{update_time:.1f}s")

            # 定期保存 checkpoint
            if iteration > 0 and iteration % config.training.save_interval == 0:
                ckpt_path = f"checkpoints/model_iter_{iteration}.pt"
                trainer_actor.save_checkpoint.remote(ckpt_path)
                pbar.write(f"[Iter {iteration}] Checkpoint saved: {ckpt_path}")

    except KeyboardInterrupt:
        print("\n训练被中断")

    finally:
        # 清理
        print("\n[Ray] 关闭...")
        trainer_actor.shutdown.remote()
        ray.shutdown()
        print("[Ray] 已关闭")


if __name__ == "__main__":
    main()
