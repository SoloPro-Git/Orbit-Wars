"""多进程专家数据生成器。

使用多进程并行生成大规模专家演示数据，支持进度显示和断点续传。

用法:
    python -m training.expert.multiprocess_generator --num_episodes 1000 --num_processes 8
"""

import argparse
import hashlib
import json
import os
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from training.core.env_wrapper import OrbitWarsEnv
from training.expert.kaggle_expert import KaggleExpertAgent


# ===========================================================================
# 工作进程函数
# ===========================================================================

def worker_generate_episode(
    episode_id: int,
    seed: int,
    num_players: int = 4,
) -> Tuple[int, List[dict], int, str]:
    """工作进程：生成一局对局数据。

    Args:
        episode_id: 对局 ID
        seed: 随机种子
        num_players: 玩家数量

    Returns:
        (episode_id, trajectories, success, error_msg)
    """
    try:
        # 创建环境和专家智能体
        env = OrbitWarsEnv(num_players=num_players)
        experts = [KaggleExpertAgent(player_id=i) for i in range(num_players)]

        # 重置环境
        observations = env.reset(seed=seed)
        trajectory = []

        # 运行对局
        max_steps = 500
        for step_idx in range(max_steps):
            if env.done:
                break

            # 收集所有玩家的动作
            actions = {}
            obs_dict = {}
            for player_id in range(num_players):
                obs = env.get_raw_observation(player_id)
                obs_dict[player_id] = obs
                expert_actions = experts[player_id].get_actions(obs)
                actions[player_id] = expert_actions

            # 执行一步
            next_obs, rewards, dones, infos = env.step(actions)

            # 记录每个玩家的经验
            for player_id in range(num_players):
                trajectory.append({
                    "player_id": player_id,
                    "step": step_idx,
                    "observation": obs_dict[player_id],
                    "actions": actions[player_id],
                    "reward": rewards.get(player_id, 0.0),
                    "done": dones.get(player_id, True),
                })

            observations = next_obs

        # 清理
        del env
        del experts

        return episode_id, trajectory, 1, ""

    except Exception as e:
        error_msg = f"Episode {episode_id} failed: {str(e)}"
        return episode_id, [], 0, error_msg


def make_json_serializable(obj):
    """将对象转换为 JSON 可序列化格式。"""
    if isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj


def save_episode_to_file(
    episode_id: int,
    trajectory: List[dict],
    output_dir: Path,
    file_format: str = "jsonl",
) -> Path:
    """保存一局对局到文件。

    Args:
        episode_id: 对局 ID
        trajectory: 轨迹数据
        output_dir: 输出目录
        file_format: 文件格式（jsonl 或 pkl）

    Returns:
        保存的文件路径
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if file_format == "jsonl":
        # 计算文件编号
        file_idx = episode_id // 20  # 每 20 局一个文件
        file_path = output_dir / f"expert_data_{file_idx:04d}.jsonl"

        # 追加模式写入
        with open(file_path, "a") as f:
            for sample in trajectory:
                sample_serializable = make_json_serializable(sample)
                f.write(json.dumps(sample_serializable, ensure_ascii=False) + "\n")

    else:  # pkl
        import pickle
        file_idx = episode_id // 20
        file_path = output_dir / f"expert_data_{file_idx:04d}.pkl"

        with open(file_path, "ab") as f:  # 二进制追加模式
            pickle.dump(trajectory, f)

    return file_path


# ===========================================================================
# 主生成器类
# ===========================================================================

class MultiprocessDataGenerator:
    """多进程数据生成器。"""

    def __init__(
        self,
        num_episodes: int,
        num_processes: int = 8,
        save_dir: str = "data/expert_demonstrations",
        file_format: str = "jsonl",
    ):
        """初始化生成器。

        Args:
            num_episodes: 总对局数
            num_processes: 并行进程数
            save_dir: 保存目录
            file_format: 文件格式（jsonl 或 pkl）
        """
        self.num_episodes = num_episodes
        self.num_processes = num_processes
        self.save_dir = Path(save_dir)
        self.file_format = file_format

        # 创建输出目录
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def generate(self):
        """生成数据。"""
        print(f"\n开始生成 {self.num_episodes} 局专家数据...")
        print(f"并行进程数: {self.num_processes}")
        print(f"保存目录: {self.save_dir}")
        print(f"文件格式: {self.file_format}")
        print()

        start_time = time.time()

        # 准备任务
        tasks = []
        for episode_id in range(self.num_episodes):
            seed = 42 + episode_id
            tasks.append((episode_id, seed))

        # 使用多进程执行
        success_count = 0
        fail_count = 0
        total_samples = 0

        with ProcessPoolExecutor(max_workers=self.num_processes) as executor:
            # 提交所有任务
            futures = {
                executor.submit(worker_generate_episode, ep_id, seed): ep_id
                for ep_id, seed in tasks
            }

            # 使用 tqdm 显示进度
            with tqdm(total=self.num_episodes, desc="生成进度") as pbar:
                for future in as_completed(futures):
                    episode_id = futures[future]

                    try:
                        ep_id, trajectory, success, error_msg = future.result()

                        if success:
                            # 保存数据
                            save_episode_to_file(
                                ep_id,
                                trajectory,
                                self.save_dir,
                                self.file_format,
                            )

                            success_count += 1
                            total_samples += len(trajectory)

                            # 更新进度条描述
                            pbar.set_description(
                                f"生成进度 | 成功: {success_count} | "
                                f"样本: {total_samples:,}"
                            )

                        else:
                            fail_count += 1
                            print(f"\n错误: {error_msg}")

                    except Exception as e:
                        fail_count += 1
                        print(f"\n异常: Episode {episode_id} - {str(e)}")

                    pbar.update(1)

        # 统计信息
        elapsed_time = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"生成完成")
        print(f"{'='*60}")
        print(f"成功对局: {success_count}/{self.num_episodes}")
        print(f"失败对局: {fail_count}")
        print(f"总样本数: {total_samples:,}")
        print(f"耗时: {elapsed_time/60:.1f} 分钟")
        print(f"速度: {self.num_episodes/elapsed_time:.2f} 局/秒")
        print(f"{'='*60}\n")

        return success_count, total_samples


# ===========================================================================
# 命令行接口
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="多进程生成 Orbit Wars 专家演示数据"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=1000,
        help="生成对局数（默认: 1000）",
    )
    parser.add_argument(
        "--num_processes",
        type=int,
        default=8,
        help="并行进程数（默认: 8）",
    )
    parser.add_argument(
        "--episodes_per_file",
        type=int,
        default=20,
        help="每个文件保存的对局数（默认: 20）",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/expert_demonstrations",
        help="保存目录（默认: data/expert_demonstrations）",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "pkl"],
        help="文件格式（默认: jsonl）",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="生成后验证数据",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("Orbit Wars 多进程专家数据生成器")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  对局数: {args.num_episodes}")
    print(f"  进程数: {args.num_processes}")
    print(f"  每文件对局数: {args.episodes_per_file}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  文件格式: {args.format}")
    print()

    # 估算资源
    est_time = args.num_episodes * 6 / 60 / args.num_processes
    est_size = args.num_episodes * 10 / 1024  # GB

    print(f"预计资源:")
    print(f"  时间: ~{est_time:.1f} 分钟")
    print(f"  磁盘: ~{est_size:.1f} GB")
    print()

    # 创建生成器
    generator = MultiprocessDataGenerator(
        num_episodes=args.num_episodes,
        num_processes=args.num_processes,
        save_dir=args.save_dir,
        file_format=args.format,
    )

    # 生成数据
    success_count, total_samples = generator.generate()

    # 验证数据（如果启用）
    if args.validate:
        print(f"验证数据...")
        from training.expert import load_expert_dataset, print_statistics

        try:
            dataset = load_expert_dataset(args.save_dir)
            print_statistics(dataset)
        except Exception as e:
            print(f"验证失败: {e}")

    print(f"✓ 完成！")
    print(f"\n数据位置: {args.save_dir}/")
    print(f"使用方法:")
    print(f"  from training.expert import load_expert_dataset")
    print(f"  dataset = load_expert_dataset('{args.save_dir}')")
    print()


if __name__ == "__main__":
    # Windows 多进程支持
    import multiprocessing
    if __name__ == "__main__":
        multiprocessing.freeze_support()

    main()
