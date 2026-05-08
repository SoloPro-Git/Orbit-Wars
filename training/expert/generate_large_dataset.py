"""大规模专家数据生成脚本。

生成 100,000+ 条专家演示数据，用于模型预训练。

计算:
- 每局对局约 2000 时间步 × 4 玩家 = 8000 样本
- 100,000 样本 ÷ 8000 样本/局 ≈ 13 局
- 建议生成 50-100 局以确保数据多样性

用法:
    # 生成 100,000 条数据（约 13 局）
    python -m training.expert.generate_large_dataset --num_samples 100000

    # 生成 500,000 条数据（约 63 局）
    python -m training.expert.generate_large_dataset --num_samples 500000

    # 生成指定局数
    python -m training.expert.generate_large_dataset --num_episodes 50
"""

import argparse
import time
from pathlib import Path

from tqdm import tqdm

from training.expert import ExpertDataGenerator, print_statistics


def calculate_episodes_for_samples(num_samples: int) -> int:
    """根据目标样本数计算需要的对局数。

    每局对局约 2000 时间步 × 4 玩家 = 8000 样本。
    """
    samples_per_episode = 8000
    episodes = (num_samples + samples_per_episode - 1) // samples_per_episode
    # 至少生成 5 局
    return max(5, episodes)


def main():
    parser = argparse.ArgumentParser(
        description="大规模生成 Orbit Wars 专家演示数据"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100000,
        help="目标样本数量 (默认: 100,000)",
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=None,
        help="指定对局数量（覆盖 num_samples）",
    )
    parser.add_argument(
        "--episodes_per_file",
        type=int,
        default=10,
        help="每个文件保存的对局数量 (默认: 10)",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="data/expert_demonstrations",
        help="数据保存目录 (默认: data/expert_demonstrations)",
    )
    parser.add_argument(
        "--num_players",
        type=int,
        default=4,
        choices=[2, 4],
        help="玩家数量 (默认: 4)",
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="expert_data",
        help="文件名前缀 (默认: expert_data)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="jsonl",
        choices=["jsonl", "pkl"],
        help="数据保存格式 (默认: jsonl)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="生成后验证数据质量和统计信息",
    )

    args = parser.parse_args()

    # 确定对局数量
    if args.num_episodes is not None:
        num_episodes = args.num_episodes
        estimated_samples = num_episodes * 8000
    else:
        num_episodes = calculate_episodes_for_samples(args.num_samples)
        estimated_samples = num_episodes * 8000

    print("=" * 80)
    print("Orbit Wars 大规模专家数据生成器")
    print("=" * 80)
    print(f"\n配置:")
    print(f"  目标样本数: {args.num_samples:,}")
    print(f"  生成对局数: {num_episodes}")
    print(f"  估计样本数: {estimated_samples:,}")
    print(f"  每文件对局数: {args.episodes_per_file}")
    print(f"  玩家数量: {args.num_players}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  文件前缀: {args.prefix}")
    print(f"\n预计时间: {num_episodes * 6 / 60:.1f} 分钟")
    print()

    # 确认开始
    if estimated_samples > 500000:
        confirm = input(f"将生成约 {estimated_samples:,} 条数据，确认继续? (y/n): ")
        if confirm.lower() != 'y':
            print("已取消")
            return

    # 创建数据生成器
    generator = ExpertDataGenerator(
        num_players=args.num_players,
        save_dir=args.save_dir,
        save_format=args.format,
    )

    # 生成数据
    start_time = time.time()

    file_paths = generator.generate_dataset(
        num_episodes=num_episodes,
        episodes_per_file=args.episodes_per_file,
        prefix=args.prefix,
    )

    elapsed_time = time.time() - start_time

    # 显示文件信息
    print(f"\n生成的文件:")
    total_size = 0
    for fp in file_paths:
        size_mb = fp.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {fp.name} ({size_mb:.2f} MB)")

    print(f"\n统计信息:")
    print(f"  总大小: {total_size:.2f} MB ({total_size / 1024:.2f} GB)")
    print(f"  生成时间: {elapsed_time / 60:.1f} 分钟")
    print(f"  平均速度: {num_episodes / elapsed_time:.2f} 局/秒")

    # 验证数据（如果启用）
    if args.validate:
        print(f"\n正在验证数据...")
        from training.expert import ExpertDataset

        dataset = ExpertDataset(data_dir=args.save_dir)
        print_statistics(dataset)

        # 检查数据质量
        print(f"\n数据质量检查:")

        # 检查每个玩家都有数据
        player_ids = set(s["player_id"] for s in dataset.data)
        print(f"  ✓ 玩家覆盖: {player_ids}")

        # 检查动作分布
        steps_with_actions = sum(1 for s in dataset.data if len(s["actions"]) > 0)
        action_rate = steps_with_actions / len(dataset.data) * 100
        print(f"  ✓ 有动作的步数: {steps_with_actions:,} ({action_rate:.1f}%)")

        # 检查奖励分布
        rewards = [s["reward"] for s in dataset.data]
        print(f"  ✓ 奖励范围: [{min(rewards):.2f}, {max(rewards):.2f}]")

    print("\n✓ 数据生成完成!")
    print(f"\n使用方法:")
    print(f"  from training.expert import load_expert_dataset")
    print(f"  dataset = load_expert_dataset('{args.save_dir}')")
    print(f"  # 训练模型...")
    print()


if __name__ == "__main__":
    main()
