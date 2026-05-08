"""生成专家演示数据脚本。

用于生成大规模专家演示数据，供模型训练使用。

用法:
    # 生成 100 局对局（约 200,000 样本）
    python generate_expert_data.py --num_episodes 100

    # 生成 500 局对局，每 50 局保存一个文件
    python generate_expert_data.py --num_episodes 500 --episodes_per_file 50

    # 生成到指定目录
    python generate_expert_data.py --num_episodes 200 --save_dir data/expert_data_v1
"""

import argparse
from pathlib import Path

from training.expert import ExpertDataGenerator, print_statistics


def main():
    parser = argparse.ArgumentParser(
        description="生成 Orbit Wars 专家演示数据"
    )
    parser.add_argument(
        "--num_episodes",
        type=int,
        default=100,
        help="生成的对局数量 (默认: 100)",
    )
    parser.add_argument(
        "--episodes_per_file",
        type=int,
        default=50,
        help="每个文件保存的对局数量 (默认: 50)",
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

    args = parser.parse_args()

    print("=" * 60)
    print("Orbit Wars 专家演示数据生成器")
    print("=" * 60)
    print(f"\n配置:")
    print(f"  对局数量: {args.num_episodes}")
    print(f"  每文件对局数: {args.episodes_per_file}")
    print(f"  玩家数量: {args.num_players}")
    print(f"  保存目录: {args.save_dir}")
    print(f"  文件前缀: {args.prefix}")
    print()

    # 创建数据生成器
    generator = ExpertDataGenerator(
        num_players=args.num_players,
        save_dir=args.save_dir,
    )

    # 生成数据
    file_paths = generator.generate_dataset(
        num_episodes=args.num_episodes,
        episodes_per_file=args.episodes_per_file,
        prefix=args.prefix,
    )

    # 显示文件信息
    print(f"\n生成的文件:")
    total_size = 0
    for fp in file_paths:
        size_mb = fp.stat().st_size / (1024 * 1024)
        total_size += size_mb
        print(f"  - {fp.name} ({size_mb:.2f} MB)")

    print(f"\n总大小: {total_size:.2f} MB")

    # 估算样本数量（基于每局约 2000 时间步）
    estimated_samples = args.num_episodes * 2000 * args.num_players
    print(f"估计样本数: {estimated_samples:,}")

    print("\n✓ 数据生成完成!")
    print(f"\n使用方法:")
    print(f"  from training.expert import load_expert_dataset")
    print(f"  dataset = load_expert_dataset('{args.save_dir}')")
    print(f"  # 训练模型...")
    print()


if __name__ == "__main__":
    main()
