"""测试 JSONL 格式数据生成。

生成少量数据并验证 JSONL 格式。
"""

from training.expert import ExpertDataGenerator, print_statistics


def test_jsonl_generation():
    """测试生成 JSONL 格式数据。"""
    print("=" * 60)
    print("测试 JSONL 格式数据生成")
    print("=" * 60)

    # 创建数据生成器
    generator = ExpertDataGenerator(
        num_players=4,
        save_dir="data/test_jsonl",
        save_format="jsonl",
    )

    # 生成 1 局对局
    print("\n生成 1 局对局（JSONL 格式）...")
    file_paths = generator.generate_dataset(
        num_episodes=1,
        episodes_per_file=1,
        prefix="test_jsonl",
    )

    print(f"\n✓ 生成了 {len(file_paths)} 个文件")

    # 显示文件内容（前几行）
    for fp in file_paths:
        print(f"\n文件: {fp.name}")
        print(f"大小: {fp.stat().st_size / 1024:.2f} KB")

        # 读取并显示前几行
        print(f"\n前 3 行内容:")
        with open(fp, "r") as f:
            for i, line in enumerate(f):
                if i >= 3:
                    break
                print(f"  [{i}] {line[:200]}...")

    # 验证加载
    print(f"\n验证数据加载...")
    dataset = generator.load_trajectories(file_paths[0])
    print(f"✓ 成功加载 {len(dataset)} 个样本")

    # 显示第一个样本
    if dataset:
        sample = dataset[0]
        print(f"\n第一个样本:")
        print(f"  player_id: {sample['player_id']}")
        print(f"  step: {sample['step']}")
        print(f"  observation keys: {list(sample['observation'].keys())}")
        print(f"  actions: {sample['actions']}")
        print(f"  reward: {sample['reward']}")

    print("\n✓ JSONL 格式测试通过!")


if __name__ == "__main__":
    test_jsonl_generation()
